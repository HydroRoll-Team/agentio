import os
import re
import json
import asyncio
import hashlib
from typing import List, Dict, Any

import httpx
import chromadb
from chromadb.config import Settings
from prompt_toolkit import prompt
from pypdf import PdfReader


class OllamaClient:
    def __init__(
        self, api_base: str = "http://localhost:11434", model_name: str = "gemma3:12b"
    ):
        self.api_base = api_base.rstrip("/")
        self.model_name = model_name

    async def generate(self, _prompt: str) -> str:
        """非流式：一次性返回完整 response"""
        url = f"{self.api_base}/api/generate"
        payload = {"model": self.model_name, "prompt": _prompt, "stream": False}
        async with httpx.AsyncClient() as client:
            r = await client.post(url, json=payload, timeout=300)
            r.raise_for_status()
            return r.json()["response"]

    async def stream_generate(self, _prompt: str):
        """
        流式：异步生成器，逐 token yield
        Ollama 返回 NDJSON：一行一个 JSON
        """
        url = f"{self.api_base}/api/generate"
        payload = {"model": self.model_name, "prompt": _prompt, "stream": True}

        # timeout=None：避免长输出被 httpx 超时中断
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("POST", url, json=payload) as r:
                r.raise_for_status()
                async for line in r.aiter_lines():
                    if not line:
                        continue
                    data = json.loads(line)
                    if "response" in data and data["response"]:
                        yield data["response"]
                    if data.get("done"):
                        break

    async def embed(self, embedding_model: str, text: str) -> List[float]:
        """
        Ollama embeddings API:
        POST /api/embeddings
        { "model": "...", "prompt": "..." }
        """
        url = f"{self.api_base}/api/embeddings"
        payload = {"model": embedding_model, "prompt": text}
        async with httpx.AsyncClient(timeout=300) as client:
            r = await client.post(url, json=payload)
            r.raise_for_status()
            data = r.json()
            return data["embedding"]


def read_pdf(path: str) -> str:
    reader = PdfReader(path)
    texts = []
    for page in reader.pages:
        t = page.extract_text() or ""
        texts.append(t)
    return "\n".join(texts)


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def load_documents(root_dir: str) -> List[Dict[str, Any]]:
    """
    return: [{"source": "...", "text": "..."}]
    """
    docs = []
    for base, _, files in os.walk(root_dir):
        for fn in files:
            fp = os.path.join(base, fn)
            ext = os.path.splitext(fn)[1].lower()

            if ext == ".pdf":
                text = read_pdf(fp)
            elif ext in [".md", ".markdown", ".txt"]:
                text = read_text(fp)
            else:
                continue

            text = (text or "").strip()
            if text:
                docs.append({"source": fp, "text": text})
    return docs


def normalize_whitespace(s: str) -> str:
    s = s.replace("\u00a0", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def chunk_text(text: str, chunk_size: int = 900, chunk_overlap: int = 150) -> List[str]:
    """
    分块：
    1. 先按空行分段
    2. 合并到 chunk_size 长度
    3. overlap 防止断句丢信息

    chunks = [
        "ABCDEE",
        "EE\nFGHIJ",
        "IJ\nKLMNO",
        ...
    ]
    """
    text = normalize_whitespace(text)
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    chunks = []
    buf = ""

    def flush():
        nonlocal buf
        if buf.strip():
            chunks.append(buf.strip())
        buf = ""

    for p in paras:
        if not buf:
            buf = p
        elif len(buf) + 2 + len(p) <= chunk_size:
            buf = buf + "\n\n" + p
        else:
            flush()
            # 段落超长：硬切
            while len(p) > chunk_size:
                chunks.append(p[:chunk_size])
                p = p[chunk_size - chunk_overlap :]
            buf = p

    flush()

    # 每个 chunk 前面附上一点上一块尾巴
    if chunk_overlap > 0 and len(chunks) > 1:
        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            prev_tail = overlapped[-1][-chunk_overlap:]
            overlapped.append(prev_tail + "\n" + chunks[i])
        chunks = overlapped

    return chunks


def validate_collection_name(name: str):
    # 3-512 字符, [a-zA-Z0-9._-]
    if not re.match(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{1,510}[a-zA-Z0-9]$", name):
        raise ValueError(f"无效的 collection_name='{name}'")


def make_id(source: str, chunk_index: int, chunk_text: str) -> str:
    h = hashlib.sha1(
        (source + str(chunk_index) + chunk_text).encode("utf-8", errors="ignore")
    ).hexdigest()
    return h


class ChromaRAG:
    def __init__(
        self, persist_dir: str = "./chroma_db", collection_name: str = "kb_docs_v1"
    ):
        validate_collection_name(collection_name)
        self.client = chromadb.PersistentClient(
            path=persist_dir, settings=Settings(anonymized_telemetry=False)
        )
        self.col = self.client.get_or_create_collection(name=collection_name)

    def count(self) -> int:
        return self.col.count()

    def upsert(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: List[Dict[str, Any]],
    ):
        self.col.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    def query(self, query_embedding: List[float], n_results: int = 5) -> Dict[str, Any]:
        return self.col.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )


def build_rag_prompt(user_question: str, contexts: List[Dict[str, Any]]) -> str:
    ctx_blocks = []
    for i, c in enumerate(contexts, 1):
        src = c.get("source", "unknown")
        idx = c.get("chunk_index", -1)
        ctx_blocks.append(
            f"[片段 {i} | {os.path.basename(src)} | chunk={idx}]\n{c['text']}"
        )
    ctx = "\n\n".join(ctx_blocks)

    return f"""你是一个严谨的资料问答助手。请只根据“给定资料片段”回答问题；资料没有覆盖就说不知道，不要编。

    【给定资料片段】
    {ctx}

    【用户问题】
    {user_question}

    【回答要求】
    - 用中文回答
    - 尽量引用具体片段内容（可点名“片段1/2/3”，然后在回答完毕后附上片段原文）
    - 如果资料中没有答案，请明确回复“资料中未覆盖该问题”
    - 不要胡编
    """


async def ingest_folder(
    rag: ChromaRAG,
    ollama: OllamaClient,
    embedding_model: str,
    docs_dir: str,
    chunk_size: int = 900,
    chunk_overlap: int = 150,
    batch_size: int = 128,
    concurrency: int = 256,
):
    docs = load_documents(docs_dir)
    if not docs:
        print(f"未找到可导入文档：{docs_dir}")
        return

    print(f"发现 {len(docs)} 个文档，开始切块+入库…（GPU加速，并发={concurrency}，batch_size={batch_size}）")

    # 移除并发限制，让所有请求同时提交到 GPU
    async def embed_one(text: str) -> List[float]:
        return await ollama.embed(embedding_model, text)

    ids, metadatas, documents, embeddings = [], [], [], []

    async def flush_batch():
        nonlocal ids, metadatas, documents, embeddings
        if not ids:
            return
        rag.upsert(
            ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas
        )
        print(f"已写入 {len(ids)} chunks，当前库总量：{rag.count()}")
        ids, metadatas, documents, embeddings = [], [], [], []

    # 收集所有 chunks（也可以按文档分批，避免一次性太大）
    pending = []

    for d in docs:
        source = d["source"]
        chunks = chunk_text(
            d["text"], chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

        for ci, ch in enumerate(chunks):
            _id = make_id(source, ci, ch)

            ids.append(_id)
            documents.append(ch)
            metadatas.append({"source": source, "chunk_index": ci})

            pending.append(asyncio.create_task(embed_one(ch)))

            # 每 batch_size 个就 gather 一次并写入
            if len(pending) >= batch_size:
                embs = await asyncio.gather(*pending)
                embeddings.extend(embs)
                pending.clear()
                await flush_batch()

    if pending:
        embs = await asyncio.gather(*pending)
        embeddings.extend(embs)
        pending.clear()
        await flush_batch()

    print("导入完成。")


async def answer_with_rag_stream(
    rag: ChromaRAG,
    ollama: OllamaClient,
    embedding_model: str,
    question: str,
    top_k: int = 5,
):
    # 1) embed question
    q_emb = await ollama.embed(embedding_model, question)

    # 2) retrieve
    result = rag.query(q_emb, n_results=top_k)
    docs = (result.get("documents") or [[]])[0]
    metas = (result.get("metadatas") or [[]])[0]
    dists = (result.get("distances") or [[]])[0]

    contexts = []
    for text, meta, dist in zip(docs, metas, dists):
        contexts.append(
            {
                "text": text,
                "source": meta.get("source"),
                "chunk_index": meta.get("chunk_index"),
                "distance": dist,
            }
        )

    rag_prompt = build_rag_prompt(question, contexts)

    async for token in ollama.stream_generate(rag_prompt):
        yield token


async def main():
    embedding_model = "nomic-embed-text"
    docs_dir = "./knowledge"

    rag = ChromaRAG(persist_dir="./chroma_db", collection_name="kb_docs_v1")
    ollama = OllamaClient("http://localhost:11434")

    loop = asyncio.get_event_loop()

    print("命令：")
    print("  /ingest   -> 导入 ./knowledge 下的 PDF/MD 到 Chroma")
    print("  /count    -> 查看向量库条目数")
    print("  /exit     -> 退出")

    while True:
        user_input = await loop.run_in_executor(None, lambda: prompt("User: "))
        cmd = user_input.strip()

        if cmd.lower() in {"exit", "quit", "/exit"}:
            break

        if cmd == "/count":
            print(f"Chroma count = {rag.count()}")
            continue

        if cmd == "/ingest":
            await ingest_folder(
                rag=rag,
                ollama=ollama,
                embedding_model=embedding_model,
                docs_dir=docs_dir,
            )
            continue
        
        print("LLM: ", end="", flush=True)
        async for token in answer_with_rag_stream(
            rag=rag,
            ollama=ollama,
            embedding_model=embedding_model,
            question=user_input,
            top_k=5,
        ):
            print(token, end="", flush=True)
        print("\n")


if __name__ == "__main__":
    asyncio.run(main())
