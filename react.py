"""
>>> rag = ChromaRAG(persist_dir="./chroma_db", collection_name="kb_docs_v1")
>>> ollama = OllamaClient("http://localhost:11434")
>>> agent = ReActAgent(rag=rag, ollama=ollama)
>>> answer = agent.run("什么是 Python?")
"""

import os
import re
import json
import asyncio
import hashlib
from typing import List, Dict, Any, Optional

import httpx
import chromadb
from chromadb.config import Settings
from prompt_toolkit import prompt
from pypdf import PdfReader

from mcp_client import MCPClientManager


class OllamaClient:
    """
    Ollama API 客户端

    Example:
        >>> client = OllamaClient("http://localhost:11434")
        >>> response = client.generate("你好")
        >>> for token in client.stream_generate("你好"):
        ...     print(token)
        >>> embedding = client.embed("Hello world")
    """

    def __init__(
        self,
        api_base: str = "http://localhost:11434",
        generate_model: str = "gemma3:4b",
        embed_model: str = "nomic-embed-text",
    ):
        self.api_base = api_base.rstrip("/")
        self.generate_model = generate_model
        self.embed_model = embed_model

    def generate(self, _prompt: str, model: Optional[str] = None, stop: Optional[List[str]] = None) -> str:
        """
        非流式文本生成

        Args:
            _prompt: 输入提示词
            model: 可选的模型名称，默认使用初始化时指定的模型
            stop: 停止词列表
        """
        url = f"{self.api_base}/api/generate"
        payload = {
            "model": model or self.generate_model,
            "prompt": _prompt,
            "stream": False,
            "options": {"stop": stop} if stop else {}
        }
        with httpx.Client(timeout=300) as client:
            r = client.post(url, json=payload)
            r.raise_for_status()
            return r.json().get("response", "")

    def stream_generate(self, _prompt: str, model: Optional[str] = None):
        """流式文本生成"""
        url = f"{self.api_base}/api/generate"
        payload = {"model": model or self.generate_model, "prompt": _prompt, "stream": True}
        with httpx.Client(timeout=None) as client:
            with client.stream("POST", url, json=payload) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if not line:
                        continue
                    data = json.loads(line)
                    if data.get("response"):
                        yield data["response"]
                    if data.get("done"):
                        break

    def embed(self, text: str, model: Optional[str] = None) -> List[float]:
        """生成文本嵌入向量"""
        url = f"{self.api_base}/api/embeddings"
        payload = {"model": model or self.embed_model, "prompt": text}
        with httpx.Client(timeout=300) as client:
            r = client.post(url, json=payload)
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
    text = normalize_whitespace(text)
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    chunks: List[str] = []
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
            while len(p) > chunk_size:
                chunks.append(p[:chunk_size])
                p = p[chunk_size - chunk_overlap :]
            buf = p

    flush()

    if chunk_overlap > 0 and len(chunks) > 1:
        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            prev_tail = overlapped[-1][-chunk_overlap:]
            overlapped.append(prev_tail + "\n" + chunks[i])
        chunks = overlapped

    return chunks


def validate_collection_name(name: str):
    if not re.match(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{1,510}[a-zA-Z0-9]$", name):
        raise ValueError


def make_id(source: str, chunk_index: int, chunk_text: str) -> str:
    h = hashlib.sha1((source + str(chunk_index) + chunk_text).encode("utf-8", errors="ignore")).hexdigest()
    return h


class ChromaRAG:
    def __init__(
        self, persist_dir: str = "./chroma_db", collection_name: str = "kb_docs_v1"
    ):
        validate_collection_name(collection_name)
        self.client = chromadb.PersistentClient(path=persist_dir, settings=Settings(anonymized_telemetry=False))
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
        self.col.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)

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
        ctx_blocks.append(f"[片段 {i} | {os.path.basename(src)} | chunk={idx}]\n{c['text']}")
    ctx = "\n\n".join(ctx_blocks)

    return f"""你是一个严谨的资料问答助手。请只根据“给定资料片段”回答问题；资料没有覆盖就说不知道，不要编。

    【给定资料片段】
    {ctx}

    【用户问题】
    {user_question}

    【回答要求】
    - 用中文回答
    - 尽量引用具体片段内容（可点名“片段1/2/3”）
    - 不要胡编
    - 点名后在末尾添加引用来源，例如：“（片段2原文）”
    "用户问什么就回答什么，不要回答多余内容。\n\n"
    """


def ingest_folder(
    rag: ChromaRAG,
    ollama: OllamaClient,
    embedding_model: str,
    docs_dir: str,
    chunk_size: int = 900,
    chunk_overlap: int = 150,
    batch_size: int = 128,
):
    docs = load_documents(docs_dir)
    if not docs:
        print(f"未找到可导入文档：{docs_dir}")
        return

    print(f"发现 {len(docs)} 个文档，开始切块+入库…")

    ids: List[str] = []
    metadatas: List[Dict[str, Any]] = []
    documents: List[str] = []
    embeddings: List[List[float]] = []

    def flush_batch():
        nonlocal ids, metadatas, documents, embeddings
        if not ids:
            return
        rag.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
        print(f"已写入 {len(ids)} chunks，当前库总量：{rag.count()}")
        ids, metadatas, documents, embeddings = [], [], [], []

    for d in docs:
        source = d["source"]
        chunks = chunk_text(d["text"], chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        for ci, ch in enumerate(chunks):
            _id = make_id(source, ci, ch)

            ids.append(_id)
            documents.append(ch)
            metadatas.append({"source": source, "chunk_index": ci})

            emb = ollama.embed(ch, model=embedding_model)
            embeddings.append(emb)

            if len(ids) >= batch_size:
                flush_batch()

    flush_batch()
    print("导入完成。")


def answer_with_rag_stream(
    rag: ChromaRAG,
    ollama: OllamaClient,
    llm_model: str,
    embedding_model: str,
    question: str,
    top_k: int = 5,
):
    q_emb = ollama.embed(question, model=embedding_model)

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

    for token in ollama.stream_generate(rag_prompt, model=llm_model):
        yield token


def build_react_prompt(question: str, scratchpad: str, mcp_tools_desc: str = "") -> str:
    builtin_tools = "search_kb: use this to search the knowledge base and retrieve relevant passages."

    all_tools = builtin_tools + ("\n" + mcp_tools_desc if mcp_tools_desc else "")

    format_hint = (
        "Use the following step format strictly:\n"
        "Thought: <reason about what to do>\n"
        "Action: <tool name or 'finish'>\n"
        "Action Input: <input to the tool, or the final answer if action is 'finish'>\n"
        "Observation: <tool result>\n"
        "... (repeat Thought/Action/Action Input/Observation) ...\n"
        "Final Answer: <answer to the user in Chinese>"
    )

    return (
        "你是一个可以使用工具的助手。\n"
        "工具列表:\n"
        f"- {all_tools}\n\n"
        "规则:\n"
        "- 优先使用 search_kb 查询知识库。\n"
        "- 当知识库中没有相关信息时，使用 web_search 搜索互联网。\n"
        "- 如需结束，使用 Action: finish 并在 Final Answer 给出中文答复。\n"
        "- 不要编造未检索到的事实。\n\n"
        "用户问什么就回答什么，不要回答多余内容。\n\n"
        "格式要求（务必遵守）：\n"
        f"{format_hint}\n\n"
        f"用户问题: {question}\n\n"
        "已知的思考与行动记录（可为空）：\n"
        f"{scratchpad}\n"
        "请给出下一步 Thought/Action/Action Input，或直接 Final Answer。"
    )


def _summarize_contexts_for_observation(contexts: List[Dict[str, Any]]) -> str:
    lines = ["Top retrieved passages:"]
    for i, ctx in enumerate(contexts, 1):
        src = os.path.basename(ctx.get("source", "unknown"))
        idx = ctx.get("chunk_index", "?")
        snippet = ctx.get("text", "")[:320].replace("\n", " ")
        lines.append(f"{i}. {src} | chunk={idx} | {snippet}")
    print("\n".join(lines))
    return "\n".join(lines)


class ReActAgent:
    def __init__(
        self,
        rag: ChromaRAG,
        ollama: OllamaClient,
        llm_model: str,
        embedding_model: str,
        mcp_manager: Optional[MCPClientManager] = None,
        top_k: int = 5,
        max_turns: int = 8,
        mcp_loop: Optional[asyncio.AbstractEventLoop] = None,
    ):
        self.rag = rag
        self.ollama = ollama
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.mcp = mcp_manager
        self.top_k = top_k
        self.max_turns = max_turns
        self.mcp_loop = mcp_loop

        self.ollama.generate_model = llm_model
        self.ollama.embed_model = embedding_model

    def _tool_search_kb(self, query: str) -> str:
        q_emb = self.ollama.embed(query, model=self.embedding_model)
        result = self.rag.query(q_emb, n_results=self.top_k)
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
        print("contexts from search_kb:")
        print(contexts)
        return _summarize_contexts_for_observation(contexts)

    def _call_tool(self, action: str, action_input: str) -> str:
        if self.mcp and action in self.mcp.list_tools():
            print(f"Calling MCP tool: {action}")
            if not self.mcp_loop:
                return "MCP event loop not available."
            try:
                result = self.mcp_loop.run_until_complete(
                    self.mcp.call_tool(action, {"query": action_input})
                )
                print("MCP tool result:")
                print(result)
                return result
            except Exception as e:
                return f"Error calling {action}: {e}"

        if action == "search_kb":
            print("Calling built-in tool: search_kb")
            kb_result = self._tool_search_kb(action_input)
            print("search_kb result:")
            print(kb_result)
            return kb_result
        return f"Unsupported action: {action}"

    def run(self, question: str) -> str:
        scratch: List[str] = []

        mcp_tools_desc = ""
        if self.mcp:
            mcp_tools_desc = self.mcp.get_tools_description()
            print(f"MCP tools available:\n{mcp_tools_desc}")

        for turn in range(self.max_turns):
            print(f"ReAct 轮次 {turn + 1}/{self.max_turns}")

            prompt_text = build_react_prompt(question, "\n".join(scratch), mcp_tools_desc)
            print("=" * 20)
            print(prompt_text)
            reply = self.ollama.generate(prompt_text, model=self.llm_model, stop=["Observation:"])
            print(f"LLM 回复:\n{reply}\n")

            thought_match = re.search(r"Thought\s*:\s*(.+?)(?=\nAction|$)", reply, re.S)
            if thought_match:
                thought = thought_match.group(1).strip()
                print(f"Thought: {thought}")

            action_match = re.search(r"Action\s*:\s*([a-zA-Z_]+)", reply)
            input_match = re.search(r"Action Input\s*:\s*(.+?)(?=\n|$)", reply, re.S)

            if action_match and input_match:
                action = action_match.group(1).strip()
                action_input = input_match.group(1).strip()

                # 显式处理结束状态：当 Action 为 finish 时，Action Input 就是最终答案
                if action.lower() == "finish":
                    print(f"\nFinal Answer (from Action): {action_input}\n")
                    return action_input

                print(f"Action: {action}")
                print(f"Action Input: {action_input}")
                print("调用工具…")

                observation = self._call_tool(action, action_input)
                print("=" * 20)
                print(observation)
                print(f"Observation: {observation[:300]}{'...' if len(observation) > 300 else ''}")

                scratch.append(reply.strip())
                scratch.append(f"Observation: {observation}")
                continue

            final_match = re.search(r"Final Answer\s*:\s*(.+)", reply, re.S)
            if final_match:
                final_answer = final_match.group(1).strip()
                print(f"\nFinal Answer: {final_answer}\n")
                return final_answer

            print("无法解析回复，尝试直接作为答案")
            return reply.strip()

        return "达到最大轮次仍未给出最终答案。"


def main():
    llm_model = "gemma3:4b"
    embedding_model = "nomic-embed-text"
    docs_dir = "./knowledge"

    rag = ChromaRAG(persist_dir="./chroma_db", collection_name="kb_docs_v1")
    ollama = OllamaClient("http://localhost:11434", generate_model=llm_model, embed_model=embedding_model)

    mcp_manager: Optional[MCPClientManager] = MCPClientManager()
    mcp_loop: Optional[asyncio.AbstractEventLoop] = None
    try:
        server_script = "mcp_servers/search_server.py"
        mcp_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(mcp_loop)
        mcp_loop.run_until_complete(
            mcp_manager.connect_server(name="search", command="python", args=[server_script])
        )
    except Exception as e:
        print(f"Failed to connect MCP server: {e}. Continuing without MCP support.")
        mcp_manager = None
        if mcp_loop:
            mcp_loop.close()
            asyncio.set_event_loop(None)
            mcp_loop = None

    agent = ReActAgent(
        rag=rag,
        ollama=ollama,
        llm_model=llm_model,
        embedding_model=embedding_model,
        mcp_manager=mcp_manager,
        top_k=5,
        max_turns=8,
        mcp_loop=mcp_loop,
    )

    print("  /ingest 导入 ./knowledge 下的 PDF/MD 到 Chroma")
    print("  /count   查看向量库条目数")
    print("  /react  以 ReAct 模式回答")
    print("  /exit    退出")

    try:
        while True:
            try:
                user_input = prompt("User: ")
            except (KeyboardInterrupt, EOFError):
                print("\n退出程序...")
                break

            cmd = user_input.strip()

            if cmd.lower() in {"exit", "quit", "/exit"}:
                break

            if cmd == "/count":
                print(f"Chroma count = {rag.count()}")
                continue

            if cmd == "/ingest":
                ingest_folder(
                    rag=rag,
                    ollama=ollama,
                    embedding_model=embedding_model,
                    docs_dir=docs_dir,
                )
                continue

            if cmd.startswith("/react"):
                question = cmd[len("/react") :].strip()
                if not question:
                    print("请在 /react 后输入问题。")
                    continue
                print("ReAct agent 正在思考…")
                answer = agent.run(question)
                print(f"{answer}")
                continue

            for token in answer_with_rag_stream(
                rag=rag,
                ollama=ollama,
                llm_model=llm_model,
                embedding_model=embedding_model,
                question=user_input,
                top_k=5,
            ):
                print(token, end="", flush=True)
            print("\n")
    finally:
        if mcp_manager and mcp_loop:
            try:
                mcp_loop.run_until_complete(mcp_manager.close())
            except Exception as e:
                print(f"Error closing MCP manager: {e}")
            finally:
                mcp_loop.close()
                asyncio.set_event_loop(None)


if __name__ == "__main__":
    main()