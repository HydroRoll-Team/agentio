"""
>>> rag = ChromaRAG(persist_dir="./chroma_db", collection_name="kb_docs_v1")
>>> ollama = OllamaClient("http://localhost:11434")
>>> agent = ReActAgent(rag=rag, ollama=ollama)
>>> answer = await agent.run("什么是 Python?")
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
from loguru import logger

from mcp_client import MCPClientManager


class OllamaClient:
    """
    Ollama API 客户端

    Example:
        >>> client = OllamaClient("http://localhost:11434")
        >>> response = await client.generate("gemma3:12b", "你好")
        >>> async for token in client.stream_generate("gemma3:12b", "你好"):
        ...     print(token, end="")
        >>> embedding = await client.embed("nomic-embed-text", "Hello world")
    """

    def __init__(
        self, api_base: str = "http://localhost:11434", generate_model: str = "qwen3:0.6b", embed_model: str = "nomic-embed-text"
    ):
        self.api_base = api_base.rstrip("/")
        self.generate_model = generate_model
        self.embed_model = embed_model

    async def generate(self, _prompt: str) -> str:
        """
        非流式文本生成

        Args:
            _prompt (str): 输入提示词

        Returns:
            str: 模型生成的完整响应文本

        Raises:
            httpx.HTTPError: 当 API 请求失败时抛出

        Example:
            >>> client = OllamaClient()
            >>> response = await client.generate("介绍一下 Python")
            >>> print(response)
        """
        url = f"{self.api_base}/api/generate"
        payload = {"model": self.generate_model, "prompt": _prompt, "stream": False}
        async with httpx.AsyncClient() as client:
            r = await client.post(url, json=payload, timeout=300)
            r.raise_for_status()
            return r.json()["response"]

    async def stream_generate(self, _prompt: str):
        """
        流式文本生成

        Args:
            _prompt (str): 输入提示词

        Yields:
            str: 模型生成的文本 token

        Raises:
            httpx.HTTPError: 当 API 请求失败时抛出

        Note:
            Ollama 返回 NDJSON 格式，每行是一个 JSON 对象。
            该方法设置了 timeout=None 以避免长输出被中断。

        Example:
            >>> async for token in client.stream_generate("你好"):
            ...     print(token, end="", flush=True)
        """
        url = f"{self.api_base}/api/generate"
        payload = {"model": self.generate_model, "prompt": _prompt, "stream": True}
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
                        
    async def embed(self, text: str) -> List[float]:
        """
        生成文本嵌入向量

        使用指定的嵌入模型将文本转换为向量表示，用于语义搜索和相似度计算。

        Args:
            text (str): 要生成嵌入的文本

        Returns:
            List[float]: 文本的嵌入向量，维度取决于模型

        Raises:
            httpx.HTTPError: 当 API 请求失败时抛出

        Example:
            >>> embedding = await client.embed("Hello world")
            >>> print(len(embedding))
        """
        url = f"{self.api_base}/api/embeddings"
        payload = {"model": self.embed_model, "prompt": text}
        async with httpx.AsyncClient(timeout=300) as client:
            r = await client.post(url, json=payload)
            r.raise_for_status()
            data = r.json()
            return data["embedding"]


def read_pdf(path: str) -> str:
    """
    读取 PDF 文件内容

    Args:
        path (str): PDF 文件的路径

    Returns:
        str: PDF 文件的所有页面文本内容，用换行符连接

    Example:
        >>> text = read_pdf("document.pdf")
        >>> print(text)
    """
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
    """
    规范化文本中的空白字符

    将不换行空格替换为普通空格，合并连续空格，
    并将连续的多个换行符压缩为最多
    """
    s = s.replace("\u00a0", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def chunk_text(text: str, chunk_size: int = 900, chunk_overlap: int = 150) -> List[str]:
    """
    将文本分割成重叠的块

    1. 先按空行将文本分成段落
    2. 将段落合并到接近 chunk_size 大小
    3. 添加 overlap 防止语义在边界处丢失

    Args:
        text (str): 要分割的文本
        chunk_size (int): 每个块的目标大小，默认 900 字符
        chunk_overlap (int): 块之间的重叠大小，默认 150 字符

    Returns:
        List[str]: 分割后的文本块列表

    Example:
        >>> chunks = chunk_text("Long text...", chunk_size=500, chunk_overlap=100)
        >>> print(f"Created {len(chunks)} chunks")
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

    # overlap：每个 chunk 前面附上一点上一块尾巴
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
    """
    为文档块生成唯一标识符

    Example:
        >>> doc_id = make_id("doc.pdf", 0, "First chunk")
        >>> print(doc_id)
    """
    h = hashlib.sha1(
        (source + str(chunk_index) + chunk_text).encode("utf-8", errors="ignore")
    ).hexdigest()
    return h


class ChromaRAG:
    """
    基于 ChromaDB 的 RAG

    Attributes:
        client: ChromaDB 持久化客户端
        col: ChromaDB 集合对象

    Example:
        >>> rag = ChromaRAG(persist_dir="./chroma_db", collection_name="kb_docs_v1")
        >>> rag.upsert(ids=["doc1"], embeddings=[[0.1, 0.2]], documents=["text"], metadatas=[{}])
        >>> results = rag.query([0.1, 0.2], n_results=5)
    """

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
        """
        插入或更新文档

        将文档及其嵌入向量添加到集合中，如果 ID 已存在则更新。

        Args:
            ids (List[str]): 文档唯一标识符列表
            embeddings (List[List[float]]): 文档嵌入向量列表
            documents (List[str]): 文档文本内容列表
            metadatas (List[Dict[str, Any]]): 文档元数据列表

        Example:
            >>> rag.upsert(
            ...     ids=["doc1"],
            ...     embeddings=[[0.1, 0.2]],
            ...     documents=["Hello world"],
            ...     metadatas=[{"source": "doc.txt"}]
            ... )
        """
        self.col.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    def query(self, query_embedding: List[float], n_results: int = 5) -> Dict[str, Any]:
        """
        执行语义搜索查询

        根据查询嵌入向量检索最相似的文档

        Args:
            query_embedding (List[float]): 查询文本的嵌入向量
            n_results (int): 要返回的结果数量，默认 5

        Returns:
            Dict[str, Any]: 包含文档、元数据和距离的查询结果

        Example:
            >>> results = rag.query([0.1, 0.2], n_results=5)
            >>> print(results["documents"])
        """
        return self.col.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )


def build_rag_prompt(user_question: str, contexts: List[Dict[str, Any]]) -> str:
    """
    构建 RAG 问答提示词

    根据用户问题和检索到的上下文片段，构建用于 LLM 的提示词。

    Args:
        user_question (str): 用户的问题
        contexts (List[Dict[str, Any]]): 检索到的上下文片段列表

    Returns:
        str: 格式化的提示词

    Example:
        >>> prompt = build_rag_prompt("什么是 Python?", contexts)
        >>> print(prompt)
    """
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
    - 尽量引用具体片段内容（可点名“片段1/2/3”）
    - 不要胡编
    - 点名后在末尾添加引用来源，例如：“（片段2原文）”
    "用户问什么就回答什么，不要回答多余内容。\n\n"
    """


async def ingest_folder(
    rag: ChromaRAG,
    ollama: OllamaClient,
    embedding_model: str,
    docs_dir: str,
    chunk_size: int = 900,
    chunk_overlap: int = 150,
    batch_size: int = 128,
    concurrency: int = 128,
):
    docs = load_documents(docs_dir)
    if not docs:
        print(f"未找到可导入文档：{docs_dir}")
        return

    print(f"发现 {len(docs)} 个文档，开始切块+入库…")

    sem = asyncio.Semaphore(concurrency)

    async def embed_one(text: str) -> List[float]:
        async with sem:
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

    # 收集所有 chunks
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
    llm_model: str,
    embedding_model: str,
    question: str,
    top_k: int = 5,
):
    """
    使用 RAG 流式回答问题

    1. 为问题生成嵌入向量
    2. 检索最相关的文档片段
    3. 构建提示词
    4. 流式生成回答

    Args:
        rag (ChromaRAG): RAG 系统实例
        ollama (OllamaClient): Ollama 客户端
        question (str): 用户的问题
        top_k (int): 检索的文档数量，默认 5

    Yields:
        str: 生成的文本 token

    Example:
        >>> async for token in answer_with_rag_stream(rag, ollama, "gemma3:12b", "nomic-embed-text", "什么是 Python?"):
        ...     print(token, end="", flush=True)
    """
    q_emb = await ollama.embed(question)

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


def build_react_prompt(question: str, scratchpad: str, mcp_tools_desc: str = "") -> str:
    builtin_tools = "search_kb: use this to search the knowledge base and retrieve relevant passages."

    if mcp_tools_desc:
        all_tools = builtin_tools + "\n" + mcp_tools_desc
    else:
        all_tools = builtin_tools

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
    ):
        self.rag = rag
        self.ollama = ollama
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.mcp = mcp_manager
        self.top_k = top_k
        self.max_turns = max_turns

    async def _tool_search_kb(self, query: str) -> str:
        q_emb = await self.ollama.embed(query)
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

        return _summarize_contexts_for_observation(contexts)

    async def _call_tool(self, action: str, action_input: str) -> str:
        if self.mcp and action in self.mcp.list_tools():
            try:
                logger.info(f"Calling MCP tool: {action}")
                result = await self.mcp.call_tool(action, {"query": action_input})
                return result
            except Exception as e:
                return f"Error calling {action}: {str(e)}"

        if action == "search_kb":
            return await self._tool_search_kb(action_input)

        return f"Unsupported action: {action}"

    async def run(self, question: str) -> str:
        scratch = []

        mcp_tools_desc = ""
        if self.mcp:
            mcp_tools_desc = self.mcp.get_tools_description()

        for turn in range(self.max_turns):
            print(f"ReAct 轮次 {turn + 1}/{self.max_turns}")
            
            prompt_text = build_react_prompt(
                question, "\n".join(scratch), mcp_tools_desc
            )
            reply = await self.ollama.generate(prompt_text)

            #! 提取 Thought
            thought_match = re.search(r"Thought\s*:\s*(.+?)(?=\nAction|$)", reply, re.S)
            if thought_match:
                thought = thought_match.group(1).strip()
                print(f"Thought: {thought}")
            
            #! 检查是否有 Final Answer
            final_match = re.search(r"Final Answer\s*:\s*(.+)", reply, re.S)
            if final_match:
                final_answer = final_match.group(1).strip()
                print(f"\nFinal Answer: {final_answer}\n")
                return final_answer

            #! 提取 Action 和 Action Input
            action_match = re.search(r"Action\s*:\s*([a-zA-Z_]+)", reply)
            input_match = re.search(r"Action Input\s*:\s*(.+?)(?=\n|$)", reply, re.S)

            if not action_match or not input_match:
                print(f"无法解析 Action，返回原始回复")
                return reply.strip()

            action = action_match.group(1).strip()
            action_input = input_match.group(1).strip()
            
            print(f"Action: {action}")
            print(f"Action Input: {action_input}")

            #! 执行工具并获取观察结果
            observation = await self._call_tool(action, action_input)
            print(f"Observation: {observation[:300]}{'...' if len(observation) > 300 else ''}")
            
            scratch.append(reply.strip())
            scratch.append(f"Observation: {observation}")

        return "达到最大轮次仍未给出最终答案。"


async def main():
    llm_model = "gemma3:12b"
    embedding_model = "nomic-embed-text"  
    docs_dir = "./knowledge"

    rag = ChromaRAG(persist_dir="./chroma_db", collection_name="kb_docs_v1")
    ollama = OllamaClient("http://localhost:11434")

    mcp_manager = MCPClientManager()
    try:
        server_script = "mcp_servers/search_server.py"
        await mcp_manager.connect_server(
            name="search", command="python", args=[server_script]
        )
    except Exception as e:
        logger.warning(
            f"Failed to connect MCP server: {e}. Continuing without MCP support."
        )
        mcp_manager = None

    agent = ReActAgent(
        rag=rag,
        ollama=ollama,
        llm_model=llm_model,
        embedding_model=embedding_model,
        mcp_manager=mcp_manager,
        top_k=5,
        max_turns=8,
    )

    loop = asyncio.get_event_loop()

    print("  /ingest 导入 ./knowledge 下的 PDF/MD 到 Chroma")
    print("  /count   查看向量库条目数")
    print("  /react  以 ReAct 模式回答")
    print("  /exit    退出")

    try:
        while True:
            try:
                user_input = await loop.run_in_executor(None, lambda: prompt("User: "))
            except (asyncio.CancelledError, KeyboardInterrupt):
                print("\n退出程序...")
                break
            
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

            if cmd.startswith("/react"):
                question = cmd[len("/react") :].strip()
                if not question:
                    print("请在 /react 后输入问题。")
                    continue
                print("ReAct agent 正在思考…")
                answer = await agent.run(question)
                print(f"LLM: {answer}")
                continue

            print("LLM: ", end="", flush=True)
            async for token in answer_with_rag_stream(
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
        if mcp_manager:
            try:
                await mcp_manager.close()
            except Exception as e:
                logger.debug(f"Error closing MCP manager: {e}")


if __name__ == "__main__":
    asyncio.run(main())
