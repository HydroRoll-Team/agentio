"""
MCP (Model Context Protocol) 客户端测试模块

该模块提供了对 MCP 客户端功能的测试，特别是测试 ReAct Agent 与
MCP 服务器的集成。测试包括连接 MCP 服务器、调用工具以及与 RAG
(Retrieval-Augmented Generation) 系统的协同工作。
"""

import asyncio
from loguru import logger

from examples.mcp_client import MCPClientManager

async def test_react_with_mcp():
    """
    测试 ReAct Agent 与 MCP 的集成功能
    
    该测试函数执行以下操作：
    1. 初始化 RAG 系统和 Ollama 客户端
    2. 连接到 MCP 搜索服务器
    3. 创建配置好的 ReAct Agent
    4. 向 Agent 提问并获取答案
    5. 验证 Agent 的响应能力
    
    测试流程：
    - 设置 RAG 系统使用本地 ChromaDB
    - 配置 Ollama 使用本地服务器
    - 连接到搜索服务器（如果可用）
    - 创建 Agent 并配置参数
    - 提问并记录答案
    - 清理资源（断开 MCP 连接）
    
    Raises:
        Exception: 当测试过程中出现任何错误时抛出
    
    Note:
        如果 MCP 服务器连接失败，测试会继续进行但不会使用 MCP 工具。
        该测试需要本地运行 Ollama 服务器和 ChromaDB 数据库。
    """
    print("\nTesting ReAct Agent with MCP")
    
    from examples.react import ChromaRAG, OllamaClient, ReActAgent
    
    rag = ChromaRAG(persist_dir="./chroma_db", collection_name="kb_docs_v1")
    ollama = OllamaClient("http://localhost:11434")
    
    mcp_manager = MCPClientManager()
    try:
        await mcp_manager.connect_server(
            name="search",
            command="python",
            args=["mcp_servers/search_server.py"]
        )
    except Exception as e:
        print(f"Could not connect MCP: {e}")
        mcp_manager = None

    agent = ReActAgent(
        rag=rag,
        ollama=ollama,
        llm_model="gemma3:12b",
        embedding_model="nomic-embed-text",
        mcp_manager=mcp_manager,
        top_k=3,
        max_turns=5,
    )
    
    question = "What is the latest version of Python released in 2026?"
    
    print(f"Question: {question}")
    print("Agent is thinking...")
    
    try:
        answer = await agent.run(question)
        print(f"Answer: {answer}")
        print("ReAct agent test passed!")
    except Exception as e:
        print(f"ReAct agent test failed: {e}")
        raise
    finally:
        if mcp_manager:
            await mcp_manager.disconnect_all()


async def main():
    try:
        await test_react_with_mcp()
        
        print("\nAll tests completed successfully!")
        
    except Exception as e:
        print(f"\nTests failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
