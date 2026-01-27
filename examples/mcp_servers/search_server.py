import asyncio
import sys
from typing import Any, Dict, List

try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS
    
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent


app = Server("search-server")


@app.list_tools()
async def list_tools() -> List[Tool]:
    """
    返回可用工具的列表
    Returns:
        List[Tool]: 包含可用工具的列表，目前只包含网络搜索工具
    """
    return [
        Tool(
            name="web_search",
            description=(
                "Search the web for current information using DuckDuckGo. " 
                "Use this tool when the knowledge base doesn't have information "
                "about recent events, current data, or topics outside the knowledge base. " 
                "Returns top search results with titles, snippets, and URLs."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query (supports English and Chinese)"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return (default: 5)",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: Any) -> List[TextContent]:
    if name != "web_search":
        raise ValueError(f"未知工具: {name}")

    query = arguments.get("query", "")
    max_results = arguments.get("max_results", 5)
    
    if not query:
        return [TextContent(
            type="text",
            text="错误：缺少必需的参数 'query'"
        )]
    
    try:
        results = await perform_web_search(query, max_results)
        
        if not results:
            return [TextContent(
                type="text",
                text=f"未找到与查询相关的搜索结果: {query}"
            )]
        
        response_lines = [f"网络搜索结果 '{query}':\n"]
        
        for i, result in enumerate(results, 1):
            title = result.get("title", "无标题")
            snippet = result.get("body", result.get("snippet", "无描述"))
            url = result.get("href", result.get("url", ""))
            
            response_lines.append(f"{i}. {title}")
            response_lines.append(f"   {snippet}")
            if url:
                response_lines.append(f"   链接: {url}")
            response_lines.append("") 
        
        return [TextContent(
            type="text",
            text="\n".join(response_lines)
        )]
        
    except Exception as e:
        return [TextContent(
            type="text",
            text=f"执行网络搜索时出错: {str(e)}"
        )]


async def perform_web_search(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    try:
        loop = asyncio.get_event_loop()
        
        def _search():
            for attempt in range(2):
                try:
                    ddgs = DDGS(timeout=30)
                    results = ddgs.text(
                        query,
                        region='wt-wt',
                        safesearch='moderate',
                        max_results=max_results
                    )
                    result_list = list(results)
                    if result_list:
                        return result_list
                    
                except Exception as e:
                    print(f"搜索尝试 {attempt + 1} 失败: {e}", file=sys.stderr)
                    if attempt == 0:
                        import time
                        time.sleep(1)
                        continue
            
            return []
        
        results = await loop.run_in_executor(None, _search)
        return results
        
    except Exception as e:
        print(f"搜索错误: {e}", file=sys.stderr)
        return []


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
