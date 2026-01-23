"""
Mock Search Server for MCP
Provides a mock web search tool that returns simulated results without making actual web requests.
"""
import asyncio
from typing import Any, Dict, List

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent


app = Server("mock-search-server")


@app.list_tools()
async def list_tools() -> List[Tool]:
    """
    Returns the list of available tools
    Returns:
        List[Tool]: List of available tools, currently only contains the web search tool
    """
    return [
        Tool(
            name="web_search",
            description=(
                "Mock web search for testing. Returns simulated search results. "
                "Use this when testing without requiring actual web searches."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
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
    """
    Call a tool with the given arguments
    
    Args:
        name: Tool name
        arguments: Tool arguments
        
    Returns:
        List[TextContent]: List of text content responses
    """
    if name != "web_search":
        raise ValueError(f"Unknown tool: {name}")

    query = arguments.get("query", "")
    max_results = arguments.get("max_results", 5)
    
    if not query:
        return [TextContent(
            type="text",
            text="Error: Missing required parameter 'query'"
        )]
    
    # Return mock search results
    mock_results = []
    for i in range(min(max_results, 3)):
        mock_results.append({
            "title": f"Mock Result {i+1} for '{query}'",
            "body": f"This is a simulated search result snippet for query: {query}. "
                   f"In a real scenario, this would contain actual web content.",
            "href": f"https://example.com/result-{i+1}"
        })
    
    response_lines = [f"Mock Web Search Results for '{query}':\n"]
    
    for i, result in enumerate(mock_results, 1):
        title = result.get("title", "No Title")
        snippet = result.get("body", "No description")
        url = result.get("href", "")
        
        response_lines.append(f"{i}. {title}")
        response_lines.append(f"   {snippet}")
        if url:
            response_lines.append(f"   Link: {url}")
        response_lines.append("")
    
    return [TextContent(
        type="text",
        text="\n".join(response_lines)
    )]


async def main():
    """Main entry point for the mock search server"""
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
