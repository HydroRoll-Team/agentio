from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters as ServerParams


@dataclass
class ToolSchema:
    name: str
    description: str
    server_name: str
    input_schema: Dict[str, Any]


class MCPClientManager:
    """
    MCP (Model Context Protocol) 客户端管理器
    
    该类负责管理与 MCP 服务器的连接、工具发现和工具调用。它提供了统一的接口来
    连接多个 MCP 服务器，自动发现每个服务器提供的工具，并执行这些工具。
    
    Attributes:
        servers (Dict[str, ClientSession]): 存储服务器名称到 ClientSession 的映射
        tools (Dict[str, ToolSchema]): 存储工具名称到工具架构的映射
        _server_processes (Dict[str, Any]): 存储服务器名称到进程上下文的映射
    
    Example:
        >>> manager = MCPClientManager()
        >>> await manager.connect_server(
        ...     name="my_server",
        ...     command="python",
        ...     args=["server.py"]
        ... )
        >>> result = await manager.call_tool("my_tool", {"param": "value"})
        >>> await manager.disconnect_all()
    """
    def __init__(self):
        self.servers: Dict[str, ClientSession] = {}
        self.tools: Dict[str, ToolSchema] = {}
        self._server_processes: Dict[str, Any] = {}
    
    async def connect_server(self, name: str, command: str, args: List[str], 
                            env: Optional[Dict[str, str]] = None):
        """
        连接到指定的 MCP 服务器
        
        该方法会启动一个新的 MCP 服务器进程，建立通信会话，并自动发现该服务器
        提供的所有工具。连接成功后，工具会被注册到管理器的工具字典中。
        
        Args:
            name (str): 服务器的唯一标识符，用于后续引用该服务器
            command (str): 启动服务器的命令（如 "python"、"node" 等）
            args (List[str]): 传递给服务器命令的参数列表
            env (Optional[Dict[str, str]]): 可选的环境变量字典，默认为 None
        
        Raises:
            Exception: 当连接失败或工具发现失败时抛出异常
        
        Example:
            >>> await manager.connect_server(
            ...     name="filesystem",
            ...     command="python",
            ...     args=["-m", "mcp_server.filesystem"],
            ...     env={"PATH": "/custom/path"}
            ... )
        """
        print(f"Connecting to MCP server '{name}' with command: {command} {' '.join(args)}")
        
        try:
            server_params = ServerParams(
                command=command,
                args=args,
                env=env
            )
            
            stdio_ctx = stdio_client(server_params)
            read_stream, write_stream = await stdio_ctx.__aenter__()
            
            session_ctx = ClientSession(read_stream, write_stream)
            session = await session_ctx.__aenter__()
            
            await session.initialize()
            self.servers[name] = session
            self._server_processes[name] = (stdio_ctx, session_ctx)
            await self._discover_tools(name, session)
            
            print(f"Successfully connected to MCP server '{name}'")
            
        except Exception as e:
            print(f"Failed to connect to MCP server '{name}': {e}")
            raise
    
    async def _discover_tools(self, server_name: str, session: ClientSession):
        """
        从指定的服务器会话中发现并注册工具
        
        该方法会调用服务器的 list_tools 方法获取所有可用工具，并将每个工具
        的信息（名称、描述、输入架构等）注册到工具字典中。
        
        Args:
            server_name (str): 服务器的名称标识符
            session (ClientSession): 已建立的 MCP 客户端会话
        
        Raises:
            Exception: 当工具发现失败时抛出异常
        
        Note:
            该方法是内部方法，由 connect_server 自动调用
        """
        try:
            tools_response = await session.list_tools()
            print(f"Discovered {len(tools_response.tools)} tools from '{server_name}'")
            for tool in tools_response.tools:
                tool_schema = ToolSchema(
                    name=tool.name,
                    description=tool.description or "No description available",
                    server_name=server_name,
                    input_schema=tool.inputSchema if hasattr(tool, 'inputSchema') else {}
                )
                
                self.tools[tool.name] = tool_schema
                print(f"Registered tool: {tool.name} ({tool_schema.description})")
                
        except Exception as e:
            print(f"Failed to discover tools from '{server_name}': {e}")
            raise
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """
        调用指定的工具并返回结果
        
        该方法会查找工具所属的服务器，通过该服务器的会话执行工具调用，
        并将返回的结果格式化为字符串。
        
        Args:
            tool_name (str): 要调用的工具名称
            arguments (Dict[str, Any]): 传递给工具的参数字典
        
        Returns:
            str: 工具执行结果的字符串表示
        
        Raises:
            ValueError: 当工具不存在时抛出
            RuntimeError: 当工具所属的服务器未连接时抛出
        
        Example:
            >>> result = await manager.call_tool(
            ...     "read_file",
            ...     {"path": "/path/to/file.txt"}
            ... )
            >>> print(result)
        """
        if tool_name not in self.tools:
            available_tools = ", ".join(self.tools.keys())
            raise ValueError(f"Tool '{tool_name}' not found. Available tools: {available_tools}")
        
        tool_schema = self.tools[tool_name]
        server_name = tool_schema.server_name
        
        if server_name not in self.servers:
            raise RuntimeError(f"Server '{server_name}' not connected")
        
        session = self.servers[server_name]
        
        try:
            print(f"Calling tool '{tool_name}' with arguments: {arguments}")
            
            result = await session.call_tool(tool_name, arguments=arguments)
            
            if hasattr(result, 'content') and result.content:
                content_parts = []
                for item in result.content:
                    if isinstance(item, dict):
                        if 'text' in item:
                            content_parts.append(item['text'])
                        elif 'data' in item:
                            content_parts.append(str(item['data']))
                        else:
                            content_parts.append(str(item))
                    else:
                        text = getattr(item, 'text', None)
                        if text is not None:
                            content_parts.append(text)
                        else:
                            content_parts.append(str(item))
                
                result_text = "\n".join(content_parts)
                print(f"Tool '{tool_name}' returned: {result_text[:200]}...")
                return result_text
            else:
                return str(result)
                
        except Exception as e:
            error_msg = f"Error calling tool '{tool_name}': {e}"
            print(error_msg)
            return error_msg
    
    def get_tools_description(self) -> str:
        """
        获取所有可用工具的描述信息
        Returns:
            str: 返回工具描述字符串，如果没有工具则返回提示信息
        """
        if not self.tools:
            return "No MCP tools available."  # 如果没有工具，返回提示信息
        
        descriptions = []
        for tool_name, schema in self.tools.items():
            descriptions.append(f"{tool_name}: {schema.description}")  # 将每个工具的名称和描述添加到列表中
        
        return "\n".join(descriptions)  # 将所有描述用换行符连接并返回
    
    def list_tools(self) -> List[str]:
        """
        获取所有已注册工具的名称列表
        
        Returns:
            List[str]: 包含所有工具名称的列表
        
        Example:
            >>> tools = manager.list_tools()
            >>> print(tools)
            ['tool1', 'tool2', 'tool3']
        """
        return list(self.tools.keys())
    
    async def disconnect_all(self):
        """
        断开与所有 MCP 服务器的连接
        
        该方法会遍历所有已连接的服务器，并逐个断开连接。它会正确关闭
        所有会话上下文和进程，并清理相关的服务器、工具和进程信息。
        
        该方法应该在使用完管理器后调用，以确保所有资源被正确释放。
        
        Raises:
            该方法会捕获并记录所有异常，不会向上抛出
        
        Example:
            >>> try:
            ...     # 使用管理器执行一些操作
            ...     pass
            ... finally:
            ...     await manager.disconnect_all()
        """
        for name in list(self.servers.keys()):
            try:
                # 检查服务器是否在进程中运行
                if name in self._server_processes:
                    # 获取服务器的标准输入输出上下文和会话上下文
                    stdio_ctx, session_ctx = self._server_processes[name]
                    # 退出会话上下文
                    await session_ctx.__aexit__(None, None, None)
                    # 退出标准输入输出上下文
                    await stdio_ctx.__aexit__(None, None, None)
                # 记录断开连接的信息
                print(f"Disconnected from MCP server '{name}'")
            except Exception as e:
                # 记录断开连接时可能出现的错误
                print(f"Error disconnecting from '{name}': {e}")
        
        # 清空所有服务器的字典
        self.servers.clear()
        # 清空所有工具的字典
        self.tools.clear()
        # 清空所有服务器进程的字典
        self._server_processes.clear()

    async def close(self):
        """
        关闭所有 MCP 服务器连接的别名方法
        
        这是 disconnect_all 的别名方法，提供更符合直觉的接口名称。
        
        Example:
            >>> try:
            ...     # 使用管理器执行一些操作
            ...     pass
            ... finally:
            ...     await manager.close()
        """
        await self.disconnect_all()
