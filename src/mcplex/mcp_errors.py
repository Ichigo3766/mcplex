"""Error handling for MCP operations."""

class MCPError(Exception):
    """Base class for MCP errors."""
    pass

class ConnectionError(MCPError):
    """Error establishing or maintaining connection to MCP server."""
    pass

class TimeoutError(MCPError):
    """Operation timed out."""
    pass

class ProtocolError(MCPError):
    """Error in MCP protocol communication."""
    pass

class ToolError(MCPError):
    """Error executing an MCP tool."""
    def __init__(self, tool_name: str, message: str):
        self.tool_name = tool_name
        self.message = message
        super().__init__(f"Tool '{tool_name}' error: {message}")

class ConfigurationError(MCPError):
    """Error in MCP configuration."""
    pass

class ErrorManager:
    """Centralized error handling for MCP operations."""
    
    @staticmethod
    def handle_connection_error(server_name: str, error: Exception) -> ConnectionError:
        """Handle connection-related errors."""
        return ConnectionError(f"Failed to connect to server '{server_name}': {str(error)}")
    
    @staticmethod
    def handle_timeout(server_name: str, operation: str, timeout: float) -> TimeoutError:
        """Handle timeout errors."""
        return TimeoutError(f"Operation '{operation}' on server '{server_name}' timed out after {timeout}s")
    
    @staticmethod
    def handle_protocol_error(server_name: str, message: str) -> ProtocolError:
        """Handle protocol-related errors."""
        return ProtocolError(f"Protocol error on server '{server_name}': {message}")
    
    @staticmethod
    def handle_tool_error(tool_name: str, error: Exception) -> ToolError:
        """Handle tool execution errors."""
        return ToolError(tool_name, str(error))