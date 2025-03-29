"""
MCPlex - Model Context Protocol Implementation
"""

from .client import initialize_mcp, shutdown, run_interaction
from .mcp_types import ServerConfig, MCPConnection, ToolDefinition
from .mcp_errors import MCPError, ConnectionError, TimeoutError, ProtocolError, ToolError
from .mcp_manager import MCPManager

__version__ = "2.0.0"

__all__ = [
    # Main API
    "initialize_mcp",
    "shutdown",
    "run_interaction",
    
    # Types
    "ServerConfig",
    "MCPConnection",
    "ToolDefinition",
    
    # Errors
    "MCPError",
    "ConnectionError",
    "TimeoutError",
    "ProtocolError",
    "ToolError",
    
    # Core classes
    "MCPManager"
]
