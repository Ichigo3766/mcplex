"""
MCPlex - A flexible Python client for interacting with Model Context Protocol (MCP) servers.
"""

from .client import run_interaction, MCPState, initialize_mcp
from .mcp_client import MCPClient

__version__ = "0.1.0"
__all__ = ["MCPClient", "run_interaction"]
