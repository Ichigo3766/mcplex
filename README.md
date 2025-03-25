# Dolphin MCP

A flexible Python library and CLI tool for interacting with Model Context Protocol (MCP) servers using any LLM model.

![CleanShot 2025-03-24 at 20 34 48](https://github.com/user-attachments/assets/ba0f24f2-ca1a-44f4-bd4f-01c800b0ca30)

## Overview

Dolphin MCP is both a Python library and a command-line tool that allows you to query and interact with MCP servers through natural language. It connects to any number of configured MCP servers, makes their tools available to language models (OpenAI, Anthropic, Ollama), and provides a conversational interface for accessing and manipulating data from these servers.

The project demonstrates how to:
- Connect to multiple MCP servers simultaneously
- List and call tools provided by these servers
- Use function calling capabilities to interact with external data sources
- Process and present results in a user-friendly way
- Create a reusable Python library with a clean API
- Build a command-line interface on top of the library

## Features

- **Multiple Provider Support**: Works with OpenAI, Anthropic, and Ollama models
- **Modular Architecture**: Clean separation of concerns with provider-specific modules
- **Dual Interface**: Use as a Python library or command-line tool
- **MCP Server Integration**: Connect to any number of MCP servers simultaneously
- **Tool Discovery**: Automatically discover and use tools provided by MCP servers
- **Flexible Configuration**: Configure models, servers, and timeouts through JSON configuration
- **Environment Variable Support**: Securely store API keys in environment variables
- **Comprehensive Documentation**: Detailed usage examples and API documentation
- **Installable Package**: Easy installation via pip with `dolphin-mcp-cli` command
- **Streaming Support**: Add stream=True to enable streaming
- **Connection Health Monitoring**: Automatic health checks and recovery
- **Message Queuing**: Prevents concurrent writes and ensures message delivery
- **Configurable Timeouts**: Per-server timeout settings for different operations
- **Enhanced Error Handling**: Improved error recovery and reporting

## Prerequisites

Before installing Dolphin MCP, ensure you have the following prerequisites installed:

1. **Python 3.8+**
2. **SQLite** - A lightweight database used by the demo
3. **uv/uvx** - A fast Python package installer and resolver

### Setting up Prerequisites

#### Windows

1. **Python 3.8+**:
   - Download and install from [python.org](https://www.python.org/downloads/windows/)
   - Ensure you check "Add Python to PATH" during installation

2. **SQLite**:
   - Download the precompiled binaries from [SQLite website](https://www.sqlite.org/download.html)
   - Choose the "Precompiled Binaries for Windows" section and download the sqlite-tools zip file
   - Extract the files to a folder (e.g., `C:\sqlite`)
   - Add this folder to your PATH:
     - Open Control Panel > System > Advanced System Settings > Environment Variables
     - Edit the PATH variable and add the path to your SQLite folder
     - Verify installation by opening Command Prompt and typing `sqlite3 --version`

3. **uv/uvx**:
   - Open PowerShell as Administrator and run:
     ```
     powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
     ```
   - Restart your terminal and verify installation with `uv --version`

#### macOS

1. **Python 3.8+**:
   ```bash
   brew install python
   ```

2. **SQLite**:
   ```bash
   brew install sqlite
   ```

3. **uv/uvx**:
   ```bash
   brew install uv
   ```
   Or use the official installer:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

#### Linux (Ubuntu/Debian)

1. **Python 3.8+**:
   ```bash
   sudo apt update
   sudo apt install python3 python3-pip
   ```

2. **SQLite**:
   ```bash
   sudo apt update
   sudo apt install sqlite3
   ```

3. **uv/uvx**:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

## Installation

### Option 1: Install from PyPI (Recommended)

```bash
pip install dolphin-mcp
```

### Option 2: Install from Source

1. Clone this repository:
   ```bash
   git clone https://github.com/cognitivecomputations/dolphin-mcp.git
   cd dolphin-mcp
   ```

2. Install the package in development mode:
   ```bash
   pip install -e .
   ```

3. Set up your environment variables:
   ```bash
   cp .env.example .env
   ```
   Then edit the `.env` file to add your API keys.

## Configuration

The project uses two main configuration files:

1. `.env` - Contains API configuration:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   OPENAI_MODEL=gpt-4o
   ANTHROPIC_API_KEY=your_anthropic_key_here
   ```

2. `mcp_config.json` - Defines MCP servers and their configuration:
   ```json
   {
     "mcpServers": {
       "server1": {
         "command": "command-to-start-server",
         "args": ["arg1", "arg2"],
         "env": {
           "ENV_VAR1": "value1",
           "ENV_VAR2": "value2"
         },
         "timeout": 300
       },
       "server2": {
         "command": "another-server-command",
         "args": ["--option", "value"],
         "timeout": 10,
         "disabled": true  // This server will be skipped
       }
     },
     "models": [
       {
         "model": "gpt-4o",
         "provider": "openai",
         "default": true,
         "systemMessage": "You are a smart and helpful assistant with access to MCP tools."
       }
     ]
   }
   ```

   Each server configuration supports:
   - `command`: Command to start the MCP server
   - `args`: Command-line arguments for the server
   - `env`: Environment variables for the server
   - `timeout`: Maximum time to wait for server responses (in seconds, default: 3600.0)
   - `disabled`: Optional boolean flag to disable a server without removing its configuration

## Usage

### Using the CLI Command

```bash
dolphin-mcp-cli "Your query here"
```

### Command-line Options

```
Usage: dolphin-mcp-cli [--model <name>] [--quiet] [--config <file>] [--stream] 'your question'

Options:
  --model <name>    Specify the model to use (can be model name or title from config)
  --quiet           Suppress intermediate output
  --config <file>   Specify a custom config file (default: mcp_config.json)
  --stream          Enable streaming mode
  --help, -h       Show this help message
```

### Using the Library

```python
import asyncio
from dolphin_mcp import run_interaction

## No streaming
async def main():

   ## No streaming
   result = await run_interaction(
      user_query="What dolphin species are endangered?",
      model_name="gpt-4o",  # Can use either model name ("gpt-4o") or title ("GPT-4")
      config_path="mcp_config.json",
      quiet_mode=False,
      stream=False
   )
   print(result)

    ## Streaming
   async for chunk in await run_interaction(
      user_query="What dolphin species are endangered?",,
      model_name="gpt-4o",
      stream=True,
      config_path=mcp_config.json,
      quiet_mode=False,
   ):
      print(chunk, end="", Flush=True)

asyncio.run(main())
```

## Architecture

### Package Structure

```
dolphin_mcp/
├── __init__.py
├── client.py          # Core client implementation
├── mcp_client.py      # MCP server client
├── connection_pool.py # Connection management
├── stream_processor.py # Streaming support
├── cli.py            # Command-line interface
├── utils.py          # Utility functions
└── providers/        # Model providers
    ├── openai.py
    ├── anthropic.py
    └── ollama.py
```

### Key Components

1. **Connection Management**
   - Connection pooling with configurable limits
   - Automatic health monitoring
   - Message queuing system
   - Per-server timeout configuration

2. **Error Handling**
   - Automatic recovery from failures
   - Detailed error reporting
   - Graceful degradation

3. **Resource Management**
   - Efficient cleanup
   - Memory optimization
   - Connection reuse

4. **Performance Features**
   - Message queuing prevents concurrent writes
   - Connection health monitoring
   - Efficient resource utilization

## Requirements

- Python 3.8+
- OpenAI API key (or other supported provider API keys)

### Core Dependencies
- openai
- mcp[cli]
- python-dotenv
- anthropic
- ollama
- jsonschema

### Development Dependencies
- pytest
- pytest-asyncio
- pytest-mock
- uv

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
