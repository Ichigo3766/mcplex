"""
Ollama provider implementation for Dolphin MCP.
"""

import re
import json
from typing import Dict, List, Any

async def generate_with_ollama(conversation, model_cfg, all_functions):
    """
    Generate text using Ollama's API.
    
    Args:
        conversation: The conversation history
        model_cfg: Configuration for the model
        all_functions: Available functions for the model to call (not used by Ollama)
        
    Returns:
        Dict containing assistant_text and tool_calls
    """
    from ollama import chat, ResponseError

    model_name = model_cfg["model"]
    
    # Prepare options dictionary for Ollama
    options = {}
    if "temperature" in model_cfg:
        options["temperature"] = model_cfg.get("temperature", 0.7)
    if "top_k" in model_cfg:
        options["top_k"] = model_cfg.get("top_k")
    if "repetition_penalty" in model_cfg:
        options["repeat_penalty"] = model_cfg.get("repetition_penalty")
    if "max_tokens" in model_cfg:
        options["num_predict"] = model_cfg.get("max_tokens", 1024)

    try:
        # Add tool information to system message
        system_message = None
        tool_desc = "\n\nAvailable tools:\n"
        for func in all_functions:
            tool_desc += f"- {func['name']}: {func['description']}\n"
        tool_desc += "\nTo use a tool, format your response like this:\n<tool>tool_name{arguments in JSON format}</tool>"
        
        for msg in conversation:
            if msg["role"] == "system":
                msg["content"] = msg["content"] + tool_desc
                system_message = msg
                break
        
        if not system_message and all_functions:
            conversation.insert(0, {
                "role": "system",
                "content": f"You are a helpful assistant with access to tools.{tool_desc}"
            })

        response = chat(
            model=model_name,
            messages=conversation,
            options=options,
            stream=False
        )
        
        content = response.message.content or ""
        tool_calls = []
        
        # Parse tool calls from response using regex
        tool_pattern = r'<tool>(\w+)\s*({[^}]+})</tool>'
        matches = re.finditer(tool_pattern, content)
        
        for i, match in enumerate(matches):
            tool_name = match.group(1)
            try:
                args = json.loads(match.group(2))
                tool_calls.append({
                    "id": f"call_{i+1}",
                    "function": {
                        "name": tool_name,
                        "arguments": json.dumps(args)
                    }
                })
            except json.JSONDecodeError:
                print(f"Failed to parse tool arguments: {match.group(2)}")
                continue
        
        # Remove tool call syntax from final response
        assistant_text = re.sub(tool_pattern, '', content).strip()
        
        return {"assistant_text": assistant_text, "tool_calls": tool_calls}
    except ResponseError as e:
        return {"assistant_text": f"Ollama error: {str(e)}", "tool_calls": []}
    except Exception as e:
        return {"assistant_text": f"Unexpected Ollama error: {str(e)}", "tool_calls": []}
