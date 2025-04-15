"""
OpenAI provider implementation for Dolphin MCP.
"""

import os
import json
import asyncio
import logging
from typing import Dict, List, Any, AsyncGenerator, Optional, Union, Callable, TypeVar
from openai import AsyncOpenAI, APIError, RateLimitError

logger = logging.getLogger("mcplex")

T = TypeVar('T')

async def retry_with_exponential_backoff(
    operation: Callable[..., T],
    max_retries: int = 5,
    initial_delay: float = 1,
    max_delay: float = 60,
    exponential_base: float = 2,
    *args,
    **kwargs
) -> T:
    """
    Retry an async operation with exponential backoff.
    
    Args:
        operation: The async function to retry
        max_retries: Maximum number of retries
        initial_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        exponential_base: Base for exponential backoff
        *args, **kwargs: Arguments to pass to the operation
    """
    delay = initial_delay
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            return await operation(*args, **kwargs)
        except RateLimitError as e:
            last_exception = e
            retry_after = float(e.headers.get('retry-after', delay)) if hasattr(e, 'headers') else delay
            delay = min(max_delay, retry_after)
            logger.warning(f"Rate limit hit, retrying in {delay} seconds (attempt {attempt + 1}/{max_retries})")
        except APIError as e:
            last_exception = e
            # Safely check if the error has a should_retry method before calling it
            if not hasattr(e, 'should_retry') or not e.should_retry():
                raise
            delay = min(max_delay, delay * exponential_base)
            logger.warning(f"API error, retrying in {delay} seconds (attempt {attempt + 1}/{max_retries})")
        except Exception as e:
            last_exception = e
            if "overloaded" not in str(e).lower() and "timeout" not in str(e).lower():
                raise
            delay = min(max_delay, delay * exponential_base)
            logger.warning(f"Server error, retrying in {delay} seconds (attempt {attempt + 1}/{max_retries})")
            
        await asyncio.sleep(delay)
    
    raise last_exception

async def generate_with_openai_stream(client: AsyncOpenAI, model_name: str, conversation: List[Dict],
                                    formatted_functions: List[Dict], temperature: Optional[float] = None,
                                    top_p: Optional[float] = None, max_tokens: Optional[int] = None) -> AsyncGenerator:
    """Internal function for streaming generation"""    
    try:
        async def create_stream():
            return await client.chat.completions.create(
                model=model_name,
                messages=conversation,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                tools=[{"type": "function", "function": f} for f in formatted_functions],
                tool_choice="auto",
                stream=True
            )
            
        response = await retry_with_exponential_backoff(create_stream)

        current_tool_calls = []
        current_content = ""

        async for chunk in response:
            delta = chunk.choices[0].delta
            
            if delta.content:
                yield {"assistant_text": delta.content, "tool_calls": [], "is_chunk": True, "token": True}
                current_content += delta.content

            if delta.tool_calls:
                for tool_call in delta.tool_calls:
                    while tool_call.index >= len(current_tool_calls):
                        current_tool_calls.append({
                            "id": "",
                            "function": {
                                "name": "",
                                "arguments": ""
                            }
                        })
                    
                    current_tool = current_tool_calls[tool_call.index]
                    
                    if tool_call.id:
                        current_tool["id"] = tool_call.id
                    
                    if tool_call.function.name:
                        current_tool["function"]["name"] = (
                            current_tool["function"]["name"] + tool_call.function.name
                        )
                    
                    if tool_call.function.arguments:
                        current_args = current_tool["function"]["arguments"]
                        new_args = tool_call.function.arguments
                        
                        if new_args.startswith("{") and not current_args:
                            current_tool["function"]["arguments"] = new_args
                        elif new_args.endswith("}") and current_args:
                            if not current_args.endswith("}"):
                                current_tool["function"]["arguments"] = current_args + new_args
                        else:
                            current_tool["function"]["arguments"] += new_args

            if chunk.choices[0].finish_reason is not None:
                final_tool_calls = []
                for tc in current_tool_calls:
                    if tc["id"] and tc["function"]["name"]:
                        try:
                            args = tc["function"]["arguments"].strip()
                            if not args or args.isspace():
                                args = "{}"
                            parsed_args = json.loads(args)
                            tc["function"]["arguments"] = json.dumps(parsed_args)
                            final_tool_calls.append(tc)
                        except json.JSONDecodeError:
                            args = tc["function"]["arguments"].strip()
                            args = args.rstrip(",")
                            if not args.startswith("{"):
                                args = "{" + args
                            if not args.endswith("}"):
                                args = args + "}"
                            try:
                                parsed_args = json.loads(args)
                                tc["function"]["arguments"] = json.dumps(parsed_args)
                                final_tool_calls.append(tc)
                            except json.JSONDecodeError:
                                tc["function"]["arguments"] = "{}"
                                final_tool_calls.append(tc)

                # Only yield final message if there are tool calls, and omit content since it was already streamed
                if final_tool_calls:
                    yield {
                        "assistant_text": "",  # Content already streamed
                        "tool_calls": final_tool_calls,
                        "is_chunk": False
                    }
    except Exception as e:
        yield {"assistant_text": f"OpenAI error: {str(e)}", "tool_calls": [], "is_chunk": False}

async def generate_with_openai_sync(client: AsyncOpenAI, model_name: str, conversation: List[Dict], 
                                  formatted_functions: List[Dict], temperature: Optional[float] = None,
                                  top_p: Optional[float] = None, max_tokens: Optional[int] = None) -> Dict:
    """Internal function for non-streaming generation"""
    try:
        async def create_completion():
            return await client.chat.completions.create(
                model=model_name,
                messages=conversation,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                tools=[{"type": "function", "function": f} for f in formatted_functions],
                tool_choice="auto",
                stream=False
            )
            
        response = await retry_with_exponential_backoff(create_completion)

        choice = response.choices[0]
        assistant_text = choice.message.content or ""
        tool_calls = []
        
        if hasattr(choice.message, 'tool_calls') and choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                if tc.type == 'function':
                    tool_call = {
                        "id": tc.id,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments or "{}"
                        }
                    }
                    try:
                        json.loads(tool_call["function"]["arguments"])
                    except json.JSONDecodeError:
                        tool_call["function"]["arguments"] = "{}"
                    tool_calls.append(tool_call)
        return {"assistant_text": assistant_text, "tool_calls": tool_calls}

    except APIError as e:
        return {"assistant_text": f"OpenAI API error: {str(e)}", "tool_calls": []}
    except RateLimitError as e:
        return {"assistant_text": f"OpenAI rate limit: {str(e)}", "tool_calls": []}
    except Exception as e:
        return {"assistant_text": f"Unexpected OpenAI error: {str(e)}", "tool_calls": []}

async def generate_with_openai(conversation: List[Dict], model_cfg: Dict,
                              all_functions: List[Dict], stream: bool = False) -> Union[Dict, AsyncGenerator]:
    """
    Generate text using OpenAI's API.
    
    Args:
        conversation: The conversation history
        model_cfg: Configuration for the model
        all_functions: Available functions for the model to call
        stream: Whether to stream the response
        
    Returns:
        If stream=False: Dict containing assistant_text and tool_calls
        If stream=True: AsyncGenerator yielding chunks of assistant text and tool calls
    """
    api_key = model_cfg.get("apiKey") or os.getenv("OPENAI_API_KEY")

    # Time client initialization
    if "apiBase" in model_cfg:
        client = AsyncOpenAI(
            api_key=api_key,
            base_url=model_cfg["apiBase"],
        )
    else:
        client = AsyncOpenAI(api_key=api_key)

    model_name = model_cfg["model"]
    temperature = model_cfg.get("temperature", None)
    top_p = model_cfg.get("top_p", None)
    max_tokens = model_cfg.get("max_tokens", None)

    # Create a mapping from original to sanitized names
    name_mapping = {}
    
    # Format functions for OpenAI API
    formatted_functions = []
    for func in all_functions:
        # Sanitize function name to ensure it matches API requirements
        # Replace any characters that might not match the regex pattern with underscores
        # Ensure the name only contains alphanumeric characters, underscores, and hyphens
        original_name = func["name"]
        sanitized_name = ''.join(c if c.isalnum() or c in '_-' else '_' for c in original_name)
        
        # Store the mapping
        name_mapping[original_name] = sanitized_name
        
        formatted_func = {
            "name": sanitized_name,
            "description": f"{original_name} - {func['description']}",  # Include original name in description
            "parameters": func["parameters"]
        }
        formatted_functions.append(formatted_func)
    
    # Sanitize tool names in conversation history
    sanitized_conversation = []
    for message in conversation:
        sanitized_message = message.copy()
        
        # Check if the message has tool_calls
        if "tool_calls" in message:
            sanitized_tool_calls = []
            for tool_call in message["tool_calls"]:
                sanitized_tool_call = tool_call.copy()
                
                # Sanitize function name if present
                if "function" in tool_call:
                    sanitized_function = tool_call["function"].copy()
                    if "name" in sanitized_function:
                        original_name = sanitized_function["name"]
                        if original_name in name_mapping:
                            sanitized_function["name"] = name_mapping[original_name]
                        else:
                            # If not in mapping, sanitize directly
                            sanitized_function["name"] = ''.join(c if c.isalnum() or c in '_-' else '_' for c in original_name)
                    sanitized_tool_call["function"] = sanitized_function
                
                sanitized_tool_calls.append(sanitized_tool_call)
            
            sanitized_message["tool_calls"] = sanitized_tool_calls
        
        # Check for content that might contain tool use
        if "content" in message and isinstance(message["content"], list):
            sanitized_content = []
            for content_item in message["content"]:
                if isinstance(content_item, dict) and "toolUse" in content_item:
                    sanitized_item = content_item.copy()
                    tool_use = content_item["toolUse"].copy()
                    
                    if "name" in tool_use:
                        original_name = tool_use["name"]
                        if original_name in name_mapping:
                            tool_use["name"] = name_mapping[original_name]
                        else:
                            # If not in mapping, sanitize directly
                            tool_use["name"] = ''.join(c if c.isalnum() or c in '_-' else '_' for c in original_name)
                    
                    sanitized_item["toolUse"] = tool_use
                    sanitized_content.append(sanitized_item)
                else:
                    sanitized_content.append(content_item)
            
            sanitized_message["content"] = sanitized_content
        
        sanitized_conversation.append(sanitized_message)

    if stream:
        return generate_with_openai_stream(
            client, model_name, sanitized_conversation, formatted_functions,
            temperature, top_p, max_tokens
        )
    else:
        return await generate_with_openai_sync(
            client, model_name, sanitized_conversation, formatted_functions,
            temperature, top_p, max_tokens
        )
