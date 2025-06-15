from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import json
import inspect
import sys
import requests

# Example tool implementations
def calculate_tool(args: Dict[str, Any]) -> str:
    """Example calculator tool"""
    try:
        operation = args.get("operation", "add")
        a = float(args.get("a", 0))
        b = float(args.get("b", 0))
        
        if operation == "add":
            result = a + b
        elif operation == "subtract":
            result = a - b
        elif operation == "multiply":
            result = a * b
        elif operation == "divide":
            if b == 0:
                raise ValueError("Division by zero")
            result = a / b
        else:
            raise ValueError(f"Unknown operation: {operation}")
        
        return f"Calculation: {a} {operation} {b} = {result}"
    except Exception as e:
        return f"Calculation error: {str(e)}"
    
def register_calculate_tool(mcp_server):
    mcp_server.add_tool(
        name="calculate",
        description="Perform basic mathematical calculations (add, subtract, multiply, divide)",
        input_schema={
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["add", "subtract", "multiply", "divide"],
                    "description": "The mathematical operation to perform"
                },
                "a": {
                    "type": "number",
                    "description": "First number"
                },
                "b": {
                    "type": "number", 
                    "description": "Second number"
                }
            },
            "required": ["operation", "a", "b"]
        },
        handler=calculate_tool
    )
    return mcp_server


def current_time_tool(args: Dict[str, Any]) -> str:
    """Get current time"""
    try:
        format_str = args.get("format", "%Y-%m-%d %H:%M:%S UTC")
        return f"Current time: {datetime.now(timezone.utc).strftime(format_str)}"
    except Exception as e:
        return f"Time error: {str(e)}"
    
def register_current_time_tool(mcp_server):
    mcp_server.add_tool(
        name="current_time",
        description="Get the current date and time in UTC",
        input_schema={
            "type": "object",
            "properties": {
                "format": {
                    "type": "string",
                    "description": "Time format string (default: %Y-%m-%d %H:%M:%S UTC)",
                    "default": "%Y-%m-%d %H:%M:%S UTC"
                }
            }
        },
        handler=current_time_tool
    )
    return mcp_server

def text_analyzer_tool(args: Dict[str, Any]) -> str:
    """Analyze text properties"""
    try:
        text = args.get("text", "")
        analysis = {
            "length": len(text),
            "words": len(text.split()),
            "lines": len(text.split('\n')),
            "characters_no_spaces": len(text.replace(' ', '')),
            "sentences": len([s for s in text.split('.') if s.strip()]),
            "paragraphs": len([p for p in text.split('\n\n') if p.strip()])
        }
        return f"Text Analysis:\n{json.dumps(analysis, indent=2)}"
    except Exception as e:
        return f"Analysis error: {str(e)}"
    
def register_text_analyzer_tool(mcp_server):
    mcp_server.add_tool(
        name="analyze_text",
        description="Analyze text and return detailed statistics including length, words, lines, sentences, and paragraphs",
        input_schema={
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Text to analyze"
                }
            },
            "required": ["text"]
        },
        handler=text_analyzer_tool
    )
    return mcp_server

def call_bytegenie_api(endpoint, args):
    url = "https://api.byte-genie.com/execute"
    payload = {
        "api_key": "bg_b71a85f7-04e1-4fbe-97df-298f04ef5be5",
        "tasks": {"task_1": {
                "func": endpoint,
                "args": args,
                "username": "testing-genie",
                "overwrite": 0,
                "task_mode": "sync"
            }}
    }
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "insomnia/10.3.0"
    }

    response = requests.request("POST", url, json=payload, headers=headers)
    return response.json()["response"]["task_1"]["data"]


def register_list_events_tool(mcp_server):
    def handler(args: Dict[str, Any]) -> Any:
        try:
            return call_bytegenie_api(endpoint="list_events", args=args)
        except Exception as e:
            return {"error": str(e)}

    mcp_server.add_tool(
        name="list_events",
        description="Fetch a list of events filtered by date, city, country, or event name",
        input_schema={
            "type": "object",
            "properties": {
                "event_min_date": {
                    "type": "string",
                    "description": "Minimum date for events (YYYY-MM-DD)",
                },
                "event_max_date": {
                    "type": "string",
                    "description": "Maximum date for events (YYYY-MM-DD)",
                },
                "event_city": {
                    "type": "string",
                    "description": "City where event is held",
                },
                "event_country_std": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of standardized country codes (e.g., ['USA', 'GBR'])",
                },
                "event_name": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of event names to search",
                },
            },
            "required": []
        },
        handler=handler
    )
    return mcp_server

def register_query_knowledge_graph_tool(mcp_server):
    def handler(args: Dict[str, Any]) -> Any:
        try:
            return call_bytegenie_api(endpoint="query_knowledge_graph", args=args)
        except Exception as e:
            return {"error": str(e)}

    mcp_server.add_tool(
        name="query_knowledge_graph",
        description="Searches the knowledge graph for entities based on a natural language query.",
        input_schema={
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "The search query.",
                },
            },
            "required": ["prompt"]
        },
        handler=handler
    )
    return mcp_server


def register_all_tools(mcp_server):
    current_module = sys.modules[__name__]
    
    for name, func in inspect.getmembers(current_module, inspect.isfunction):
        if name.startswith("register_") and name != "register_all_tools":
            func(mcp_server)
            print(f"{name.split('register_')[1]} registered successfully!")
    
    return mcp_server
