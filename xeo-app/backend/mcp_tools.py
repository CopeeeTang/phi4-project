import json
from typing import Dict, Any, List, Optional

# 定义设备连接工具
device_tools = [
    {
        "name": "connect_about_xeo",
        "description": "连接或断开About XEO设备。连接状态会切换（已连接则断开，已断开则连接）。",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "connect_apple_tv",
        "description": "连接或断开Apple TV设备。连接状态会切换（已连接则断开，已断开则连接）。",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "connect_playstation",
        "description": "连接或断开PlayStation 5设备。连接状态会切换（已连接则断开，已断开则连接）。",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "connect_nintendo",
        "description": "连接或断开Nintendo Switch设备。连接状态会切换（已连接则断开，已断开则连接）。",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    }
]

# 定义设置调节工具
setting_tools = [
    {
        "name": "adjust_volume",
        "description": "调节音量大小，范围为0-100%",
        "parameters": {
            "type": "object",
            "properties": {
                "value": {
                    "type": "integer",
                    "description": "音量大小，范围0-100",
                    "minimum": 0,
                    "maximum": 100
                }
            },
            "required": ["value"]
        }
    },
    {
        "name": "adjust_ipd",
        "description": "调节IPD(瞳距)设置，范围为50-80mm",
        "parameters": {
            "type": "object",
            "properties": {
                "value": {
                    "type": "integer",
                    "description": "IPD数值，范围50-80",
                    "minimum": 50,
                    "maximum": 80
                }
            },
            "required": ["value"]
        }
    },
    {
        "name": "adjust_magic",
        "description": "调节Magic Pulse强度，范围为0-100%",
        "parameters": {
            "type": "object",
            "properties": {
                "value": {
                    "type": "integer",
                    "description": "Magic Pulse强度，范围0-100",
                    "minimum": 0,
                    "maximum": 100
                }
            },
            "required": ["value"]
        }
    },
    {
        "name": "adjust_seat",
        "description": "调节座椅位置，范围为0-100",
        "parameters": {
            "type": "object",
            "properties": {
                "value": {
                    "type": "integer",
                    "description": "座椅位置，范围0-100",
                    "minimum": 0,
                    "maximum": 100
                }
            },
            "required": ["value"]
        }
    },
    {
        "name": "adjust_ventilation",
        "description": "调节通风大小，范围为0-100%",
        "parameters": {
            "type": "object",
            "properties": {
                "value": {
                    "type": "integer",
                    "description": "通风大小，范围0-100",
                    "minimum": 0,
                    "maximum": 100
                }
            },
            "required": ["value"]
        }
    }
]

# 合并所有工具
all_tools = device_tools + setting_tools

# 获取所有工具的定义
def get_all_tools() -> List[Dict[str, Any]]:
    return all_tools

# 获取所有工具名称
def get_tool_names() -> List[str]:
    return [tool["name"] for tool in all_tools]

# 根据工具名称获取工具定义
def get_tool_by_name(name: str) -> Optional[Dict[str, Any]]:
    for tool in all_tools:
        if tool["name"] == name:
            return tool
    return None

# 验证工具参数
def validate_tool_parameters(tool_name: str, parameters: Dict[str, Any]) -> bool:
    tool = get_tool_by_name(tool_name)
    if not tool:
        return False
    
    # 检查必需参数
    required_params = tool.get("parameters", {}).get("required", [])
    for param in required_params:
        if param not in parameters:
            return False
    
    # 检查参数范围（针对设置类工具）
    if "adjust" in tool_name and "value" in parameters:
        value = parameters["value"]
        properties = tool.get("parameters", {}).get("properties", {})
        if "value" in properties:
            minimum = properties["value"].get("minimum")
            maximum = properties["value"].get("maximum")
            
            if minimum is not None and value < minimum:
                return False
            if maximum is not None and value > maximum:
                return False
    
    return True