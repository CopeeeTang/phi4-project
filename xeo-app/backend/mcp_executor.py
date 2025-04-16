import os
import sys
import json
import requests
from typing import Dict, Any, List, Optional, Tuple
import re

# 导入工具定义
from mcp_tools import get_tool_by_name, validate_tool_parameters

# API地址
API_BASE_URL = "http://localhost:5000/api"

# XEO应用工具定义
xeo_tools = [
    {
        "name": "connect_device",
        "description": "连接或断开XEO应用中的设备",
        "parameters": {
            "device_id": {
                "description": "设备ID，可以是'about-xeo', 'apple-tv', 'playstation', 'nintendo'中的一个",
                "type": "str",
                "enum": ["about-xeo", "apple-tv", "playstation", "nintendo"]
            }
        }
    },
    {
        "name": "adjust_setting",
        "description": "调整XEO应用中的设置参数",
        "parameters": {
            "setting_id": {
                "description": "设置ID，可以是'volume', 'ipd', 'magic', 'seat', 'ventilation'中的一个",
                "type": "str",
                "enum": ["volume", "ipd", "magic", "seat", "ventilation"]
            },
            "value": {
                "description": "设置的新值",
                "type": "int"
            }
        }
    }
]

# 全局设备和设置状态，由app.py维护
devices = {}
settings = {}

def set_state_reference(app_devices, app_settings):
    """从app.py获取设备和设置状态的引用"""
    global devices, settings
    devices = app_devices
    settings = app_settings

class ToolExecutor:
    """Phi4工具执行器类"""
    
    def __init__(self):
        """初始化工具执行器"""
        self.api_base_url = "http://localhost:5000/api"
    
    def execute_tool(self, tool_name, arguments=None):
        """执行指定的工具"""
        if not arguments:
            arguments = {}
            
        result = {"success": False, "message": "未知工具或执行失败"}
        
        if tool_name == "connect_device":
            result = self.execute_connect_device(arguments.get("device_id", ""))
        elif tool_name == "adjust_setting":
            result = self.execute_adjust_setting(
                arguments.get("setting_id", ""), 
                arguments.get("value", 0)
            )
        
        return {
            "tool_name": tool_name,
            "arguments": arguments,
            "result": result,
            "content": [{"type": "text", "text": result.get("message", "")}]
        }
    
    def execute_connect_device(self, device_id):
        """连接或断开设备"""
        if device_id not in devices:
            return {"success": False, "message": f"未知设备ID: {device_id}"}
        
        # 切换连接状态
        current_status = devices[device_id]["connected"]
        devices[device_id]["connected"] = not current_status
        
        new_status = "connected" if devices[device_id]["connected"] else "disconnected"
        status_text = "已连接" if devices[device_id]["connected"] else "已断开"
        
        # 尝试通过API调用
        try:
            response = requests.post(
                f"{self.api_base_url}/devices/{device_id}/connect",
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code != 200:
                # 如果API调用失败，回滚状态更改
                devices[device_id]["connected"] = current_status
                return {"success": False, "message": f"API调用失败: {response.status_code}"}
            
        except Exception as e:
            # 直接使用内存中的状态，不依赖API
            pass
        
        return {
            "success": True,
            "device": devices[device_id],
            "status": new_status,
            "message": f"设备 {devices[device_id]['name']} {status_text}"
        }
    
    def execute_adjust_setting(self, setting_id, value):
        """调整设置参数"""
        if setting_id not in settings:
            return {"success": False, "message": f"未知设置ID: {setting_id}"}
        
        # 验证值的范围
        min_max = {
            "volume": (0, 100),
            "ipd": (50, 80),
            "magic": (0, 100),
            "seat": (0, 100),
            "ventilation": (0, 100)
        }
        
        min_val, max_val = min_max[setting_id]
        if value < min_val or value > max_val:
            return {"success": False, "message": f"设置值超出范围 ({min_val}-{max_val}): {value}"}
        
        # 保存旧值，以便在API调用失败时回滚
        old_value = settings[setting_id]
        
        # 更新设置值
        settings[setting_id] = value
        
        # 获取单位
        units = {
            "volume": "%",
            "ipd": "mm",
            "magic": "%",
            "seat": "",
            "ventilation": "%"
        }
        
        # 尝试通过API调用
        try:
            response = requests.put(
                f"{self.api_base_url}/settings/{setting_id}",
                headers={"Content-Type": "application/json"},
                json={"value": value}
            )
            
            if response.status_code != 200:
                # 如果API调用失败，回滚状态更改
                settings[setting_id] = old_value
                return {"success": False, "message": f"API调用失败: {response.status_code}"}
            
        except Exception as e:
            # 直接使用内存中的状态，不依赖API
            pass
        
        return {
            "success": True,
            "setting_id": setting_id,
            "old_value": old_value, 
            "new_value": value,
            "unit": units[setting_id],
            "message": f"设置 {setting_id} 已从 {old_value}{units[setting_id]} 更改为 {value}{units[setting_id]}"
        }

# 创建工具执行器实例
tool_executor = ToolExecutor()

# 测试函数
def test_tools():
    """测试工具执行"""
    # 获取当前状态
    print("获取当前状态...")
    status = tool_executor.get_current_status()
    print(json.dumps(status, indent=2))
    
    # 测试设备连接
    print("\n测试设备连接...")
    result = tool_executor.execute_tool("connect_about_xeo", {})
    print(json.dumps(result, indent=2))
    
    # 测试设置调节
    print("\n测试设置调节...")
    result = tool_executor.execute_tool("adjust_volume", {"value": 75})
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    test_tools()

def parse_tool_calls(response_text):
    """
    解析模型返回的工具调用，支持Phi4的格式
    """
    tool_calls = []
    
    # 尝试提取 <|tool_call|>[...]<|/tool_call|> 格式
    tool_call_pattern = r'<\|tool_call\|>(.+?)<\|/tool_call\|>'
    tool_call_matches = re.findall(tool_call_pattern, response_text, re.DOTALL)
    
    if tool_call_matches:
        for match in tool_call_matches:
            try:
                # 尝试解析JSON
                calls = json.loads(match)
                if isinstance(calls, list):
                    for call in calls:
                        if isinstance(call, dict) and "name" in call:
                            # 将arguments字段映射到parameters字段
                            if "arguments" in call:
                                call["parameters"] = call.pop("arguments")
                            tool_calls.append(call)
                elif isinstance(calls, dict) and "name" in calls:
                    if "arguments" in calls:
                        calls["parameters"] = calls.pop("arguments")
                    tool_calls.append(calls)
            except json.JSONDecodeError:
                print(f"无法解析JSON: {match}")
    
    # 如果没有找到tool_call标记，尝试直接在文本中查找JSON数组或对象
    if not tool_calls:
        # 尝试查找JSON格式的数组或对象
        json_pattern = r'\[\s*\{.*?\}\s*\]|\{\s*"name"\s*:.*?\}'
        json_matches = re.findall(json_pattern, response_text, re.DOTALL)
        
        for match in json_matches:
            try:
                data = json.loads(match)
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and "name" in item:
                            # 将arguments字段映射到parameters字段
                            if "arguments" in item:
                                item["parameters"] = item.pop("arguments")
                            tool_calls.append(item)
                elif isinstance(data, dict) and "name" in data:
                    if "arguments" in data:
                        data["parameters"] = data.pop("arguments")
                    tool_calls.append(data)
            except json.JSONDecodeError:
                continue
    
    return tool_calls

def analyze_message_keywords(message):
    """
    分析消息中的关键词，用于辅助理解用户意图
    
    Args:
        message: 用户消息
    
    Returns:
        关键词分析结果
    """
    device_keywords = {
        "about-xeo": ["about", "xeo", "关于", "信息", "简介"],
        "apple-tv": ["apple", "tv", "苹果", "电视", "视频", "电影", "播放器"],
        "playstation": ["ps", "ps5", "playstation", "游戏", "索尼", "索尼游戏", "控制台"],
        "nintendo": ["nintendo", "switch", "ns", "任天堂", "游戏", "马里奥", "塞尔达"]
    }
    
    setting_keywords = {
        "volume": ["volume", "音量", "声音", "大小", "静音", "放大", "调小"],
        "ipd": ["ipd", "瞳距", "眼睛", "调整", "视觉", "瞳孔", "眼镜"], 
        "magic": ["magic", "魔法", "pulse", "脉冲", "强度", "体验", "震动"],
        "seat": ["seat", "座椅", "位置", "调整", "高度", "座位", "椅子"],
        "ventilation": ["ventilation", "通风", "风扇", "温度", "调节", "凉爽", "风量"]
    }
    
    action_keywords = {
        "connect": ["connect", "连接", "启动", "打开", "启用", "开始", "使用"],
        "disconnect": ["disconnect", "断开", "关闭", "停止", "关掉", "禁用", "不用"]
    }
    
    results = {
        "devices": [],
        "settings": [],
        "actions": []
    }
    
    # 转为小写并分词
    words = message.lower().split()
    
    # 检测设备关键词
    for device, keywords in device_keywords.items():
        for keyword in keywords:
            if keyword.lower() in message.lower():
                if device not in results["devices"]:
                    results["devices"].append(device)
    
    # 检测设置关键词
    for setting, keywords in setting_keywords.items():
        for keyword in keywords:
            if keyword.lower() in message.lower():
                if setting not in results["settings"]:
                    results["settings"].append(setting)
    
    # 检测动作关键词
    for action, keywords in action_keywords.items():
        for keyword in keywords:
            if keyword.lower() in message.lower():
                if action not in results["actions"]:
                    results["actions"].append(action)
    
    # 检测数值
    value_pattern = r'\b(\d+)(?:\s*(?:%|mm|percent|millimeter|百分比|毫米))?\b'
    value_matches = re.findall(value_pattern, message)
    if value_matches:
        results["values"] = [int(v) for v in value_matches]
    
    return results