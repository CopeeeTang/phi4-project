from flask import Flask, jsonify, request, send_from_directory, render_template
from flask_cors import CORS
from flask_socketio import SocketIO
import json
import os
import sys
import re
from dotenv import load_dotenv
import base64
from PIL import Image
import io
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")

# 加载环境变量
load_dotenv()

app = Flask(__name__)
CORS(app)  # 允许跨域请求
socketio = SocketIO(app, cors_allowed_origins="*")

# 获取前端文件目录的绝对路径
frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# 初始化设备状态
devices = {
    "about-xeo": {"connected": False, "name": "About XEO"},
    "apple-tv": {"connected": False, "name": "Apple TV"},
    "playstation": {"connected": False, "name": "Play Station 5"},
    "nintendo": {"connected": False, "name": "Nintendo Switch"}
}

# 初始化设置状态
settings = {
    "volume": 80,
    "ipd": 65,
    "magic": 80,
    "seat": 50,
    "ventilation": 100
}

# 保存对话历史记录
conversation_history = []

# 导入MCP工具执行器
try:
    from mcp_executor import set_state_reference, tool_executor, parse_tool_calls
    # 传递状态引用
    set_state_reference(devices, settings)
    logger.info("已加载MCP工具执行器")
except ImportError as e:
    logger.error(f"导入MCP工具执行器失败: {str(e)}")
    tool_executor = None

# 导入Phi4意图处理器
try:
    phi_model_path = os.environ.get("PHI_MODEL_PATH", "/home/lab/phi4/phi4")
    use_local_model = os.environ.get("USE_LOCAL_MODEL", "True").lower() == "true"
    
    from phi_intent import get_intent_processor
    intent_processor = get_intent_processor(model_path=phi_model_path, use_local_model=use_local_model)
    logger.info(f"已加载Phi4意图处理器，使用模型路径: {phi_model_path}")
except ImportError as e:
    logger.error(f"导入Phi4意图处理器失败: {str(e)}")
    intent_processor = None

# 路由：提供前端文件
@app.route('/')
def index():
    return send_from_directory(frontend_dir, 'index.html')

# 路由：提供其他静态文件（CSS、JS等）
@app.route('/<path:path>')
def static_files(path):
    return send_from_directory(frontend_dir, path)

# 路由：获取所有设备状态
@app.route('/api/devices', methods=['GET'])
def get_devices():
    return jsonify(devices)

# 路由：获取特定设备状态
@app.route('/api/devices/<device_id>', methods=['GET'])
def get_device(device_id):
    if device_id in devices:
        return jsonify(devices[device_id])
    return jsonify({"error": "Device not found"}), 404

# 路由：连接/断开设备
@app.route('/api/devices/<device_id>/connect', methods=['POST'])
def connect_device(device_id):
    if device_id not in devices:
        return jsonify({"error": "Device not found"}), 404
    
    # 切换连接状态
    devices[device_id]['connected'] = not devices[device_id]['connected']
    status = "connected" if devices[device_id]['connected'] else "disconnected"
    
    # 通过WebSocket广播状态变化
    socketio.emit('device_status_change', {
        'device_id': device_id,
        'connected': devices[device_id]['connected']
    })
    
    return jsonify({
        "status": status,
        "device": devices[device_id]
    })

# 路由：获取所有设置
@app.route('/api/settings', methods=['GET'])
def get_settings():
    return jsonify(settings)

# 路由：更新特定设置
@app.route('/api/settings/<setting_id>', methods=['PUT'])
def update_setting(setting_id):
    if setting_id not in settings:
        return jsonify({"error": "Setting not found"}), 404
    
    data = request.get_json()
    if 'value' not in data:
        return jsonify({"error": "Value is required"}), 400
    
    try:
        # 确保值是整数
        value = int(data['value'])
        
        # 根据不同设置类型进行验证
        if setting_id == "volume" and 0 <= value <= 100:
            settings[setting_id] = value
        elif setting_id == "ipd" and 50 <= value <= 80:
            settings[setting_id] = value
        elif setting_id == "magic" and 0 <= value <= 100:
            settings[setting_id] = value
        elif setting_id == "seat" and 0 <= value <= 100:
            settings[setting_id] = value
        elif setting_id == "ventilation" and 0 <= value <= 100:
            settings[setting_id] = value
        else:
            return jsonify({"error": "Invalid value range"}), 400
        
        # 通过WebSocket广播设置变化
        socketio.emit('setting_change', {
            'setting_id': setting_id,
            'value': settings[setting_id]
        })
        
        return jsonify({
            "setting_id": setting_id,
            "value": settings[setting_id]
        })
    except ValueError:
        return jsonify({"error": "Value must be a number"}), 400

# ===========================================
# MCP集成路由
# ===========================================

# 路由：MCP聊天界面
@app.route('/mcp')
def mcp_chat():
    return render_template('mcp_chat.html')

# 路由：多模态交互界面
@app.route('/phi')
def phi_interact():
    return render_template('phi_interact.html')

# 路由：MCP工具调用
@app.route('/api/mcp/chat', methods=['POST'])
def mcp_chat_api():
    data = request.get_json()
    
    if not data or 'message' not in data:
        return jsonify({"error": "Message is required"}), 400
    
    user_message = data['message']
    
    # 添加用户消息到历史记录
    conversation_history.append({"role": "user", "content": user_message})
    
    try:
        # 使用phi_intent处理器分析并处理工具调用
        if intent_processor is not None:
            # 构建提示词，添加工具信息
            prompt = f"<|user|>{user_message}<|end|>"
            
            # 调用工具处理模式
            response, tools_called = process_chat_with_tools(prompt)
            
            # 获取生成的响应文本
            assistant_message = response  
        else:
            # 如果没有phi_intent处理器，使用简单回复
            assistant_message = generate_fallback_message(user_message)
            tools_called = []
    
    except Exception as e:
        # 处理任何可能发生的错误
        assistant_message = f"在处理您的请求时出错: {str(e)}"
        print(f"Error processing message: {str(e)}")
        tools_called = []
    
    # 添加助手回复到历史记录
    conversation_history.append({"role": "assistant", "content": assistant_message})
    
    # 只保留最近的10条对话
    if len(conversation_history) > 20:
        conversation_history.pop(0)
        conversation_history.pop(0)
    
    return jsonify({
        "message": assistant_message,
        "tools_called": tools_called
    })

def process_chat_with_tools(prompt):
    """使用phi_intent处理器处理带工具的聊天请求"""
    # 调用模型进行推理
    response_text, _ = intent_processor.call_model(prompt, image=None, max_new_tokens=250, use_tools=True)
    
    # 解析工具调用
    tool_calls = intent_processor.parse_tool_calls(response_text)
    
    # 执行工具调用
    if tool_calls:
        tool_results = []
        for tool_call in tool_calls:
            try:
                # 调用工具
                result = execute_tool(
                    tool_call["name"], 
                    tool_call.get("parameters", {})
                )
                tool_results.append(result)
            except Exception as e:
                print(f"执行工具调用时出错: {str(e)}")
        
        # 提取响应文本（去除工具调用部分）
        import re
        response_text = re.sub(r'<\|tool_call\|>.*?<\|/tool_call\|>', '', response_text, flags=re.DOTALL).strip()
    
    return response_text, [tool.get("name") for tool in tool_calls]

def execute_tool(tool_name, parameters):
    """执行工具调用"""
    if not tool_executor:
        return {"error": "工具执行器未初始化"}
    
    # 调用工具执行器处理
    device_mapping = {
        "connect_device": {
            "about-xeo": "connect_about_xeo",
            "apple-tv": "connect_apple_tv",
            "playstation": "connect_playstation",
            "nintendo": "connect_nintendo"
        },
        "adjust_setting": {
            "volume": "adjust_volume",
            "ipd": "adjust_ipd",
            "magic": "adjust_magic",
            "seat": "adjust_seat",
            "ventilation": "adjust_ventilation"
        }
    }
    
    try:
        if tool_name == "connect_device":
            device_id = parameters.get("device_id", "")
            mapped_tool = device_mapping["connect_device"].get(device_id)
            if mapped_tool:
                return tool_executor.execute_tool(mapped_tool, {})
        
        elif tool_name == "adjust_setting":
            setting_id = parameters.get("setting_id", "")
            value = parameters.get("value", 0)
            mapped_tool = device_mapping["adjust_setting"].get(setting_id)
            if mapped_tool:
                return tool_executor.execute_tool(mapped_tool, {"value": value})
    
    except Exception as e:
        return {"error": str(e)}
    
    return {"error": "未实现的工具或无效参数"}

def generate_fallback_message(user_message):
    """生成备用回复，当没有工具调用时使用"""
    if "你好" in user_message or "您好" in user_message or "hi" in user_message.lower() or "hello" in user_message.lower():
        return "您好！我是XEO智能助手。我可以帮您连接设备或调整设置。例如，您可以说'连接Apple TV'或'调整音量到75%'。"
    
    elif "功能" in user_message or "能做什么" in user_message or "help" in user_message.lower():
        capabilities = [
            "1. 连接/断开设备：About XEO、Apple TV、PlayStation 5、Nintendo Switch",
            "2. 调节设置：音量(0-100%)、IPD瞳距(50-80mm)、Magic Pulse强度(0-100%)、座椅位置(0-100)、通风大小(0-100%)"
        ]
        return "我可以帮您控制XEO设备和调整设置，具体功能包括：\n" + "\n".join(capabilities)
    
    else:
        return "抱歉，我不确定您想要做什么。您可以尝试连接设备（如'连接Apple TV'）或调整设置（如'将音量调到80%'）。"

# WebSocket事件：客户端连接
@socketio.on('connect')
def handle_connect():
    print('Client connected')

# WebSocket事件：客户端断开
@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

# 创建必要的模板文件
def create_templates():
    """创建必要的模板文件"""
    os.makedirs(os.path.join(app.root_path, 'templates'), exist_ok=True)
    
    # 创建多模态交互页面模板
    phi_template_path = os.path.join(app.root_path, 'templates', 'phi_interact.html')
    if not os.path.exists(phi_template_path):
        with open(phi_template_path, 'w') as f:
            f.write("""<!DOCTYPE html>
<html>
<head>
    <title>Phi4 多模态交互</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="/css/styles.css">
</head>
<body>
    <div class="container">
        <h1>Phi4 多模态交互</h1>
        <div id="image-container">
            <div id="dropzone">
                <p>拖放图像到此处或点击上传</p>
                <input type="file" id="fileInput" accept="image/*" style="display: none;">
            </div>
            <div id="preview" style="display: none;">
                <img id="previewImage" src="">
                <button id="clearImage">清除图像</button>
            </div>
        </div>
        
        <div id="interaction-container">
            <div id="gesture-controls">
                <h3>手势控制</h3>
                <select id="gestureSelect">
                    <option value="pinch">捏合 (Pinch)</option>
                    <option value="thumbs_up">竖起大拇指 (Thumbs Up)</option>
                    <option value="point">指向 (Point)</option>
                    <option value="swipe">滑动 (Swipe)</option>
                    <option value="rotate">旋转 (Rotate)</option>
                </select>
            </div>
            
            <div id="gaze-controls">
                <h3>视线位置</h3>
                <canvas id="gazeCanvas" width="300" height="200"></canvas>
                <div id="gaze-info">
                    <p>X: <span id="gazeX">0.5</span></p>
                    <p>Y: <span id="gazeY">0.5</span></p>
                    <p>半径: <input type="range" id="gazeRadius" min="0.05" max="0.3" step="0.01" value="0.15"></p>
                </div>
            </div>
            
            <button id="analyzeButton" disabled>分析意图</button>
        </div>
        
        <div id="result-container" style="display: none;">
            <h2>分析结果</h2>
            <div id="result-content"></div>
        </div>
    </div>
    
    <script src="/js/phi_interact.js"></script>
</body>
</html>""")
    
    # 创建MCP聊天页面模板
    mcp_template_path = os.path.join(app.root_path, 'templates', 'mcp_chat.html')
    if not os.path.exists(mcp_template_path):
        with open(mcp_template_path, 'w') as f:
            f.write("""<!DOCTYPE html>
<html>
<head>
    <title>XEO MCP聊天</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="/css/styles.css">
</head>
<body>
    <div class="container">
        <h1>XEO MCP聊天</h1>
        <div id="chat-container">
            <div id="chat-messages">
                <div class="message assistant">
                    <div class="message-content">
                        您好！我是XEO智能助手。我可以帮您连接设备或调整设置。例如，您可以说"连接Apple TV"或"调整音量到75%"。
                    </div>
                </div>
            </div>
            <div id="chat-input-container">
                <input type="text" id="chat-input" placeholder="输入您的消息...">
                <button id="send-button">发送</button>
            </div>
        </div>
    </div>
    
    <script src="/js/mcp_chat.js"></script>
</body>
</html>""")

def handle_state_change(change_type, change_data):
    """处理状态变化，并通过WebSocket发送更新"""
    if change_type == "device":
        device_id = change_data.get("device_id")
        connected = change_data.get("connected")
        
        if device_id in devices:
            devices[device_id]["connected"] = connected
            
            # 广播状态更新
            socketio.emit('device_status_change', {
                'device_id': device_id,
                'connected': connected
            })
    
    elif change_type == "setting":
        setting_id = change_data.get("setting_id")
        value = change_data.get("value")
        
        if setting_id in settings:
            settings[setting_id] = value
            
            # 广播状态更新
            socketio.emit('setting_change', {
                'setting_id': setting_id,
                'value': value
            })

# 路由：统一首页路由
@app.route('/index.html')
def index_html():
    return send_from_directory(frontend_dir, 'index.html')

# 路由：Phi4 UI分析接口
@app.route('/api/phi/analyze_ui', methods=['POST'])
def analyze_ui():
    # 检查是否有图像文件
    if not request.json or 'image' not in request.json:
        return jsonify({"error": "未提供图像数据"}), 400
    
    image_data = request.json['image']
    
    # 检查intent_processor是否可用
    if not intent_processor:
        return jsonify({"error": "Phi4意图处理器未初始化"}), 500
    
    try:
        # 处理图像数据
        image = intent_processor.process_base64_image(image_data)
        if not image:
            return jsonify({"error": "无法处理图像数据"}), 400
        
        # 分析UI
        result = intent_processor.analyze_ui(image)
        
        if "error" in result:
            return jsonify({"error": result["error"]}), 500
        
        # 返回分析结果
        return jsonify({
            "success": True,
            "analysis": result["analysis"],
            "response_time": result.get("response_time", 0)
        })
    
    except Exception as e:
        logger.error(f"UI分析错误: {str(e)}")
        return jsonify({"error": f"分析UI时出错: {str(e)}"}), 500

# 路由：Phi4意图分析接口
@app.route('/api/phi/intent', methods=['POST'])
def phi_intent_analysis():
    # 检查是否有图像文件
    if not request.json or 'image' not in request.json:
        return jsonify({"error": "未提供图像数据"}), 400
    
    image_data = request.json['image']
    gesture = request.json.get('gesture', 'unknown')
    gaze_data = request.json.get('gaze', None)
    
    # 检查intent_processor是否可用
    if not intent_processor:
        return jsonify({"error": "Phi4意图处理器未初始化"}), 500
    
    try:
        # 处理图像数据
        image = intent_processor.process_base64_image(image_data)
        if not image:
            return jsonify({"error": "无法处理图像数据"}), 400
        
        # 进行意图分析
        result = intent_processor.infer_intent(image, gesture, gaze_data)
        
        if "error" in result:
            return jsonify({"error": result["error"]}), 500
        
        # 重新编码裁剪图像（如果有）
        if "cropped_images" in result:
            # 移除不可JSON序列化的图像对象
            for cropped in result["cropped_images"]:
                if "cropped_image" in cropped:
                    del cropped["cropped_image"]
        
        # 返回分析结果
        return jsonify({
            "success": True,
            "ui_analysis": result.get("ui_analysis", ""),
            "intent_description": result.get("intent_description", ""),
            "tool_calls": result.get("tool_calls", []),
            "response_time": result.get("response_time", {})
        })
    
    except Exception as e:
        logger.error(f"意图分析错误: {str(e)}")
        return jsonify({"error": f"分析意图时出错: {str(e)}"}), 500

# 确保模板存在
create_templates()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    socketio.run(app, host="0.0.0.0", port=port, debug=True)