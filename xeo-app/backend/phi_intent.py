import os
import io
import base64
import json
import numpy as np
from PIL import Image
import time
import tempfile
import logging
import re

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("phi_intent")

# 检查依赖项是否安装
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
    PHI_MODEL_AVAILABLE = True
except ImportError:
    logger.warning("未安装PyTorch或Transformers，将使用模拟模式")
    PHI_MODEL_AVAILABLE = False

# 定义XEO应用工具
xeo_tools = [
    {
        "name": "connect_device",
        "description": "连接或断开XEO应用中的设备",
        "parameters": {
            "device_id": {
                "description": "当前页面能被控制的设备，可以是'xeo-about', 'apple-tv', 'playstation', 'nintendo'中的一个",
                "type": "str",
                "enum": ["xeo-about", "apple-tv", "playstation", "nintendo"]
            }
        }
    },
    {
        "name": "adjust_setting",
        "description": "调整XEO应用中的设置参数",
        "parameters": {
            "setting_id": {
                "description": "当前页面能被控制的设置，可以是'volume', 'ipd', 'magic', 'seat', 'ventilation'中的一个",
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

# 全局变量，用于存储已加载的模型和处理器
_MODEL = None
_PROCESSOR = None
_GENERATION_CONFIG = None

class PhiIntentProcessor:
    """Phi4用户意图处理器"""
    
    def __init__(self, model_path="/home/lab/phi4/phi4", use_local_model=True):
        """
        初始化用户意图处理器
        
        Args:
            model_path: Phi4模型路径
            use_local_model: 是否使用本地模型（如果为False，使用模拟模式）
        """
        self.model_path = model_path
        self.use_local_model = use_local_model and PHI_MODEL_AVAILABLE
        
        # 定义提示词结构
        self.system_prompt_start = '<|system|>'
        self.system_prompt_end = '<|end|>'
        self.user_prompt = '<|user|>'
        self.user_prompt_end = '<|end|>'
        self.assistant_prompt = '<|assistant|>'
        self.assistant_prompt_end = '<|end|>'
        self.tool_call_start = '<|tool_call|>'
        self.tool_call_end = '<|/tool_call|>'
        
        # 存储UI分析缓存
        self.ui_analysis_cache = {}
        
        # 如果使用本地模型，加载模型
        if self.use_local_model:
            self._load_model()
        else:
            logger.info("使用模拟模式，不加载实际模型")
    
    def _load_model(self):
        """加载Phi4模型（如果全局已加载则复用）"""
        global _MODEL, _PROCESSOR, _GENERATION_CONFIG
        
        if _MODEL is not None and _PROCESSOR is not None and _GENERATION_CONFIG is not None:
            logger.info("使用已加载的模型实例")
            self.model = _MODEL
            self.processor = _PROCESSOR
            self.generation_config = _GENERATION_CONFIG
            return
            
        try:
            logger.info(f"正在加载模型: {self.model_path}")
            _PROCESSOR = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
            _MODEL = AutoModelForCausalLM.from_pretrained(
                self.model_path, 
                device_map="cuda", 
                torch_dtype="auto", 
                trust_remote_code=True,
                _attn_implementation='flash_attention_2',
            ).cuda()
            _GENERATION_CONFIG = GenerationConfig.from_pretrained(self.model_path)
            
            # 保存引用
            self.processor = _PROCESSOR
            self.model = _MODEL
            self.generation_config = _GENERATION_CONFIG
            logger.info("模型加载成功")
        except Exception as e:
            logger.error(f"加载模型失败: {str(e)}")
            self.use_local_model = False
    
    def call_model(self, prompt, image=None, max_new_tokens=500, use_tools=False):
        """调用phi4模型进行推理"""
        if not self.use_local_model:
            # 模拟模式
            logger.info(f"模拟模型调用: {prompt[:50]}...")
            time.sleep(1.5)  # 模拟推理延迟
            
            # 生成模拟响应
            if image:
                if use_tools:
                    return f'''{self.tool_call_start}[{{"name":"connect_device","arguments":{{"device_id":"apple-tv"}}}}]{self.tool_call_end}
我可以帮您连接Apple TV设备。''', 1.5
                return "这是一个XEO虚拟现实界面，显示了设备连接状态和各种设置选项。", 1.5
            elif "手势" in prompt:
                if use_tools:
                    return f'''{self.tool_call_start}[{{"name":"adjust_setting","arguments":{{"setting_id":"volume","value":80}}}}]{self.tool_call_end}
我已帮您将音量调整到80%。''', 1.5
                return "根据用户的手势，可能想要调整设置或连接设备", 1.5
            else:
                return "我理解您的指令，请告诉我您想要执行的操作。", 1.5
        
        # 实际模型调用
        logger.info(f"调用模型: {prompt[:50]}...")
        
        # 是否在系统提示中添加工具
        if use_tools:
            # 添加工具信息到提示词
            tools_json = json.dumps(xeo_tools)
            system_prompt = f'''{self.system_prompt_start}
你是一个具备工具调用能力的XEO虚拟现实系统助手，可以控制设备连接和调整设置,你只需要返回工具调用的具体格式。

可用函数：<|tool|>
{tools_json}
<|/tool|><|end|>

函数调用规则:
1. 所有函数调用应以以下格式生成：{self.tool_call_start}[{{"name": "函数名", "arguments": {{"参数"}}}}]{self.tool_call_end}
2. 遵循提供的JSON架构，不要编造参数或值
3. 确保选择正确匹配用户意图的函数
{self.system_prompt_end}'''
            prompt = f"{system_prompt}\n{prompt}"
        
        # 处理输入
        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors='pt'
        ).to('cuda:0')
        
        # 计时并生成响应
        start_time = time.time()
        generate_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            generation_config=self.generation_config,
            num_logits_to_keep=1,
        )
        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
        response = self.processor.batch_decode(
            generate_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        end_time = time.time()
        
        response_time = end_time - start_time
        logger.info(f"响应用时: {response_time:.2f}秒")
        
        return response, response_time
    
    def process_base64_image(self, base64_image):
        """处理Base64编码的图像"""
        try:
            # 去除可能的data URL前缀
            if ',' in base64_image:
                base64_image = base64_image.split(',')[1]
            
            # 解码Base64数据
            image_data = base64.b64decode(base64_image)
            image = Image.open(io.BytesIO(image_data))
            
            return image
        except Exception as e:
            logger.error(f"处理Base64图像时出错: {str(e)}")
            return None
    
    def crop_image_at_gaze(self, image, coordinates, radius):
        """根据眼动坐标和半径裁剪图像"""
        width, height = image.size
        x_pixel = int(coordinates["x"] * width)
        y_pixel = int(coordinates["y"] * height)
        r_pixel = int(radius * min(width, height))
        
        left = max(0, x_pixel - r_pixel)
        top = max(0, y_pixel - r_pixel)
        right = min(width, x_pixel + r_pixel)
        bottom = min(height, y_pixel + r_pixel)
        
        logger.info(f"裁剪坐标: ({left}, {top}, {right}, {bottom})")
        
        # 裁剪图像
        cropped = image.crop((left, top, right, bottom))
        
        # 创建保存目录
        save_dir = os.path.join(os.path.dirname(__file__), "cropped_images")
        os.makedirs(save_dir, exist_ok=True)
        
        # 生成文件名并保存
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"gaze_crop_{timestamp}_{x_pixel}_{y_pixel}.png"
        save_path = os.path.join(save_dir, filename)
        cropped.save(save_path)
        
        logger.info(f"保存裁剪图像到: {save_path}")
        
        return cropped
    
    def analyze_ui(self, image_data):
        """
        分析整体UI界面
        
        Args:
            image_data: 图像数据（文件对象或Base64字符串）
        
        Returns:
            UI分析结果
        """
        # 处理输入图像
        if isinstance(image_data, str) and image_data.startswith(('data:image', 'http')):
            # 处理Base64编码的图像或URL
            if image_data.startswith('data:image'):
                image = self.process_base64_image(image_data)
            else:
                # 未实现URL处理，为简化示例
                return {"error": "不支持URL图像"}
        elif isinstance(image_data, Image.Image):
            # 已经是PIL图像对象
            image = image_data
        else:
            return {"error": "不支持的图像格式"}
        
        if not image:
            return {"error": "无法处理图像"}
        
        # 使用图像指纹作为缓存键
        image_key = hash(image.tobytes())
        if image_key in self.ui_analysis_cache:
            logger.info("使用缓存的UI分析结果")
            return self.ui_analysis_cache[image_key]
        
        # 构建提示词
        prompt = f'''{self.user_prompt}<|image_1|>
分析界面:
详细描述当前页面的功能、主要UI元素及可能的交互方式。
{self.user_prompt_end}
{self.assistant_prompt}'''
        
        # 调用模型
        analysis, analysis_time = self.call_model(prompt, image, max_new_tokens=256)
        
        # 构建结果
        result = {
            "analysis": analysis,
            "response_time": analysis_time
        }
        
        # 缓存结果
        self.ui_analysis_cache[image_key] = result
        
        return result
    
    def parse_tool_calls(self, response_text):
        """从响应文本中解析工具调用"""
        tool_calls = []
        
        # 匹配工具调用
        pattern = re.compile(f'{re.escape(self.tool_call_start)}(.*?){re.escape(self.tool_call_end)}', re.DOTALL)
        matches = pattern.findall(response_text)
        
        if not matches:
            logger.warning("未找到工具调用格式")
            return []
        
        for match in matches:
            try:
                # 尝试解析JSON
                json_str = match.strip()
                # 处理非标准JSON格式，如[{"name": "tool_name"}]
                if json_str.startswith('[') and json_str.endswith(']'):
                    json_data = json.loads(json_str)
                    if isinstance(json_data, list) and len(json_data) > 0:
                        for item in json_data:
                            tool_calls.append(item)
                else:
                    # 尝试作为单个对象解析
                    json_data = json.loads(json_str)
                    if isinstance(json_data, dict):
                        tool_calls.append(json_data)
            except json.JSONDecodeError as e:
                logger.warning(f"无法解析工具调用JSON: {e}\n原始文本: {match}")
                continue
        
        # 验证工具调用格式
        valid_calls = []
        for call in tool_calls:
            if "name" in call and call["name"] in [tool["name"] for tool in xeo_tools]:
                valid_calls.append(call)
            else:
                logger.warning(f"无效的工具调用: {call}")
        
        return valid_calls
    
    def infer_intent(self, image_data, gesture, gaze_data=None):
        """
        完整的意图推理流程
        
        Args:
            image_data: 图像数据（文件对象或Base64字符串）
            gesture: 用户手势 (如 'pinch', 'thumb up')
            gaze_data: 可选的眼动数据 {'x': 0.5, 'y': 0.5, 'radius': 0.1}
        
        Returns:
            包含分析结果的字典
        """
        start_time = time.time()
        logger.info(f"开始处理手势: {gesture}")
        
        # 处理输入图像
        if isinstance(image_data, str) and image_data.startswith(('data:image', 'http')):
            # 处理Base64编码的图像或URL
            if image_data.startswith('data:image'):
                image = self.process_base64_image(image_data)
            else:
                # 未实现URL处理，为简化示例
                return {"error": "不支持URL图像"}
        elif isinstance(image_data, Image.Image):
            # 已经是PIL图像对象
            image = image_data
        else:
            return {"error": "不支持的图像格式"}
        
        if not image:
            return {"error": "无法处理图像"}
        
        # 步骤1: 获取UI整体分析
        ui_analysis = self.analyze_ui(image)
        if "error" in ui_analysis:
            return ui_analysis
        
        # 裁剪眼动关注区域的图像（如果有眼动数据）
        cropped_image = None
        if gaze_data:
            logger.info(f"裁剪眼动点: 坐标({gaze_data['x']:.2f}, {gaze_data['y']:.2f})")
            cropped_image = self.crop_image_at_gaze(
                image, 
                {"x": gaze_data['x'], "y": gaze_data['y']}, 
                gaze_data.get('radius', 0.1)
            )
        
        # 步骤2: 根据手势和UI分析推断意图，使用工具调用
        logger.info(f"推断意图")
        
        # 构建提示词
        prompt = f'''{self.user_prompt}<|image_1|>
当前界面分析: {ui_analysis['analysis']}

用户手势: {gesture}
'''
        
        # 添加眼动信息（如果有）
        if gaze_data:
            prompt += f'''
用户视线位置: 
- X坐标: {gaze_data['x']:.2f}（屏幕范围0-1，0是左边缘，1是右边缘）
- Y坐标: {gaze_data['y']:.2f}（屏幕范围0-1，0是上边缘，1是下边缘）
'''
        
        prompt += f'''
根据界面分析和用户手势（及视线位置），推断用户可能想要执行的操作，并使用合适的工具执行该操作。
{self.user_prompt_end}
{self.assistant_prompt}'''
        
        # 调用模型（使用工具）
        intent_response, intent_time = self.call_model(prompt, image, max_new_tokens=400, use_tools=True)
        
        # 解析工具调用
        tool_calls = self.parse_tool_calls(intent_response)
        
        # 提取意图描述（工具调用之外的文本）
        intent_description = intent_response
        for call in tool_calls:
            call_json = json.dumps(call)
            intent_description = intent_description.replace(f"{self.tool_call_start}[{call_json}]{self.tool_call_end}", "").strip()
        
        # 计算总时间
        total_time = time.time() - start_time
        
        # 构建完整结果
        result = {
            "ui_analysis": ui_analysis["analysis"],
            "gesture": gesture,
            "gaze_data": gaze_data,
            "intent_description": intent_description.strip(),
            "tool_calls": tool_calls,
            "response_time": {
                "ui_analysis": ui_analysis.get("response_time", 0),
                "intent": intent_time,
                "total": total_time
            }
        }
        
        return result


# 单例模式获取处理器
def get_intent_processor(model_path="/home/lab/phi4/phi4", use_local_model=True):
    """获取意图处理器的单例实例"""
    # 使用缓存避免重复加载
    if not hasattr(get_intent_processor, "instance"):
        get_intent_processor.instance = PhiIntentProcessor(
            model_path=model_path,
            use_local_model=use_local_model
        )
    return get_intent_processor.instance


# 测试代码
if __name__ == "__main__":
    processor = get_intent_processor(use_local_model=True)
    
    # 创建测试图像
    test_image = Image.new('RGB', (800, 600), color=(73, 109, 137))
    
    # 分析UI
    ui_result = processor.analyze_ui(test_image)
    print("UI分析结果:", ui_result)
    
    # 推断意图
    intent_result = processor.infer_intent(
        test_image,
        gesture="pinch",
        gaze_data={"x": 0.5, "y": 0.5, "radius": 0.1}
    )
    print("意图推断结果:", intent_result)