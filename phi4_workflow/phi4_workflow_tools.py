import base64
import json
import torch
from PIL import Image
import io
import re
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

class ToolManager:
    """工具管理类，处理工具定义、调用和结果处理"""
    
    def __init__(self):
        """初始化工具管理器"""
        self.tools = self._define_tools()
        
    def _define_tools(self):
        """定义可用的工具列表"""
        tools = [
            {
                "name": "crop_image_at_gaze",
                "description": "根据注视坐标和半径裁剪图像的特定区域，并可选择进行图像预处理",
                "parameters": {
                    "properties": {
                        "image_base64": {
                            "title": "Image Base64",
                            "type": "string"
                        },
                        "coordinates": {
                            "additionalProperties": {
                                "type": "number"
                            },
                            "title": "Coordinates",
                            "type": "object"
                        },
                        "radius": {
                            "default": 0.05,
                            "title": "Radius",
                            "type": "number"
                        },
                        "crop_size": {
                            "anyOf": [
                                {
                                    "additionalProperties": {
                                        "type": "integer"
                                    },
                                    "type": "object"
                                },
                                {
                                    "type": "null"
                                }
                            ],
                            "default": None,
                            "title": "Crop Size"
                        },
                        "apply_preprocessing": {
                            "default": True,
                            "title": "Apply Preprocessing",
                            "type": "boolean"
                        },
                        "preprocessing_options": {
                            "anyOf": [
                                {
                                    "type": "object"
                                },
                                {
                                    "type": "null"
                                }
                            ],
                            "default": None,
                            "title": "Preprocessing Options"
                        }
                    },
                    "required": [
                        "image_base64",
                        "coordinates"
                    ],
                }
            },
            {
                "name": "analyze_gaze_region",
                "description": "分析用户注视的区域，并识别页面可能的功能",
                "parameters": {
                    "properties": {
                        "image_base64": {
                            "title": "Image Base64",
                            "type": "string"
                        },
                        "coordinates": {
                            "additionalProperties": {
                                "type": "number"
                            },
                            "title": "Coordinates",
                            "type": "object"
                        },
                        "page_context": {
                            "anyOf": [
                                {
                                    "type": "string"
                                },
                                {
                                    "type": "null"
                                }
                            ],
                            "default": None,
                            "title": "Page Context"
                        }
                    },
                    "required": [
                        "image_base64",
                        "coordinates"
                    ]
                }
            },
            {
                "name": "interpret_gaze_intent",
                "description": "解释用户的注视意图",
                "parameters": {
                    "properties": {
                        "coordinates": {
                            "additionalProperties": {
                                "type": "number"
                            },
                            "title": "Coordinates",
                            "type": "object"
                        },
                        "screen_area": {
                            "anyOf": [
                                {
                                    "type": "string"
                                },
                                {
                                    "type": "null"
                                }
                            ],
                            "default": None,
                            "title": "Screen Area"
                        },
                        "page_context": {
                            "anyOf": [
                                {
                                    "type": "string"
                                },
                                {
                                    "type": "null"
                                }
                            ],
                            "default": None,
                            "title": "Page Context"
                        },
                        "region_analysis": {
                            "anyOf": [
                                {
                                    "type": "object"
                                },
                                {
                                    "type": "null"
                                }
                            ],
                            "default": None,
                            "title": "Region Analysis"
                        }
                    },
                    "required": [
                        "coordinates"
                    ]
                }
            },
            {
                "name": "interpret_gesture_intent",
                "description": "解释手势意图，结合页面上下文和眼动数据进行理解",
                "parameters": {
                    "properties": {
                        "gesture_name": {
                            "title": "Gesture Name",
                            "type": "string"
                        },
                        "confidence": {
                            "title": "Confidence",
                            "type": "number"
                        },
                        "page_context": {
                            "anyOf": [
                                {
                                    "type": "string"
                                },
                                {
                                    "type": "null"
                                }
                            ],
                            "default": None,
                            "title": "Page Context"
                        },
                        "context_info": {
                            "anyOf": [
                                {
                                    "type": "object"
                                },
                                {
                                    "type": "null"
                                }
                            ],
                            "default": None,
                            "title": "Context Info"
                        },
                        "gaze_data": {
                            "anyOf": [
                                {
                                    "type": "object"
                                },
                                {
                                    "type": "null"
                                }
                            ],
                            "default": None,
                            "title": "Gaze Data"
                        }
                    },
                    "required": [
                        "gesture_name",
                        "confidence"
                    ]
                }
            },
            {
                "name": "map_gesture_to_action",
                "description": "将手势映射到应用程序操作",
                "parameters": {
                    "properties": {
                        "gesture_name": {
                            "title": "Gesture Name",
                            "type": "string"
                        },
                        "application_context": {
                            "default": "general",
                            "title": "Application Context",
                            "type": "string"
                        },
                        "intent": {
                            "anyOf": [
                                {
                                    "type": "string"
                                },
                                {
                                    "type": "null"
                                }
                            ],
                            "default": None,
                            "title": "Intent"
                        },
                        "ui_elements": {
                            "anyOf": [
                                {
                                    "items": {
                                        "type": "string"
                                    },
                                    "type": "array"
                                },
                                {
                                    "type": "null"
                                }
                            ],
                            "default": None,
                            "title": "Ui Elements"
                        }
                    },
                    "required": [
                        "gesture_name"
                    ]
                }
            },
            {
                "name": "get_ui_element_at_position",
                "description": "获取屏幕特定位置的UI元素信息",
                "parameters": {
                    "properties": {
                        "image_base64": {
                            "title": "Image Base64",
                            "type": "string"
                        },
                        "coordinates": {
                            "additionalProperties": {
                                "type": "number"
                            },
                            "title": "Coordinates",
                            "type": "object"
                        },
                        "element_type": {
                            "anyOf": [
                                {
                                    "type": "string"
                                },
                                {
                                    "type": "null"
                                }
                            ],
                            "default": None,
                            "title": "Element Type"
                        }
                    },
                    "required": [
                        "image_base64",
                        "coordinates"
                    ]
                }
            },
            {
                "name": "generate_ui_action",
                "description": "生成用户界面操作指令",
                "parameters": {
                    "properties": {
                        "intent": {
                            "title": "Intent",
                            "type": "string"
                        },
                        "ui_element": {
                            "anyOf": [
                                {
                                    "type": "object"
                                },
                                {
                                    "type": "null"
                                }
                            ],
                            "default": None,
                            "title": "Ui Element"
                        },
                        "action_type": {
                            "anyOf": [
                                {
                                    "type": "string"
                                },
                                {
                                    "type": "null"
                                }
                            ],
                            "default": None,
                            "title": "Action Type"
                        },
                        "coordinates": {
                            "anyOf": [
                                {
                                    "additionalProperties": {
                                        "type": "number"
                                    },
                                    "type": "object"
                                },
                                {
                                    "type": "null"
                                }
                            ],
                            "default": None,
                            "title": "Coordinates"
                        }
                    },
                    "required": [
                        "intent"
                    ]
                }
            }
        ]
        return tools
    
    def get_tools_json(self):
        """返回工具的JSON表示"""
        return json.dumps(self.tools)
    
    def parse_tool_calls(self, response):
        """从响应中解析工具调用"""
        tool_call_pattern = r'```tool_call\n(.*?)\n```'
        matches = re.finditer(tool_call_pattern, response, re.DOTALL)
        tool_calls = []
        
        for match in matches:
            try:
                tool_call_json = json.loads(match.group(1))
                tool_calls.append(tool_call_json)
            except json.JSONDecodeError:
                print(f"无法解析工具调用JSON: {match.group(1)}")
        
        return tool_calls


class Phi4WorkflowWithTools:
    def __init__(self, model_path="/home/lab/phi4/phi4"):
        """初始化工作流"""
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            device_map="cuda", 
            torch_dtype="auto", 
            trust_remote_code=True,
            _attn_implementation='flash_attention_2',
        ).cuda()
        self.tool_manager = ToolManager()
        self.results = {}  # 存储工作流各步骤的结果
    
    def _encode_image_to_base64(self, image):
        """将PIL图像编码为base64字符串"""
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    def _decode_base64_to_image(self, base64_string):
        """将base64字符串解码为PIL图像"""
        return Image.open(io.BytesIO(base64.b64decode(base64_string)))
    
    def _call_phi4_with_tools(self, prompt, images=None, audios=None):
        """使用Phi-4处理带工具的提示词"""
        # 包含tools参数
        prompt_with_tools = prompt + f"\n\nTools:\n{self.tool_manager.get_tools_json()}"
        
        inputs = self.processor(
            text=prompt_with_tools,
            images=images,
            return_tensors='pt'
        ).to('cuda:0')
        
        generation_config = GenerationConfig.from_pretrained("microsoft/Phi-4-multimodal-instruct")
        
        generate_ids = self.model.generate(
            **inputs,
            max_new_tokens=1024,
            generation_config=generation_config,
        )
        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
        
        response = self.processor.batch_decode(
            generate_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        return response
    
    def _call_phi4_without_tools(self, prompt, images=None, audios=None):
        """使用Phi-4处理不带工具的提示词（标准推理）"""
        inputs = self.processor(
            text=prompt,
            images=images,
            return_tensors='pt'
        ).to('cuda:0')
        
        generation_config = GenerationConfig.from_pretrained("microsoft/Phi-4-multimodal-instruct")
        
        generate_ids = self.model.generate(
            **inputs,
            max_new_tokens=512,
            generation_config=generation_config,
        )
        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
        
        response = self.processor.batch_decode(
            generate_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        return response
    
    def step1_raw_input_processing(self, gesture_name="pinch", gaze_data={"x": 0.4, "y": 0.5, "r": 0.1}, screenshot=None):
        """步骤1: 处理原始输入，准备场景分析"""
        
        # 格式化输入
        formatted_input = {
            "gesture": {
                "name": gesture_name,
                "confidence": 0.95  # 假设置信度
            },
            "gaze": {
                "x": gaze_data["x"],
                "y": gaze_data["y"],
                "radius": gaze_data.get("r", 0.1)
            },
            "screenshot": screenshot
        }
        
        # 存储结果
        self.results["step1_input"] = formatted_input
        
        # 编码图像为base64（如果有）
        if screenshot:
            formatted_input["screenshot_base64"] = self._encode_image_to_base64(screenshot)
        
        return formatted_input
    
    def step2_page_function_extraction(self, input_data):
        """步骤2: 页面功能提取，理解当前UI的功能和元素"""
        
        screenshot = input_data.get("screenshot")
        if not screenshot:
            return {"error": "没有提供截图，无法分析页面"}
        
        # 构建提示词
        prompt = """<|user|><|image_1|>
        你是一个图像UI分析助手，请分析当前屏幕上显示的界面。
        1. 首先，用一到两句话概括这个界面的主要功能和类型
        2. 列出界面上的主要元素和它们的功能
        3. 确定这是什么应用或网站
        <|end|>
        <|assistant|>"""
        
        # 调用Phi4进行分析
        response = self._call_phi4_without_tools(prompt, images=screenshot)
        
        # 解析结果
        analysis = {
            "raw_analysis": response,
            "page_type": None,  # 通过后处理提取
            "ui_elements": [],  # 通过后处理提取
            "application": None # 通过后处理提取
        }
        
        # 这里可以添加后处理逻辑，从response提取结构化信息
        # ...
        
        # 存储结果
        self.results["step2_page_analysis"] = analysis
        
        return analysis
    
    def step3_intent_recognition(self, input_data, scene_analysis):
        """步骤3: 意图识别，理解用户想要做什么"""
        
        gesture = input_data.get("gesture", {})
        gaze = input_data.get("gaze", {})
        screenshot = input_data.get("screenshot")
        
        if not screenshot:
            return {"error": "没有提供截图，无法分析意图"}
        
        # 构建提示词，使用工具调用进行更精确的分析
        prompt = f"""<|user|><|image_1|>
        你是一个混合模态用户意图分析师。请用以下信息理解用户意图：

        用户当前界面信息：
        {scene_analysis.get("raw_analysis", "未提供界面分析")}

        用户手势：{gesture.get("name", "未知")}
        手势置信度：{gesture.get("confidence", 0.0)}

        用户视线位置：
        - X坐标：{gaze.get("x", 0.0)}（范围0-1，0是左边缘，1是右边缘）
        - Y坐标：{gaze.get("y", 0.0)}（范围0-1，0是上边缘，1是下边缘）

        请通过调用合适的工具，分析用户的意图和可能想要执行的操作。

        首先，根据用户的视线位置分析其正在关注的区域。
        然后，结合用户的手势和上下文，推断用户的完整意图。
        最后，生成一个具体的UI操作建议。
        <|end|>
        <|assistant|>"""
        
        # 调用Phi4进行分析（带工具）
        response = self._call_phi4_with_tools(prompt, images=screenshot)
        
        # 解析工具调用
        tool_calls = self.tool_manager.parse_tool_calls(response)
        
        # 处理结果
        intent_analysis = {
            "raw_response": response,
            "tool_calls": tool_calls,
            "interpreted_intent": None,  # 通过后处理提取
            "suggested_action": None,    # 通过后处理提取
            "confidence": None           # 通过后处理提取
        }
        
        # 这里可以添加从工具调用和响应中提取结构化信息的后处理逻辑
        # ...
        
        # 从响应中提取主要意图（简单示例）
        intent_match = re.search(r"用户意图:(.*?)(?:\n|$)", response, re.MULTILINE | re.IGNORECASE)
        if intent_match:
            intent_analysis["interpreted_intent"] = intent_match.group(1).strip()
        
        # 从响应中提取建议动作（简单示例）
        action_match = re.search(r"建议动作:(.*?)(?:\n|$)", response, re.MULTILINE | re.IGNORECASE)
        if action_match:
            intent_analysis["suggested_action"] = action_match.group(1).strip()
        
        # 存储结果
        self.results["step3_intent_analysis"] = intent_analysis
        
        return intent_analysis
    
    def visualize_gaze_on_screenshot(self, screenshot, gaze_data):
        """在截图上可视化眼动点"""
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(screenshot)
        
        # 获取图像尺寸
        width, height = screenshot.size
        
        # 转换相对坐标为像素坐标
        x_pixel = gaze_data["x"] * width
        y_pixel = gaze_data["y"] * height
        
        # 计算圆的半径（相对值转为像素值）
        radius_pixel = gaze_data.get("radius", 0.05) * min(width, height)
        
        # 添加表示视线的圆圈
        gaze_circle = Circle((x_pixel, y_pixel), radius_pixel, 
                             color='red', alpha=0.5, fill=True)
        ax.add_patch(gaze_circle)
        
        # 添加视线坐标点
        ax.plot(x_pixel, y_pixel, 'ro', markersize=5)
        
        # 添加坐标标签
        ax.text(x_pixel + 10, y_pixel + 10, 
                f"({gaze_data['x']:.2f}, {gaze_data['y']:.2f})", 
                color='white', fontsize=12, 
                bbox=dict(facecolor='black', alpha=0.7))
        
        # 关闭坐标轴
        ax.axis('off')
        
        return fig
    
    def visualize_results(self):
        """可视化工作流结果"""
        if not self.results:
            print("没有可视化的结果")
            return None
        
        # 获取输入数据
        input_data = self.results.get("step1_input", {})
        screenshot = input_data.get("screenshot")
        gaze_data = input_data.get("gaze", {})
        gesture = input_data.get("gesture", {})
        
        # 获取分析结果
        page_analysis = self.results.get("step2_page_analysis", {})
        intent_analysis = self.results.get("step3_intent_analysis", {})
        
        if not screenshot:
            print("没有截图，无法可视化")
            return None
        
        # 创建可视化
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(screenshot)
        
        # 获取图像尺寸
        width, height = screenshot.size
        
        # 转换相对坐标为像素坐标
        x_pixel = gaze_data.get("x", 0.5) * width
        y_pixel = gaze_data.get("y", 0.5) * height
        
        # 计算圆的半径（相对值转为像素值）
        radius_pixel = gaze_data.get("radius", 0.05) * min(width, height)
        
        # 添加表示视线的圆圈
        gaze_circle = Circle((x_pixel, y_pixel), radius_pixel, 
                             color='red', alpha=0.3, fill=True)
        ax.add_patch(gaze_circle)
        
        # 添加视线坐标点
        ax.plot(x_pixel, y_pixel, 'ro', markersize=8)
        
        # 在图像上添加分析结果文本
        txt = f"手势: {gesture.get('name', '未知')}\n"
        txt += f"视线位置: ({gaze_data.get('x', 0):.2f}, {gaze_data.get('y', 0):.2f})\n\n"
        
        if intent_analysis.get("interpreted_intent"):
            txt += f"意图: {intent_analysis['interpreted_intent']}\n"
        
        if intent_analysis.get("suggested_action"):
            txt += f"建议操作: {intent_analysis['suggested_action']}\n"
        
        # 添加文本框
        props = dict(boxstyle='round', facecolor='black', alpha=0.7)
        ax.text(0.02, 0.98, txt, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', color='white', bbox=props)
        
        # 关闭坐标轴
        ax.axis('off')
        plt.tight_layout()
        
        return fig
    
    def run_workflow(self, gesture_name="pinch", gaze_data={"x": 0.4, "y": 0.5, "r": 0.1}, screenshot=None, visualize=True):
        """运行完整工作流"""
        print("🚀 开始多模态用户意图分析工作流")
        
        # 步骤1: 处理原始输入
        print("\n🔹 步骤1: 处理输入数据")
        input_data = self.step1_raw_input_processing(gesture_name, gaze_data, screenshot)
        
        # 步骤2: 页面功能提取
        print("\n🔹 步骤2: 分析页面功能")
        scene_analysis = self.step2_page_function_extraction(input_data)
        
        # 步骤3: 意图识别
        print("\n🔹 步骤3: 识别用户意图")
        intent_analysis = self.step3_intent_recognition(input_data, scene_analysis)
        
        # 可视化结果
        if visualize and screenshot:
            print("\n🔹 生成可视化结果")
            fig = self.visualize_results()
            if fig:
                plt.show()
        
        # 输出结果摘要
        print("\n✅ 工作流完成！")
        print(f"页面分析: {scene_analysis.get('raw_analysis', '')[:100]}...")
        print(f"推断意图: {intent_analysis.get('interpreted_intent', '未能识别意图')}")
        if intent_analysis.get("suggested_action"):
            print(f"建议操作: {intent_analysis['suggested_action']}")
        
        return {
            "page_analysis": scene_analysis,
            "intent_analysis": intent_analysis
        }


# 测试代码
if __name__ == "__main__":
    import os
    from PIL import Image
    
    # 初始化工作流
    workflow = Phi4WorkflowWithTools()
    
    # 测试图像路径
    test_image_path = "test_video_player.png"
    
    # 如果测试图像不存在
    if not os.path.exists(test_image_path):
        print(f"测试图像 {test_image_path} 不存在，创建示例图像")
        # 创建一个简单的测试图像
        img = Image.new('RGB', (800, 600), color=(73, 109, 137))
        img.save(test_image_path)
    
    # 加载测试图像
    screenshot = Image.open(test_image_path)
    
    # 测试手势和视线数据
    gesture_name = "pinch"  # 捏合手势
    gaze_data = {"x": 0.6, "y": 0.4, "r": 0.15}  # 视线位置和半径
    
    # 运行工作流
    result = workflow.run_workflow(
        gesture_name=gesture_name,
        gaze_data=gaze_data,
        screenshot=screenshot,
        visualize=True
    )
    
    print("\n结果:", result)