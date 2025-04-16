import requests
import torch
import os
import io
from PIL import Image
import soundfile as sf
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from urllib.request import urlopen
import pandas as pd
import time

class PhiUserIntentWorkflow:
    def __init__(self, model_path="/home/lab/phi4/phi4", verbose=True):
        """
        初始化多模态LLM推理工作流
        
        Args:
            model_path: Phi-4模型路径
            verbose: 是否显示详细调试信息
        """
        print("🔄 初始化模型...")
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            device_map="cuda", 
            torch_dtype="auto", 
            trust_remote_code=True,
            _attn_implementation='flash_attention_2',
        ).cuda()
        self.generation_config = GenerationConfig.from_pretrained(model_path)
        
        # 定义提示词结构
        self.user_prompt = '<|user|>'
        self.assistant_prompt = '<|assistant|>'
        self.prompt_suffix = '<|end|>'
        self.verbose = verbose
        print("✅ 模型加载完成")
    
    def log(self, message, level="INFO"):
        """调试输出函数"""
        if self.verbose:
            if level == "INFO":
                print(f"ℹ️ {message}")
            elif level == "STEP":
                print(f"\n🔶 {message}")
            elif level == "SUCCESS":
                print(f"✅ {message}")
            elif level == "ERROR":
                print(f"❌ {message}")
            elif level == "TIME":
                print(f"⏱️ {message}")
    
    def call_model(self, prompt, image=None, max_new_tokens=500):
        """调用phi4模型进行推理"""
        self.log(f"提示词: {prompt[:50]}...", "INFO")
        
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
        )
        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
        response = self.processor.batch_decode(
            generate_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        end_time = time.time()
        
        response_time = end_time - start_time
        self.log(f"响应用时: {response_time:.2f}秒", "TIME")
        self.log(f"模型响应: {response}", "INFO")
        
        return response, response_time
    
    def crop_image_at_gaze(self, image, coordinates, radius, save_path=None):
        """根据眼动坐标和半径裁剪图像
        
        Args:
            image: PIL图像对象
            coordinates: 眼动坐标字典,包含x和y
            radius: 裁剪半径
            save_path: 可选,保存裁剪图像的路径
        """
        width, height = image.size
        x_pixel = int(coordinates["x"] * width)
        y_pixel = int(coordinates["y"] * height)
        r_pixel = int(radius * min(width, height))
        
        left = max(0, x_pixel - r_pixel)
        top = max(0, y_pixel - r_pixel)
        right = min(width, x_pixel + r_pixel)
        bottom = min(height, y_pixel + r_pixel)
        
        self.log(f"裁剪坐标: ({left}, {top}, {right}, {bottom})", "INFO")
        cropped_image = image.crop((left, top, right, bottom))
        
        # 如果提供了保存路径,则保存裁剪后的图像
        if save_path:
            try:
                # 确保保存目录存在
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                cropped_image.save(save_path)
                self.log(f"裁剪图像已保存至: {save_path}", "INFO")
            except Exception as e:
                self.log(f"保存裁剪图像失败: {str(e)}", "ERROR")
                
        return cropped_image

    def infer_intent(self, image_path, gesture, gaze_data=None):
        """
        完整的意图推理工作流
        
        Args:
            image_path: UI图像路径
            gesture: 用户手势 (如 'pinch', 'thumb up')
            gaze_data: 眼动数据列表, 每个包含坐标和半径
                格式: [{'coordinates': {'x': 0.5, 'y': 0.5}, 'radius': 0.1}, ...]
        
        Returns:
            包含分析结果的字典
        """
        self.log(f"开始处理图像: {os.path.basename(image_path)}", "STEP")
        
        try:
            # 步骤1: 加载图像
            full_image = Image.open(image_path)
            self.log(f"图像尺寸: {full_image.size[0]}x{full_image.size[1]}", "INFO")
        except Exception as e:
            self.log(f"加载图像失败: {str(e)}", "ERROR")
            return {"error": str(e)}
        
        # 步骤2: 分析整体UI界面
        self.log("分析整体UI界面", "STEP")
        ui_prompt = f'''{self.user_prompt}<|image_1|>
分析界面:
一句话描述当前UI页面的功能和主要组件。
{self.prompt_suffix}{self.assistant_prompt}'''
        
        overview, overview_time = self.call_model(ui_prompt, full_image, 100)
        self.log(f"UI概览: {overview}", "SUCCESS")
        
        # 步骤3: 裁剪眼动关注区域的图像
        cropped_images = []
        if gaze_data and len(gaze_data) > 0:
            self.log(f"裁剪{len(gaze_data)}个眼动关注区域", "STEP")
            
            for i, gaze in enumerate(gaze_data):
                self.log(f"裁剪眼动点 #{i+1}: 坐标({gaze['coordinates']['x']:.2f}, {gaze['coordinates']['y']:.2f})", "INFO")
                
                # 裁剪眼动注视区域
                cropped_image = self.crop_image_at_gaze(
                    full_image, 
                    gaze['coordinates'], 
                    gaze['radius']
                )
                
                cropped_images.append({
                    'gaze_id': i+1,
                    'coordinates': gaze['coordinates'],
                    'cropped_image': cropped_image
                })
                
                self.log(f"眼动区域 #{i+1} 裁剪完成", "SUCCESS")
        else:
            self.log("未提供眼动数据，跳过图像裁剪步骤", "INFO")
        
        # 步骤4: 结合UI和手势推断意图 (使用简化的提示词)
        self.log(f"使用简化提示词推断意图", "STEP")
        
        intent_prompt = f'''
        <|user|><|image_1|>
        UI界面: {overview}
        用户手势: {gesture}
        根据当前UI页面和用户手势，推测用户可能想要执行的操作。
        <|end|><|assistant|>
        '''
        
        intent, intent_time = self.call_model(intent_prompt, full_image, 150)
        self.log(f"推断意图: {intent}", "SUCCESS")
        
        # 返回完整分析结果
        result = {
            'image_path': image_path,
            'ui_overview': overview,
            'gesture': gesture,
            'cropped_images': cropped_images,  # 只包含裁剪后的图像，不做分析
            'inferred_intent': intent,
            'total_response_time': overview_time + intent_time
        }
        
        self.log(f"完成分析! 总用时: {result['total_response_time']:.2f}秒", "STEP")
        return result


# 使用示例
if __name__ == "__main__":
    # 初始化工作流
    workflow = PhiUserIntentWorkflow()
    
    # 测试单张图像
    print("\n========== 单图像手势处理测试 ==========")
    # 定义测试数据
    image_path = 'dataset/dataset/rico/16.jpg'
    gesture = 'thumb up'
    
    # 眼动数据示例 (规范化坐标)
    gaze_data = [
        {
            'coordinates': {'x': 0.5, 'y': 0.3},  # 屏幕中上部
            'radius': 0.15                        # 关注半径
        },
        {
            'coordinates': {'x': 0.2, 'y': 0.5},  # 屏幕左侧
            'radius': 0.15
        }
    ]
    
    # 运行意图分析
    result = workflow.infer_intent(image_path, gesture, gaze_data)
    
    # 输出完整结果
    print("\n========== 意图分析结果 ==========")
    print(f"UI概览: {result['ui_overview']}")
    print(f"手势: {result['gesture']}")
    print(f"裁剪区域数量: {len(result['cropped_images'])}")
    print(f"推断意图: {result['inferred_intent']}")
    print(f"总响应时间: {result['total_response_time']:.2f}秒")