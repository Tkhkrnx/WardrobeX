# app.py
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import torch
from transformers import (
    ViTImageProcessor,
    ViTModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from pymilvus import connections, Collection
import numpy as np
import io
import re
import cv2
import os
from ultralytics import YOLO
from typing import List, Dict
from peft import PeftModel
from langchain_openai import ChatOpenAI
import logging
from dotenv import load_dotenv
import gc
import traceback
import asyncio
import time

load_dotenv(override=True)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# 创建安全的日志记录函数
def safe_log_info(message, *args, **kwargs):
    try:
        logger.info(message, *args, **kwargs)
    except UnicodeEncodeError:
        # 处理可能包含特殊字符的消息
        if isinstance(message, str):
            safe_message = message.encode('utf-8', errors='replace').decode('utf-8')
            logger.info(safe_message, *args, **kwargs)
        else:
            logger.info(message, *args, **kwargs)


app = FastAPI()

# 全局变量用于存储模型
extractor = None
vit = None
coll = None
yolo_model = None
# 文案生成模型相关
text_tokenizer = None
text_model = None
# 强模型API配置
LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_BASE_URL = "https://www.chataiapi.com/v1"
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "claude-3-5-sonnet-20241022")

# 量化控制：环境变量 QUANT_BITS = "4" 或 "8"
QUANT_BITS = os.getenv("QUANT_BITS", "4").strip()

class_names = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时执行
    load_models()
    yield
    # 关闭时执行（如果需要）
    # 在应用退出时清理（如果有必要）
    try:
        safe_log_info("Shutting down - clearing models from memory.")
        global vit, text_model
        del vit
        del text_model
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
    except Exception as e:
        safe_log_info(f"Error during shutdown cleanup: {e}")


app = FastAPI(lifespan=lifespan)


def _get_quant_kwargs():
    quant_kwargs = {}
    if QUANT_BITS == "4":
        try:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16 if hasattr(torch, "bfloat16") else torch.float16
            )
            quant_kwargs.update({
                "quantization_config": bnb_config,
                "device_map": "auto",
                "low_cpu_mem_usage": True
            })
            safe_log_info("Configured 4-bit (NF4) quantization parameters.")
            return quant_kwargs
        except Exception as e:
            safe_log_info(f"Failed to create 4-bit BitsAndBytesConfig: {e}")
            safe_log_info("Falling back to 8-bit configuration.")
    try:
        quant_kwargs.update({
            "load_in_8bit": True,
            "device_map": "auto",
            "low_cpu_mem_usage": True
        })
        safe_log_info("Configured 8-bit quantization parameters.")
    except Exception as e:
        safe_log_info(f"Failed to configure 8-bit params: {e}")
    return quant_kwargs


def load_models():
    """
    加载所有模型（尽量使用量化加载以减小内存）
    """
    global extractor, vit, coll, yolo_model, text_tokenizer, text_model

    safe_log_info("Loading models...")

    # 加载 ViT 特征提取器
    try:
        extractor = ViTImageProcessor.from_pretrained('/data/google/vit-base-patch16-224')
        safe_log_info("ViTImageProcessor loaded.")
    except Exception as e:
        safe_log_info(f"Failed to load ViTImageProcessor: {e}")
        safe_log_info(traceback.format_exc())

    # 加载基础 ViT 模型，优先 float16 以节省显存，fallback 到 float32
    try:
        safe_log_info("Loading ViT model with float16 precision...")
        vit = ViTModel.from_pretrained('/data/google/vit-base-patch16-224',
                                       torch_dtype=torch.float16,
                                       low_cpu_mem_usage=True)
        safe_log_info("ViT model loaded with float16 precision.")
    except Exception as e:
        safe_log_info(f"Failed to load ViT model with float16: {e}")
        safe_log_info("Trying fallback to float32 load...")
        try:
            vit = ViTModel.from_pretrained('/data/google/vit-base-patch16-224',
                                           torch_dtype=torch.float32,
                                           low_cpu_mem_usage=True)
            safe_log_info("ViT model loaded with float32 precision.")
        except Exception as e2:
            safe_log_info(f"Failed to load ViT model entirely: {e2}")
            safe_log_info(traceback.format_exc())
            vit = None

    # 加载使用 LoRA 微调的最佳模型权重（如存在）
    try:
        if vit is not None and os.path.exists('outputs/vit_lora_best_peft') and os.path.isdir('outputs/vit_lora_best_peft'):
            config_file = os.path.join('outputs/vit_lora_best_peft', 'adapter_config.json')
            if os.path.exists(config_file):
                try:
                    vit = PeftModel.from_pretrained(vit, 'outputs/vit_lora_best_peft')
                    try:
                        vit = vit.merge_and_unload()
                        safe_log_info("LoRA model merged and loaded into ViT.")
                    except Exception:
                        safe_log_info("LoRA loaded but merge_and_unload not supported/failed; using PeftModel wrapper.")
                except Exception as e:
                    safe_log_info(f"Failed to apply LoRA to ViT: {e}")
            else:
                safe_log_info("LoRA config file not found, using base ViT model")
        else:
            safe_log_info("LoRA model directory not found, using base ViT model")
    except Exception as e:
        safe_log_info(f"Failed to load LoRA model: {e}")

    try:
        if vit is not None:
            vit.eval()
    except Exception as e:
        safe_log_info(f"Failed to set ViT eval: {e}")

    # 加载本地微调的文案生成模型 (Prompt Tuning) 使用量化配置
    try:
        safe_log_info("Loading fine-tuned text generation model (tokenizer first)...")
        base_model_name = '/data/Qwen3-0.6B'
        text_tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
        if text_tokenizer.pad_token is None:
            text_tokenizer.pad_token = text_tokenizer.eos_token

        safe_log_info("Preparing to load base text model with quantization (if available)...")

        # ** 这里改为调用 _get_quant_kwargs() 函数，确保量化参数获取统一 **
        quant_kwargs = _get_quant_kwargs()

        text_model = None
        load_success = False

        # 尝试使用量化加载（4-bit 或 8-bit）
        try:
            text_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                trust_remote_code=True,
                **quant_kwargs
            )
            load_success = True
            safe_log_info("Base text model loaded with quantization parameters.")
        except Exception as e:
            safe_log_info(f"Quantized load failed for text model: {e}")
            safe_log_info("Trying fallback to float16 low_cpu_mem_usage load...")
            try:
                text_model = AutoModelForCausalLM.from_pretrained(
                    base_model_name,
                    trust_remote_code=True,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True
                )
                load_success = True
                safe_log_info("Base text model loaded with float16 fallback.")
            except Exception as e2:
                safe_log_info(f"Fallback load failed for text model: {e2}")
                safe_log_info(traceback.format_exc())

        if not load_success:
            safe_log_info("Text model could not be loaded in any optimized mode. Setting text_model to None.")
            text_model = None
        else:
            # 检查是否存在训练后最好的 PEFT 权重
            best_model_path = 'outputs/qwen3_0.6b_prompt_best'
            final_model_path = 'outputs/qwen3_0.6b_prompt_final'

            if os.path.exists(best_model_path) and os.path.isdir(best_model_path):
                config_file = os.path.join(best_model_path, 'adapter_config.json')
                if os.path.exists(config_file):
                    try:
                        text_model = PeftModel.from_pretrained(text_model, best_model_path)
                        text_model.eval()
                        safe_log_info("Best Prompt Tuning model loaded successfully from qwen3_0.6b_prompt_best!")
                    except Exception as e:
                        safe_log_info(f"Failed to load PEFT best model: {e}")
                        safe_log_info(traceback.format_exc())
                else:
                    safe_log_info("Best Prompt Tuning config file not found, checking final model...")
                    if os.path.exists(final_model_path) and os.path.isdir(final_model_path):
                        config_file2 = os.path.join(final_model_path, 'adapter_config.json')
                        if os.path.exists(config_file2):
                            try:
                                text_model = PeftModel.from_pretrained(text_model, final_model_path)
                                text_model.eval()
                                safe_log_info("Final Prompt Tuning model loaded successfully from qwen3_0.6b_prompt_final!")
                            except Exception as e:
                                safe_log_info(f"Failed to load PEFT final model: {e}")
                                safe_log_info(traceback.format_exc())
                        else:
                            safe_log_info("Final Prompt Tuning config file not found, using base model")
                    else:
                        safe_log_info("Prompt Tuning model directory not found, using base model")
            else:
                safe_log_info("Best Prompt Tuning model directory not found, checking final model...")
                if os.path.exists(final_model_path) and os.path.isdir(final_model_path):
                    config_file2 = os.path.join(final_model_path, 'adapter_config.json')
                    if os.path.exists(config_file2):
                        try:
                            text_model = PeftModel.from_pretrained(text_model, final_model_path)
                            text_model.eval()
                            safe_log_info("Final Prompt Tuning model loaded successfully from qwen3_0.6b_prompt_final!")
                        except Exception as e:
                            safe_log_info(f"Failed to load PEFT final model: {e}")
                            safe_log_info(traceback.format_exc())
                    else:
                        safe_log_info("Final Prompt Tuning config file not found, using base model")
                else:
                    safe_log_info("Prompt Tuning model directory not found, using base model")

            try:
                if text_model is not None:
                    text_model.eval()
            except Exception as e:
                safe_log_info(f"Failed to set text_model eval: {e}")

        safe_log_info("Fine-tuned text generation model loading finished.")
    except Exception as e:
        safe_log_info(f"Failed to load fine-tuned text model: {e}")
        safe_log_info(traceback.format_exc())
        text_tokenizer = None
        text_model = None


    # 连接 Milvus
    try:
        connections.connect(host='150.158.55.76', port='19530')
        coll = Collection('fashion_features')
        coll.load()
        safe_log_info("Milvus connected successfully!")
    except Exception as e:
        safe_log_info(f"Failed to connect to Milvus: {e}")
        coll = None

    # 加载 YOLO 模型用于检测衣物类型（减少日志输出）
    try:
        yolo_model = YOLO('modanet_clothing_model.pt')
        # 减少YOLO模型的日志输出
        if hasattr(yolo_model, 'names') and yolo_model.names:
            class_names.update(yolo_model.names)
        else:
            class_names.update({
                0: 'top', 1: 'bottom', 2: 'dress', 3: 'outer', 4: 'pants',
                5: 'skirt', 6: 'headwear', 7: 'eyewear', 8: 'bag', 9: 'shoes',
                10: 'belt', 11: 'socks', 12: 'tights', 13: 'gloves', 14: 'scarf'
            })
        safe_log_info("YOLO model loaded successfully!")
    except Exception as e:
        safe_log_info(f"Failed to load YOLO model: {e}")
        yolo_model = None
        class_names.update({
            0: 'top', 1: 'bottom', 2: 'dress', 3: 'outer', 4: 'pants',
            5: 'skirt', 6: 'headwear', 7: 'eyewear', 8: 'bag', 9: 'shoes',
            10: 'belt', 11: 'socks', 12: 'tights', 13: 'gloves', 14: 'scarf'
        })

    safe_log_info("All models loaded (or attempted). Current memory state:")
    try:
        safe_log_info(f" - ViT loaded: {vit is not None}")
        safe_log_info(f" - Text model loaded: {text_model is not None}")
        safe_log_info(f" - Tokenizer loaded: {text_tokenizer is not None}")
        safe_log_info(f" - Milvus collection: {coll is not None}")
        safe_log_info(f" - YOLO loaded: {yolo_model is not None}")
    except Exception:
        pass


def generate_outline_local(prompt: str, max_tokens: int = 100) -> str:
    """
    使用本地微调模型生成搭配要点
    """
    global text_tokenizer, text_model
    if text_tokenizer is None or text_model is None:
        return "文案生成模型未加载，请检查模型路径。"

    try:
        # 对提示进行预处理，确保格式正确
        if not prompt.strip():
            return "输入提示为空，无法生成文案。"

        # 清理提示词，确保没有多余的空白字符
        prompt = prompt.strip()

        # 使用float16精度并避免使用GPU
        inputs = text_tokenizer(prompt, return_tensors='pt', padding=True, truncation=True, max_length=512)

        with torch.no_grad():
            outputs = text_model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                pad_token_id=text_tokenizer.pad_token_id,
                eos_token_id=text_tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                length_penalty=1.0,
                no_repeat_ngram_size=3
                # 移除了 early_stopping 参数，因为它不被当前版本支持
            )
            text = text_tokenizer.decode(outputs[0], skip_special_tokens=True)
            # 移除输入部分，只返回生成的部分
            generated_text = text[len(prompt):].strip()

        # 检查生成的文本是否有效
        if not generated_text or len(generated_text) < 5:
            return "无法生成有效的搭配要点，请稍后重试。"

        return generated_text
    except Exception as e:
        safe_log_info(f"Error generating outline with local model: {e}")
        safe_log_info(traceback.format_exc())
        # 返回一个默认的搭配建议
        return "建议选择简约风格的鞋子来平衡整体造型，搭配一款时尚的包袋增加亮点，根据季节选择合适的外套进行叠穿。"


def call_qwen_api(prompt: str) -> str:
    """
    调用API生成高质量文案
    """
    if not LLM_API_KEY:
        return "API密钥未配置，无法调用大模型生成文案。"

    try:
        llm = ChatOpenAI(
            temperature=0.7,
            model=LLM_MODEL_NAME,
            api_key=LLM_API_KEY,
            base_url=LLM_BASE_URL,
            request_timeout=30  # 添加30秒超时
        )

        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        safe_log_info(f"Error calling Qwen API: {e}")
        safe_log_info(traceback.format_exc())
        return "调用大模型API时出错，请稍后重试。"


def detect_clothing_category(image: Image.Image) -> str:
    """
    使用YOLO模型检测衣物类别
    """
    global class_names, yolo_model

    # 如果YOLO模型未加载，直接返回unknown
    if yolo_model is None:
        safe_log_info("YOLO model not loaded")
        return 'unknown'

    try:
        # 将PIL图像转换为OpenCV格式（在内存中）
        # 先转换为numpy数组
        img_array = np.array(image)
        # RGB转BGR（PIL使用RGB，OpenCV使用BGR）
        cv_image = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        safe_log_info(f"Input image size: {img_array.shape}")

        # 使用YOLO模型进行检测，尝试不同的置信度阈值
        results = yolo_model.predict(source=cv_image, conf=0.1)  # 进一步降低置信度阈值

        safe_log_info(f"Detection results: {results}")

        # 获取检测到的类别
        for r in results:
            if hasattr(r, 'boxes') and r.boxes is not None and len(r.boxes) > 0:
                # 打印检测到的所有框的信息
                confs = r.boxes.conf.cpu().numpy()
                classes = r.boxes.cls.cpu().numpy()

                safe_log_info(f"Detected boxes - confidences: {confs}, classes: {classes}")

                if len(confs) > 0:
                    # 按置信度排序，取最高的
                    sorted_indices = np.argsort(confs)[::-1]  # 降序排列
                    for idx in sorted_indices:
                        conf = confs[idx]
                        class_id = int(classes[idx])

                        safe_log_info(f"Checking detection - confidence: {conf}, class_id: {class_id}")

                        # 使用全局class_names映射获取类别名称
                        detected_class_name = class_names.get(class_id, 'unknown')
                        safe_log_info(f"Detected class name: {detected_class_name}")

                        # 只要置信度大于一定阈值就返回
                        if conf > 0.05:  # 非常低的阈值
                            return detected_class_name

    except Exception as e:
        safe_log_info(f"Error in detect_clothing_category: {e}")
        safe_log_info(traceback.format_exc())

    safe_log_info("No valid detection found, returning 'unknown'")
    return 'unknown'

def get_complementary_categories(input_category: str) -> List[str]:
    complementary_map = {
        'top': ['bottom', 'pants', 'skirt', 'shoes', 'bag'],
        'bottom': ['top', 'shoes', 'bag'],
        'pants': ['top', 'shoes', 'bag'],
        'skirt': ['top', 'shoes', 'bag'],
        'dress': ['shoes', 'bag', 'belt', 'outer'],  # 为dress增加更多搭配选项
        'outer': ['top', 'bottom', 'pants', 'skirt', 'shoes', 'bag'],
        'shoes': ['top', 'bottom', 'pants', 'skirt', 'dress'],
        'bag': ['top', 'bottom', 'pants', 'skirt', 'dress', 'shoes'],
        'headwear': ['top', 'dress'],
        'belt': ['pants', 'skirt', 'dress'],
        'unknown': ['top', 'bottom', 'pants', 'skirt', 'dress', 'shoes', 'bag', 'outer']
    }
    return complementary_map.get(input_category, ['top', 'bottom', 'shoes', 'bag'])


def search_items_by_category(category: str, input_embedding: np.ndarray, limit: int = 20) -> List[Dict]:
    if coll is None:
        return []
    try:
        # 确保向量类型为float32以匹配Milvus中的VECTOR_FLOAT类型
        if input_embedding.dtype != np.float32:
            input_embedding = input_embedding.astype(np.float32)

        # 添加超时和错误处理
        search_results = coll.search(
            input_embedding,
            'embedding',
            param={"metric_type": "IP", "params": {"nprobe": 128}},  # 进一步降低nprobe
            limit=limit,
            expr=f"category == '{category}'",
            output_fields=['image_name', 'color', 'shape', 'material'],
            timeout=10.0  # 添加10秒超时
        )
        items = []
        for result in search_results[0]:
            try:
                items.append({
                    "id": result.id,
                    "image_name": result.entity.image_name,
                    "category": category,
                    "color": getattr(result.entity, 'color', 'unknown'),
                    "shape": getattr(result.entity, 'shape', 'unknown'),
                    "material": getattr(result.entity, 'material', 'unknown'),
                    "distance": result.distance
                })
            except Exception as e:
                safe_log_info(f"Error processing search result: {e}")
                continue
        safe_log_info(f"Found {len(items)} items for category {category}")
        return items
    except Exception as e:
        safe_log_info(f"Error searching items by category {category}: {e}")
        safe_log_info(traceback.format_exc())
        return []


def get_complementary_items(input_category: str, input_embedding: np.ndarray,
                            input_color: str = 'unknown', input_shape: str = 'unknown',
                            input_material: str = 'unknown') -> Dict[str, Dict]:
    recommendations = {}
    complementary_categories = get_complementary_categories(input_category)
    safe_log_info(f"Finding complementary items for {input_category}, looking for: {complementary_categories}")

    for category in complementary_categories:
        # 添加超时保护和错误处理
        try:
            items = search_items_by_category(category, input_embedding, limit=30)  # 降低limit
            if not items:
                safe_log_info(f"No items found for category {category}")
                continue

            safe_log_info(f"Found {len(items)} items for {category}, calculating scores...")

            for item in items:
                score = item['distance']

                # 处理颜色匹配
                if input_color != 'unknown' and item['color'] != 'unknown':
                    # 将逗号分隔的颜色字符串转换为列表
                    input_colors = input_color.split(',')
                    item_colors = item['color'].split(',')

                    # 计算颜色匹配度
                    color_match = False
                    for in_color in input_colors:
                        for item_color in item_colors:
                            if in_color == item_color:
                                score += 0.15
                                color_match = True
                                break
                            elif in_color in ['black', 'white', 'gray', 'silver', 'grey'] and item_color in ['black',
                                                                                                             'white',
                                                                                                             'gray',
                                                                                                             'silver',
                                                                                                             'grey']:
                                score += 0.1
                                color_match = True
                                break
                            elif in_color in ['red', 'pink', 'burgundy', 'maroon'] and item_color in ['red', 'pink',
                                                                                                      'burgundy',
                                                                                                      'maroon']:
                                score += 0.1
                                color_match = True
                                break
                            elif in_color in ['blue', 'navy', 'indigo', 'teal'] and item_color in ['blue', 'navy',
                                                                                                   'indigo',
                                                                                                   'teal']:
                                score += 0.1
                                color_match = True
                                break
                            elif in_color in ['green', 'olive', 'lime'] and item_color in ['green', 'olive', 'lime']:
                                score += 0.1
                                color_match = True
                                break
                            elif in_color in ['brown', 'beige', 'tan', 'camel'] and item_color in ['brown', 'beige',
                                                                                                   'tan',
                                                                                                   'camel']:
                                score += 0.1
                                color_match = True
                                break
                            elif in_color in ['yellow', 'orange', 'coral'] and item_color in ['yellow', 'orange',
                                                                                              'coral']:
                                score += 0.1
                                color_match = True
                                break
                            elif in_color in ['purple', 'violet', 'lavender'] and item_color in ['purple', 'violet',
                                                                                                 'lavender']:
                                score += 0.1
                                color_match = True
                                break

                    # 如果没有精确匹配但都是中性色，给予小奖励
                    if not color_match and any(
                            c in ['black', 'white', 'gray', 'silver', 'grey'] for c in input_colors) and \
                            any(c in ['black', 'white', 'gray', 'silver', 'grey'] for c in item_colors):
                        score += 0.05

                # 处理款式匹配
                if input_shape != 'unknown' and item['shape'] != 'unknown':
                    # 将逗号分隔的款式字符串转换为列表
                    input_shapes = input_shape.split(',')
                    item_shapes = item['shape'].split(',')

                    # 计算款式匹配度
                    for in_shape in input_shapes:
                        for item_shape in item_shapes:
                            if in_shape == item_shape:
                                score += 0.1
                                break
                            elif in_shape in ['skinny', 'slim', 'tight'] and item_shape in ['fitted', 'slim',
                                                                                            'tailored']:
                                score += 0.08
                                break
                            elif in_shape in ['loose', 'oversized', 'relaxed'] and item_shape in ['loose', 'oversized',
                                                                                                  'relaxed']:
                                score += 0.08
                                break
                            elif in_shape in ['straight', 'regular'] and item_shape in ['straight', 'regular',
                                                                                        'classic']:
                                score += 0.08
                                break
                            elif in_shape in ['bootcut', 'flare', 'wide'] and item_shape in ['bootcut', 'flare',
                                                                                             'wide']:
                                score += 0.08
                                break

                # 处理材质匹配
                if input_material != 'unknown' and item['material'] != 'unknown':
                    # 将逗号分隔的材质字符串转换为列表
                    input_materials = input_material.split(',')
                    item_materials = item['material'].split(',')

                    # 计算材质匹配度
                    for in_material in input_materials:
                        for item_material in item_materials:
                            if in_material == item_material:
                                score += 0.1
                                break
                            elif in_material in ['cotton', 'linen', 'jersey'] and item_material in ['cotton', 'linen',
                                                                                                    'jersey']:
                                score += 0.08
                                break
                            elif in_material in ['leather', 'suede'] and item_material in ['leather', 'suede']:
                                score += 0.08
                                break
                            elif in_material in ['wool', 'cashmere', 'silk'] and item_material in ['wool', 'cashmere',
                                                                                                   'silk']:
                                score += 0.08
                                break
                            elif in_material in ['denim'] and item_material in ['cotton', 'jersey']:
                                score += 0.05
                                break
                            elif in_material in ['polyester', 'nylon'] and item_material in ['polyester', 'nylon']:
                                score += 0.05
                                break

                item['match_score'] = score

            # 找到最高分的项目
            if items:  # 确保items不为空
                best_item = max(items, key=lambda x: x.get('match_score', x.get('distance', 0)))
                recommendations[category] = best_item
                safe_log_info(
                    f"Best item for {category}: {best_item['image_name']} with score {best_item.get('match_score', 'unknown')}")
            else:
                safe_log_info(f"No valid items for category {category} after scoring")

        except Exception as e:
            safe_log_info(f"Error processing category {category}: {e}")
            safe_log_info(traceback.format_exc())
            continue

    safe_log_info(f"Total recommendations found: {len(recommendations)}")
    return recommendations


def find_most_similar_item_with_attributes(category: str, embedding: np.ndarray) -> Dict:
    if coll is None:
        return {"color": "unknown", "shape": "unknown", "material": "unknown"}
    try:
        embedding = np.asarray(embedding)
        if embedding.dtype != np.float32:
            embedding = embedding.astype(np.float32)
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)  # 转为二维，Milvus期望的格式

        safe_category = category.replace("'", "\\'")
        expr = f"category == '{safe_category}'"

        search_results = coll.search(
            data=embedding,
            anns_field='embedding',
            param={"metric_type": "IP", "params": {"nprobe": 128}},
            limit=30,
            expr=expr,
            output_fields=['color', 'shape', 'material'],
            timeout=10.0
        )

        for result in search_results[0]:
            color = getattr(result.entity, 'color', 'unknown')
            shape = getattr(result.entity, 'shape', 'unknown')
            material = getattr(result.entity, 'material', 'unknown')
            if color != 'unknown' or shape != 'unknown' or material != 'unknown':
                return {"color": color, "shape": shape, "material": material}

        if search_results[0]:
            result = search_results[0][0]
            return {
                "color": getattr(result.entity, 'color', 'unknown'),
                "shape": getattr(result.entity, 'shape', 'unknown'),
                "material": getattr(result.entity, 'material', 'unknown')
            }

    except Exception as e:
        safe_log_info(f"Error finding similar item with attributes: {e}")
        safe_log_info(traceback.format_exc())

    return {"color": "unknown", "shape": "unknown", "material": "unknown"}



@app.post("/recommend")
async def recommend_complete_outfit(file: UploadFile = File(...)):
    start_time = time.time()
    try:
        img = Image.open(io.BytesIO(await file.read())).convert('RGB')
        safe_log_info(f"Uploaded image size: {img.size}")

        input_category = detect_clothing_category(img)
        safe_log_info(f"Detected category: {input_category}")

        # 避免使用GPU并使用较低精度
        inputs = extractor(images=img, return_tensors='pt')['pixel_values']
        with torch.no_grad():
            # 在一些量化加载下，模型可能在 GPU/CPU 的不同位置，直接调用并把结果转到 CPU
            emb = vit(pixel_values=inputs).last_hidden_state[:, 0, :].cpu().numpy()

        # 确保向量类型为float32以匹配Milvus中的VECTOR_FLOAT类型
        if emb.dtype != np.float32:
            emb = emb.astype(np.float32)

        # 释放输入变量以节省内存
        del inputs
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

        input_attributes = find_most_similar_item_with_attributes(input_category, emb)
        input_color = input_attributes["color"]
        input_shape = input_attributes["shape"]
        input_material = input_attributes["material"]

        # 记录输入属性
        safe_log_info(f"Input attributes - Color: {input_color}, Shape: {input_shape}, Material: {input_material}")

        recommendations = get_complementary_items(
            input_category, emb, input_color, input_shape, input_material)

        safe_log_info(f"Recommendations received: {recommendations}")

        final_outfit = {}

        # 方案1：不再筛选，直接使用所有推荐结果构建final_outfit
        for category, item in recommendations.items():
            final_outfit[category] = item

        safe_log_info(f"Final outfit: {final_outfit}")

        # 添加容错处理，确保即使没有推荐结果也能继续执行
        if not recommendations and input_category != 'unknown':
            # 如果没有推荐结果，尝试获取一些基本的推荐项
            safe_log_info("No recommendations found, attempting to get basic items")
            complementary_categories = get_complementary_categories(input_category)
            if complementary_categories:
                # 尝试获取第一个互补类别的一些基本项目
                try:
                    basic_items = search_items_by_category(complementary_categories[0], emb, limit=5)
                    if basic_items:
                        recommendations[complementary_categories[0]] = basic_items[0]
                        final_outfit[complementary_categories[0]] = basic_items[0]
                        safe_log_info(f"Added basic item for fallback: {basic_items[0]}")
                except Exception as e:
                    safe_log_info(f"Error getting basic items: {e}")
                    safe_log_info(traceback.format_exc())
                    # 即使获取基本项目失败，也要确保能继续执行
                    pass

        # 使用0.6B模型生成要点
        category_names = {
            'top': '上衣',
            'bottom': '下装',
            'pants': '裤子',
            'skirt': '裙子',
            'dress': '连衣裙',
            'outer': '外套',
            'shoes': '鞋子',
            'bag': '包袋',
            'belt': '腰带',
            'hat': '帽子',
            'scarf': '围巾',
            'accessories': '配饰'
        }

        chinese_category = category_names.get(input_category, input_category)

        # 构建输出格式
        if final_outfit:
            outline_points = []
            for i, (category, item) in enumerate(final_outfit.items(), 1):
                color = item.get('color', 'unknown')
                shape = item.get('shape', 'unknown')
                material = item.get('material', 'unknown')
                chinese_cat = category_names.get(category, category)
                outline_points.append(f"{i}. 搭配{chinese_cat}(颜色:{color},款式:{shape},材质:{material})")

            outline_text = "\n".join(outline_points)
        else:
            outline_prompt = f"""你是一位专业的时尚搭配师，请根据用户的单品提供3-5条核心搭配建议。

                    用户单品：{input_color}{chinese_category}

                    要求：
                    1. 搭配建议应自然流畅，突出整体造型效果
                    2. 结合颜色、款式、材质等要素，体现风格特点
                    3. 建议可涉及上衣、下装、外套、鞋包、配饰等不同品类
                    4. 每条建议之间保持多样化
                    5. 不要输出额外解释或免责声明

                    请直接给出搭配建议：
                    """

            outline_text = generate_outline_local(outline_prompt, max_tokens=500)

            if not outline_text or "错误" in outline_text or "无法生成" in outline_text or len(outline_text) < 10:
                outline_text = f"建议搭配{chinese_category}可以考虑以下几种方式：\n1. 选择简约风格的鞋子来平衡整体造型\n2. 搭配一款时尚的包袋增加亮点\n3. 根据季节选择合适的外套进行叠穿"

        safe_log_info(f"Generated outline text: {outline_text}")

        # 使用大模型生成高质量文案
        if LLM_API_KEY:
            # 构造给大模型的提示词，让模型输出包含用户单品信息和穿搭建议
            final_prompt = f"""你是一个专业的时尚搭配师，请根据以下信息生成详细的穿搭建议：

                    用户上传的单品信息：
                    - 品类：{input_category}
                    - 颜色：{input_color}
                    - 款式：{input_shape if input_shape != 'unknown' else '未识别'}
                    - 材质：{input_material if input_material != 'unknown' else '未识别'}

                    检索到的推荐搭配衣品或上游本地大模型生成的推荐搭配要点：
                    {outline_text}

                    注意上游模型生产内容会有一些无用信息或错误信息，请正确识别保留有效的信息，作为穿搭要点：

                    请用中文提供详细且实用的穿搭建议，要求：
                    1. 首先描述用户单品的特点和搭配优势
                    2. 详细说明每个推荐单品的搭配理由
                    3. 给出整体搭配的风格描述
                    4. 提供适合的穿着场景
                    5. 给出个性化调整建议
                    6. 提供购买关键词以便用户搜索类似单品

                    输出格式要求：
                    📌 单品分析
                    [用户单品的特点和搭配优势]

                    👗 搭配建议
                    [详细说明每个推荐单品的搭配理由]

                    💄 风格定位
                    [整体搭配的风格描述]

                    🎯 适用场景
                    [适合的穿着场景]

                    🎨 个性调整
                    [个性化调整建议]

                    🛍️ 购买指南
                    [购买关键词，帮助用户搜索类似单品]"""

            outfit_text = call_qwen_api(final_prompt)
        else:
            # 如果没有API密钥，直接使用要点作为输出
            outfit_text = f"核心搭配要点：\n{outline_text}\n\n注意：如需生成详细文案，请配置大模型API密钥。"

        safe_log_info(f"Generated outfit text length: {len(outfit_text)}")
        try:
            safe_log_info(f"Generated outfit text preview: {outfit_text[:500]}...")
        except Exception:
            safe_log_info("Generated outfit text preview: [Contains special characters]")

        end_time = time.time()
        safe_log_info(f"Total processing time: {end_time - start_time:.2f} seconds")

        # 简化返回结构，只返回一个模型输出
        return {
            "outfit_description": outfit_text
        }

    except Exception as e:
        safe_log_info(f"Error in recommend_complete_outfit: {e}")
        safe_log_info(traceback.format_exc())
        return {
            "outfit_description": f"处理出错: {str(e)}"
        }


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


def main():
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1,
        log_level="info"
    )


if __name__ == "__main__":
    main()
