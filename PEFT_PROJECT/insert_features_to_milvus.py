# insert_features_to_milvus.py
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, MilvusClient
import os
import torch
from transformers import ViTModel, ViTImageProcessor
from peft import PeftModel
from data.fashion_dataset import FashionDataset
from torch.utils.data import DataLoader, Subset
import time
from typing import List
import random
import re
import pandas as pd
import numpy as np


def connect_milvus(max_retries=5, retry_delay=5):
    # Milvus连接参数
    MILVUS_HOST = '150.158.55.76'
    MILVUS_PORT = '19530'
    COLLECTION_NAME = "fashion_features"

    # 尝试多次连接
    for attempt in range(max_retries):
        try:
            connections.connect(host=MILVUS_HOST, port=MILVUS_PORT, timeout=10)
            print(f"[Milvus] 连接成功: {MILVUS_HOST}:{MILVUS_PORT}")
            break
        except Exception as e:
            print(f"[Milvus] 连接尝试 {attempt + 1}/{max_retries} 失败: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                raise

    client = MilvusClient(uri=f"http://{MILVUS_HOST}:{MILVUS_PORT}")

    # 定义集合schema - 添加更多属性字段
    schema = client.create_schema()
    schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field("image_name", DataType.VARCHAR, max_length=255)
    schema.add_field("category", DataType.VARCHAR, max_length=1000)
    schema.add_field("color", DataType.VARCHAR, max_length=1000)
    schema.add_field("shape", DataType.VARCHAR, max_length=1000)
    schema.add_field("material", DataType.VARCHAR, max_length=1000)
    schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=768)

    # 准备索引参数
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="embedding",
        index_name="embedding_index",
        index_type="IVF_FLAT",
        metric_type="IP",
        params={"nlist": 128}
    )

    # 如果集合已存在，先释放、删除索引和集合，再重新创建
    if COLLECTION_NAME in client.list_collections():
        print(f"[Milvus] 集合 {COLLECTION_NAME} 已存在，重新创建...")
        try:
            client.release_collection(collection_name=COLLECTION_NAME)
            print("[Milvus] 集合已释放")
        except Exception as e:
            print(f"[Milvus] 释放集合失败: {e}")

        try:
            client.drop_index(collection_name=COLLECTION_NAME, index_name="embedding_index")
            print("[Milvus] 索引已删除")
        except Exception as e:
            print(f"[Milvus] 删除索引失败: {e}")

        try:
            client.drop_collection(collection_name=COLLECTION_NAME)
            print("[Milvus] 集合已删除")
        except Exception as e:
            print(f"[Milvus] 删除集合失败: {e}")

    # 创建新集合
    client.create_collection(
        collection_name=COLLECTION_NAME,
        schema=schema,
        index_params=index_params
    )
    print(f"[Milvus] 集合 {COLLECTION_NAME} 创建成功")

    # 获取集合对象
    coll = Collection(name=COLLECTION_NAME)
    return coll, client


def extract_category_from_filename(filename: str) -> str:
    """
    从文件名中提取类别信息
    文件名格式: {原始文件名}_{类别}_crop_{序号}.jpg
    """
    # 使用正则表达式提取类别
    pattern = r'_([a-zA-Z]+)_crop_\d+\.'
    match = re.search(pattern, filename)
    if match:
        return match.group(1).lower()
    return 'unknown'


def extract_attributes_from_description(description: str) -> dict:
    """
    从描述中提取颜色、款式、材质信息
    """
    attributes = {
        'color': 'unknown',
        'shape': 'unknown',
        'material': 'unknown'
    }

    # 颜色关键词
    color_keywords = {
        'black': ['black', '黑色', 'charcoal', 'onyx', 'obsidian', 'ebony'],
        'white': ['white', '白色', 'ivory', 'off-white', 'cream', 'snow', 'eggshell'],
        'red': ['red', '红色', 'burgundy', 'crimson', 'scarlet', 'maroon', 'ruby', 'cardinal'],
        'blue': ['blue', '蓝色', 'indigo', 'navy', 'azure', 'cerulean', 'sapphire', 'turquoise', 'robin-egg',
                 'sky blue'],
        'green': ['green', '绿色', 'olive', 'emerald', 'lime', 'forest', 'mint', 'teal', 'sea green', 'jade'],
        'yellow': ['yellow', '黄色', 'golden', 'gold', 'mustard', 'amber', 'sunshine', 'canary', 'lemon'],
        'purple': ['purple', '紫色', 'violet', 'lavender', 'mauve', 'plum', 'orchid'],
        'pink': ['pink', '粉色', 'rose', 'coral', 'fuchsia', 'magenta', 'salmon', 'blush'],
        'orange': ['orange', '橙色', 'coral', 'rust', 'copper', 'apricot', 'tangerine'],
        'brown': ['brown', '棕色', 'beige', 'khaki', 'taupe', 'tan', 'camel', 'bronze', 'cognac', 'mahogany', 'coffee'],
        'gray': ['gray', 'grey', '灰色', 'charcoal', 'heather', 'slate', 'silver', 'gunmetal'],
        'multicolor': ['multicolored', 'multi-color', 'multi', 'variegated', 'heathered', 'tonal', 'tones'],
        'metallic': ['metallic', 'gunmetal-tone', 'silver-tone', 'gold-tone'],
        'rainbow': ['rainbow', 'prismatic']
    }

    # 查找颜色
    desc_lower = description.lower()
    found_colors = []
    for color, keywords in color_keywords.items():
        if any(keyword in desc_lower for keyword in keywords):
            found_colors.append(color)

    # 如果找到多个颜色，使用逗号连接；如果没找到，保持'unknown'
    if found_colors:
        attributes['color'] = ','.join(found_colors)

    # 款式关键词（包括图案、设计特征、闭合方式等）
    shape_keywords = {
        # 裤型和基本款式
        'skinny': ['skinny', '紧身'],
        'slim': ['slim', '修身', 'narrow', 'tailored'],
        'regular': ['regular', '常规', 'classic'],
        'loose': ['loose', '宽松', 'relaxed', 'oversized', 'baggy'],
        'straight': ['straight', '直筒'],
        'tapered': ['tapered', '锥形'],
        'bootcut': ['bootcut', '靴型'],
        'flare': ['flare', '喇叭'],
        'wide': ['wide', '宽', 'wide-leg'],
        'cropped': ['cropped', '截短', 'short'],
        'fitted': ['fitted', '合身'],
        'high-waisted': ['high-waisted', '高腰'],
        'low-rise': ['low-rise', '低腰'],
        'mid-rise': ['mid-rise', '中腰'],
        'drop-crotch': ['drop crotch', 'drop-crotch'],
        'pleated': ['pleated', '褶裥'],
        'boxy': ['boxy', '箱型'],
        'long': ['long', '长款'],
        'short': ['short', '短款'],

        # 图案
        'floral': ['floral', '花卉', 'flower'],
        'camo': ['camo', 'camouflage', '迷彩'],
        'polka dot': ['polka dot', '圆点'],
        'stripe': ['stripe', '条纹', 'striped'],
        'plaid': ['plaid', '格子', 'check'],
        'houndstooth': ['houndstooth', '千鸟格'],
        'gingham': ['gingham', '方格'],
        'herringbone': ['herringbone', '人字纹'],
        'paisley': ['paisley', '佩斯利'],
        'animal print': ['animal print', '豹纹', '蛇纹', '斑马纹', 'leopard', 'snake', 'zebra'],
        'geometric': ['geometric', '几何'],
        'anchor': ['anchor', '锚'],
        'skull': ['skull', '骷髅'],
        'tie-dye': ['tie-dye', '扎染'],
        'ombre': ['ombre', '渐变'],
        'diamond': ['diamond', '菱形'],

        # 领型设计
        'crewneck': ['crewneck', '圆领'],
        'v-neck': ['v-neck', 'v领'],
        'spread collar': ['spread collar', '宽领'],
        'button-down collar': ['button-down collar', '纽扣领'],
        'shawl collar': ['shawl collar', '披肩领'],
        'notched lapel': ['notched lapel', '缺口领'],
        'band collar': ['band collar', '立领'],
        'hooded': ['hooded', '连帽'],
        'scoopneck': ['scoopneck', '挖空领'],

        # 袖型设计
        'long sleeve': ['long sleeve', '长袖'],
        'short sleeve': ['short sleeve', '短袖'],
        'sleeveless': ['sleeveless', '无袖'],
        'raglan sleeves': ['raglan sleeves', '插肩袖'],

        # 口袋设计
        'five-pocket': ['five-pocket', '五口袋'],
        'flap pockets': ['flap pockets', '带盖口袋'],
        'welt pockets': ['welt pockets', '贴袋'],
        'breast pocket': ['breast pocket', '胸袋'],
        'patch pockets': ['patch pockets', '贴边口袋'],

        # 闭合方式
        'button closure': ['button closure', 'button-down', '纽扣', '纽扣闭合'],
        'zip closure': ['zip closure', 'zip', 'zippered', '拉链'],
        'snap-stud closure': ['snap-stud', '按扣'],
        'drawstring closure': ['drawstring', '抽绳'],
        'elasticized': ['elasticized', '弹性'],

        # 特殊设计特征
        'distressed': ['distressed', '破洞', '磨损'],
        'embroidered': ['embroidered', '刺绣'],
        'printed': ['printed', '印花'],
        'knit': ['knit', '针织'],
        'ribbed': ['ribbed', '罗纹'],
        'sheer': ['sheer', '透明'],
        'layered': ['layered', '层次'],
        'asymmetrical': ['asymmetrical', '不对称'],
        'ruched': ['ruched', '褶皱'],
        'frayed': ['frayed', '毛边'],
        'pleated': ['pleated', '百褶'],
        'tuxedo': ['tuxedo', '燕尾服'],
        'double-breasted': ['double-breasted', '双排扣'],
        'single-breasted': ['single-breasted', '单排扣'],
        'peplum': ['peplum', '荷叶边'],
        'belted': ['belted', '系带'],
        'cutout': ['cutout', '镂空'],
        'ruffled': ['ruffled', '褶边'],
        'padded': ['padded', '填充'],
        'vented': ['vented', '开衩'],
        'lined': ['lined', '衬里'],
        'seamed': ['seamed', '接缝'],
        'gathered': ['gathered', '皱褶'],
        'draped': ['draped', '垂坠'],
        'structured': ['structured', '结构式'],
        'deconstructed': ['deconstructed', '解构'],
        'tapered': ['tapered', '收窄'],
        'cinched': ['cinched', '束腰'],
        'tiered': ['tiered', '分层'],
        'wrap': ['wrap', '裹身'],
        'cold shoulder': ['cold shoulder', '露肩'],
        'off-shoulder': ['off-shoulder', '露肩'],
        'halter': ['halter', '绕颈'],
        'strapless': ['strapless', '无肩带'],
        'backless': ['backless', '露背'],
        'illusion': ['illusion', '透视'],
        'cutwork': ['cutwork', '雕花'],
        'lace-up': ['lace-up', '系带'],
        'tie-front': ['tie-front', '系带前襟'],
        'peekaboo': ['peekaboo', '透视'],
        'mesh': ['mesh', '网眼'],
        'perforated': ['perforated', '穿孔'],
        'slit': ['slit', '开衩'],
        'split': ['split', '分叉'],
        'paneled': ['paneled', '拼接'],
        'colorblocked': ['colorblocked', '色块'],
        'marbleized': ['marbleized', '大理石纹'],
        'textured': ['textured', '纹理'],
        'embossed': ['embossed', '压花'],
        'quilted': ['quilted', '绗缝'],
        'puckered': ['puckered', '皱褶'],
        'smocked': ['smocked', '皱褶'],
        'tucked': ['tucked', ' tucked'],
        'layered': ['layered', '叠层'],
        'asymmetrical': ['asymmetrical', '不对称'],
        'irregular': ['irregular', '不规则'],
        'raw': ['raw', '毛边'],
        'washed': ['washed', '水洗'],
        'faded': ['faded', '褪色'],
        'bleached': ['bleached', '漂白'],
        'acid-washed': ['acid-washed', '酸洗'],
        'whiskering': ['whiskering', '猫须'],
        'sandblasted': ['sandblasted', '喷砂'],
        'hand-distressed': ['hand-distressed', '手工做旧'],
        'paint splatter': ['paint splatter', '油漆泼溅'],
        'paint speckling': ['paint speckling', '油漆斑点'],
        'contrast': ['contrast', '对比'],
        'tonal': ['tonal', '同色'],
        'reversible': ['reversible', '双面'],
        'convertible': ['convertible', '可转换'],
        'adjustable': ['adjustable', '可调节'],
        'removable': ['removable', '可拆卸'],
        'detachable': ['detachable', '可拆卸'],
        'interchangeable': ['interchangeable', '可互换']
    }

    # 查找款式（支持多个）
    found_shapes = []
    for shape, keywords in shape_keywords.items():
        if any(keyword in desc_lower for keyword in keywords):
            found_shapes.append(shape)

    # 如果找到多个款式，使用逗号连接；如果没找到，保持'unknown'
    if found_shapes:
        attributes['shape'] = ','.join(found_shapes)

    # 材质关键词
    material_keywords = {
        'cotton': ['cotton', '棉', 'cotton-wool', 'cotton-piqu\u00e9', 'cotton-linen'],
        'denim': ['denim', '牛仔', 'jeans', 'buffed denim'],
        'silk': ['silk', '丝绸', 'silken'],
        'leather': ['leather', '皮革', 'lambskin', 'calfskin', 'suede', 'buffed leather', 'grained calfskin'],
        'wool': ['wool', '羊毛', 'wool-mohair', 'cashmere', 'merino'],
        'linen': ['linen', '亚麻', 'linen-cotton'],
        'polyester': ['polyester', '聚酯'],
        'nylon': ['nylon', '尼龙'],
        'velvet': ['velvet', '天鹅绒', 'velvety'],
        'suede': ['suede', '麂皮'],
        'flannel': ['flannel', '法兰绒'],
        'fleece': ['fleece', '抓绒', 'fleecy'],
        'cashmere': ['cashmere', '羊绒'],
        'jersey': ['jersey', '针织'],
        'satin': ['satin', '缎子'],
        'neoprene': ['neoprene', '氯丁橡胶'],
        'ribbed': ['ribbed', '罗纹'],
        'sheer': ['sheer', '透明'],
        'lace': ['lace', '蕾丝'],
        'mesh': ['mesh', '网眼'],
        'corduroy': ['corduroy', '灯芯绒'],
        'chenille': ['chenille', '绳绒'],
        'tweed': ['tweed', '粗花呢'],
        'jacquard': ['jacquard', '提花'],
        'plaid': ['plaid', '格子'],
        'gabardine': ['gabardine', '华达呢'],
        'chiffon': ['chiffon', '雪纺'],
        'crepe': ['crepe', '绉纱'],
        'twill': ['twill', '斜纹'],
        'canvas': ['canvas', '帆布'],
        'organza': ['organza', '欧根纱'],
        'tulle': ['tulle', '塔夫绸'],
        'voile': ['voile', '巴里纱'],
        'muslin': ['muslin', '棉纱'],
        'poplin': ['poplin', '府绸'],
        'seersucker': ['seersucker', '泡泡纱'],
        'terry': ['terry', '毛巾布'],
        'waffle': ['waffle', '华夫格'],
        'sherpa': ['sherpa', '雪帕尔'],
        'felt': ['felt', '毛毡'],
        'fur': ['fur', '毛皮'],
        'faux fur': ['faux fur', '人造毛皮'],
        'faux leather': ['faux leather', '人造革'],
        'pleather': ['pleather', '皮面革'],
        'spandex': ['spandex', '氨纶'],
        'elastane': ['elastane', '弹性纤维'],
        'rayon': ['rayon', '人造丝'],
        'viscose': ['viscose', '粘胶纤维'],
        'modal': ['modal', '莫代尔'],
        'bamboo': ['bamboo', '竹纤维'],
        'tencel': ['tencel', '天丝'],
        'lyocell': ['lyocell', '莱赛尔'],
        'acrylic': ['acrylic', '腈纶'],
        'microfiber': ['microfiber', '超细纤维'],
        'supplex': ['supplex', '超级莱卡'],
        'cupro': ['cupro', '铜氨纤维']
    }

    # 查找材质（支持多个）
    found_materials = []
    for material, keywords in material_keywords.items():
        if any(keyword in desc_lower for keyword in keywords):
            found_materials.append(material)

    # 如果找到多个材质，使用逗号连接；如果没找到，保持'unknown'
    if found_materials:
        attributes['material'] = ','.join(found_materials)

    return attributes


# 定义类别映射关系：将CSV类别映射到检测模型类别
CSV_TO_MODEL_CATEGORY_MAP = {
    'BACKPACKS': 'bag',
    'BAG ACCESSORIES': 'bag',
    'BELTS & SUSPENDERS': 'belt',
    'BOAT SHOES & MOCCASINS': 'shoes',
    'BOOTS': 'shoes',
    'BRIEFCASES': 'bag',
    'CLUTCHES & POUCHES': 'bag',
    'DRESSES': 'dress',
    'DUFFLE & TOP HANDLE BAGS': 'bag',
    'DUFFLE BAGS': 'bag',
    'ESPADRILLES': 'shoes',
    'FLATS': 'shoes',
    'GLOVES': 'gloves',
    'HATS': 'headwear',
    'HEELS': 'shoes',
    'JACKETS & COATS': 'outer',
    'JEANS': 'pants',
    'JUMPSUITS': 'bottom',
    'KEYCHAINS': 'bag',
    'LACE UPS': 'shoes',
    'LINGERIE': 'top',
    'LOAFERS': 'shoes',
    'MESSENGER BAGS': 'bag',
    'MESSENGER BAGS & SATCHELS': 'bag',
    'MONKSTRAPS': 'shoes',
    'PANTS': 'pants',
    'POUCHES & DOCUMENT HOLDERS': 'bag',
    'SANDALS': 'shoes',
    'SCARVES': 'scarf',
    'SHIRTS': 'top',
    'SHORTS': 'shorts',
    'SHOULDER BAGS': 'bag',
    'SKIRTS': 'skirt',
    'SNEAKERS': 'shoes',
    'SOCKS': 'socks',
    'SUITS & BLAZERS': 'outer',
    'SWEATERS': 'top',
    'SWIMWEAR': 'top',
    'TIES': 'top',
    'TOPS': 'top',
    'TOTE BAGS': 'bag',
    'TRAVEL BAGS': 'bag',
    'UNDERWEAR & LOUNGEWEAR': 'top'
}


def reconnect_milvus(max_retries=3):
    """重新连接Milvus"""
    MILVUS_HOST = '150.158.55.76'
    MILVUS_PORT = '19530'

    for attempt in range(max_retries):
        try:
            connections.connect(host=MILVUS_HOST, port=MILVUS_PORT, timeout=10)
            print(f"[Milvus] 重新连接成功: {MILVUS_HOST}:{MILVUS_PORT}")
            return True
        except Exception as e:
            print(f"[Milvus] 重新连接尝试 {attempt + 1}/{max_retries} 失败: {e}")
            time.sleep(5)
    return False


def insert_with_retry(coll, names: List, categories: List, colors: List, shapes: List, materials: List, embs: List,
                      max_retries: int = 3) -> bool:
    """带重试机制的插入操作 - 包含更多属性信息"""
    for attempt in range(max_retries):
        try:
            insert_result = coll.insert([names, categories, colors, shapes, materials, embs])
            coll.flush()
            return True
        except Exception as e:
            print(f"[Milvus] 插入尝试 {attempt + 1}/{max_retries} 失败: {e}")
            if attempt < max_retries - 1:
                # 尝试重新连接
                if reconnect_milvus():
                    # 重新获取集合对象
                    try:
                        coll = Collection(name="fashion_features")
                    except:
                        pass
                time.sleep(10)  # 等待一段时间再重试
            else:
                print(f"[Milvus] 插入最终失败，经过 {max_retries} 次尝试")
                return False
    return False


def extract_and_insert(model, processor, csv_path, sample_ratio=0.1, sample_count=None):
    """
    提取特征并插入Milvus

    Args:
        model: 模型
        processor: 图像处理器
        csv_path: CSV文件路径
        sample_ratio: 采样比例 (0-1之间)
        sample_count: 采样数量 (如果指定，则优先使用此值)
    """
    try:
        coll, client = connect_milvus()
    except Exception as e:
        print(f"[Milvus] 连接失败: {e}")
        return

    model.eval()

    # 加载CSV数据
    print(f"[CSV] 加载CSV数据: {csv_path}")
    try:
        csv_data = pd.read_csv(csv_path, encoding='utf-8')
        # 创建索引映射，使用index列作为键
        csv_index_map = {}
        for _, row in csv_data.iterrows():
            csv_index_map[str(row['index'])] = {
                'description': row['description'],
                'category': row['category']
            }
        print(f"[CSV] 成功加载CSV数据，共 {len(csv_data)} 条记录")
    except Exception as e:
        print(f"[CSV] 加载CSV数据失败: {e}")
        return

    # 适配FashionDataset参数
    full_dataset = FashionDataset(extractor=processor, root='outputs/train/fashiongen_crops')
    total_count = len(full_dataset)
    print(f"[数据集] 总共 {total_count} 张图片")

    if total_count == 0:
        print("[错误] 数据集为空，请检查路径和文件")
        return

    # 随机采样数据
    if sample_count is not None:
        sample_size = min(sample_count, total_count)
        print(f"[采样] 采样数量: {sample_size} 张图片")
    else:
        sample_size = int(total_count * sample_ratio)
        print(f"[采样] 采样比例: {sample_ratio * 100:.1f}%, 采样数量: {sample_size} 张图片")

    # 生成随机索引
    indices = list(range(total_count))
    random.shuffle(indices)
    sample_indices = indices[:sample_size]

    # 创建采样子集
    dataset = Subset(full_dataset, sample_indices)
    print(f"[采样] 实际采样 {len(dataset)} 张图片")

    # 优化数据加载器配置
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    names, categories, colors, shapes, materials, embs = [], [], [], [], [], []
    total_inserted = 0
    insert_batch_size = 1000

    # 记录开始时间
    start_time = time.time()
    last_log_time = start_time

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # 显示处理进度
            if batch_idx % 10 == 0:
                current_time = time.time()
                elapsed_time = current_time - last_log_time
                total_elapsed = current_time - start_time

                if batch_idx > 0:
                    batches_per_second = 10 / elapsed_time
                    images_per_second = batches_per_second * 32
                    print(f"[进度] 批次 {batch_idx}, "
                          f"速度 {images_per_second:.1f} 图片/秒, "
                          f"已用时 {total_elapsed / 3600:.2f} 小时, "
                          f"总计处理 {total_inserted} 条")

                last_log_time = current_time

            pixel_values = batch['pixel_values'].cuda(non_blocking=True)
            image_names = batch['image_name']

            # 提取特征 - 使用混合精度
            with torch.cuda.amp.autocast():
                outputs = model(pixel_values=pixel_values)
                # 使用[CLS] token的特征
                embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

            # 转换image_names为列表形式
            if isinstance(image_names, list):
                names.extend(image_names)
                # 提取类别信息
                batch_categories = [extract_category_from_filename(name) for name in image_names]
                categories.extend(batch_categories)

                # 从CSV提取属性信息 - 改进版本
                for name in image_names:
                    # 从文件名中提取原始索引
                    pattern = r'^(\d+)_'  # 匹配文件名开头的数字
                    match = re.search(pattern, name)
                    image_category = extract_category_from_filename(name)
                    if match:
                        original_index = match.group(1)
                        if original_index in csv_index_map:
                            # 获取CSV中的类别
                            csv_category = csv_index_map[original_index]['category']
                            # 映射CSV类别到模型类别
                            mapped_category = CSV_TO_MODEL_CATEGORY_MAP.get(csv_category.upper(), 'unknown')

                            # 只有当图片类别与CSV类别匹配时才使用CSV中的描述信息
                            if image_category == mapped_category:
                                description = csv_index_map[original_index]['description']
                                attributes = extract_attributes_from_description(description)
                                colors.append(attributes['color'])
                                shapes.append(attributes['shape'])
                                materials.append(attributes['material'])
                            else:
                                # 如果类别不匹配，属性标记为unknown
                                colors.append('unknown')
                                shapes.append('unknown')
                                materials.append('unknown')
                        else:
                            # 如果找不到对应的描述，使用文件名中的类别信息
                            colors.append('unknown')
                            shapes.append('unknown')
                            materials.append('unknown')
                    else:
                        colors.append('unknown')
                        shapes.append('unknown')
                        materials.append('unknown')
            else:
                # 如果是tensor，需要转换为列表
                name_list = [str(name) for name in image_names]
                names.extend(name_list)
                # 提取类别信息
                batch_categories = [extract_category_from_filename(name) for name in name_list]
                categories.extend(batch_categories)

                # 从CSV提取属性信息 - 改进版本
                for name in name_list:
                    # 从文件名中提取原始索引
                    pattern = r'^(\d+)_'  # 匹配文件名开头的数字
                    match = re.search(pattern, name)
                    image_category = extract_category_from_filename(name)
                    if match:
                        original_index = match.group(1)
                        if original_index in csv_index_map:
                            # 获取CSV中的类别
                            csv_category = csv_index_map[original_index]['category']
                            # 映射CSV类别到模型类别
                            mapped_category = CSV_TO_MODEL_CATEGORY_MAP.get(csv_category.upper(), 'unknown')

                            # 只有当图片类别与CSV类别匹配时才使用CSV中的描述信息
                            if image_category == mapped_category:
                                description = csv_index_map[original_index]['description']
                                attributes = extract_attributes_from_description(description)
                                colors.append(attributes['color'])
                                shapes.append(attributes['shape'])
                                materials.append(attributes['material'])
                            else:
                                # 如果类别不匹配，属性标记为unknown
                                colors.append('unknown')
                                shapes.append('unknown')
                                materials.append('unknown')
                        else:
                            # 如果找不到对应的描述，使用文件名中的类别信息
                            colors.append('unknown')
                            shapes.append('unknown')
                            materials.append('unknown')
                    else:
                        colors.append('unknown')
                        shapes.append('unknown')
                        materials.append('unknown')

            embs.extend(embeddings)

            # 批量插入以避免内存问题
            if len(names) >= insert_batch_size:
                if names:
                    success = insert_with_retry(coll, names, categories, colors, shapes, materials, embs, max_retries=3)
                    if success:
                        total_inserted += len(names)
                        print(f"[Milvus] 插入向量 {len(names)} 条，总计: {total_inserted}")
                    else:
                        print(f"[错误] 插入失败，退出")
                        return
                names, categories, colors, shapes, materials, embs = [], [], [], [], [], []

    # 插入剩余的数据
    if names:
        success = insert_with_retry(coll, names, categories, colors, shapes, materials, embs, max_retries=3)
        if success:
            total_inserted += len(names)
            print(f"[Milvus] 插入向量 {len(names)} 条，总计: {total_inserted}")
        else:
            print(f"[错误] 插入剩余数据失败")
            return

    total_time = time.time() - start_time
    print(f"[Milvus] 数据插入完成，共处理 {total_inserted} 条记录")
    print(f"[总计] 用时 {total_time / 3600:.2f} 小时 ({total_time:.0f} 秒)")

    if total_inserted == 0:
        print("[错误] 没有数据被插入")
        return

    try:
        # 加载集合
        print("[Milvus] 开始加载集合...")
        coll.load()
        print("[Milvus] 集合加载完成")

        # 验证集合状态
        print(f"[Milvus] 集合中的实体数: {coll.num_entities}")

    except Exception as e:
        print(f"[Milvus] 后处理失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    # 加载预训练模型
    print("[模型] 加载ViT基础模型...")
    base_model = ViTModel.from_pretrained('google/vit-base-patch16-224')

    # 检查适配器权重路径是否存在
    adapter_path = "outputs/vit_lora_best_peft"
    if not os.path.exists(adapter_path):
        print(f"警告: 适配器路径 {adapter_path} 不存在，使用基础模型")
        model = base_model
    else:
        print(f"[模型] 加载LoRA适配器权重从 {adapter_path}...")
        # 加载LoRA适配器权重
        model = PeftModel.from_pretrained(base_model, adapter_path)
    model = model.cuda()

    # 加载图像处理器
    print("[处理器] 加载ViT图像处理器...")
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')

    print("[开始] 提取特征并插入Milvus...")
    # 按比例采样 (例如采样15%的数据)
    extract_and_insert(model, processor, 'data/fashiongen_train.csv', sample_ratio=0.15)
