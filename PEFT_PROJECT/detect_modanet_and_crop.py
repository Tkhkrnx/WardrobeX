from ultralytics import YOLO
import cv2
import os
from pathlib import Path
import json

# 加载你训练好的 YOLOv8 模型（必须是 YOLOv8 格式）
model = YOLO('modanet_clothing_model.pt')  # 你自己训练的衣物检测模型

# 设置输入输出路径
image_dir = 'data/images/train'  # 输入图像文件夹
output_dir = Path('outputs/train')  # 输出目录
output_dir.mkdir(parents=True, exist_ok=True)

# 只创建裁剪图像目录
crop_dir = output_dir / "fashiongen_crops"
crop_dir.mkdir(exist_ok=True)

# 获取模型的类别名称（假设你的模型配置文件中有类别名称）
# 如果没有，你需要手动定义类别映射
try:
    # 尝试从模型获取类别名称
    class_names = model.names if hasattr(model, 'names') else {}
except:
    # 如果无法获取，使用默认的 ModaNet 类别
    class_names = {
        0: 'top', 1: 'bottom', 2: 'dress', 3: 'outer', 4: 'pants',
        5: 'skirt', 6: 'headwear', 7: 'eyewear', 8: 'bag', 9: 'shoes',
        10: 'belt', 11: 'socks', 12: 'tights', 13: 'gloves', 14: 'scarf'
    }

print(f"类别映射: {class_names}")

# 推理每张图像
for img_name in os.listdir(image_dir):
    if not img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
        continue

    img_path = os.path.join(image_dir, img_name)
    image = cv2.imread(img_path)
    if image is None:
        print(f"Failed to read image {img_path}")
        continue

    # 推理，返回结果对象
    results = model.predict(source=image, save=False, conf=0.3)

    detected_objects = 0

    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()  # 获取边界框
        confs = r.boxes.conf.cpu().numpy()  # 获取置信度
        classes = r.boxes.cls.cpu().numpy()  # 获取类别

        # 裁剪图像
        for i, (box, conf, cls_id) in enumerate(zip(boxes, confs, classes)):
            x1, y1, x2, y2 = map(int, box)
            crop = image[y1:y2, x1:x2]  # 从原始图像裁剪

            # 获取类别名称
            class_name = class_names.get(int(cls_id), f"class_{int(cls_id)}")

            # 保存裁剪图像，文件名包含类别信息
            crop_name = f"{Path(img_name).stem}_{class_name}_crop_{i}.jpg"
            crop_path = crop_dir / crop_name
            success = cv2.imwrite(str(crop_path), crop)
            if not success:
                print(f"Failed to write crop image {crop_path}")

            detected_objects += 1

    print(f"Processed {img_name}, detected {detected_objects} objects")
