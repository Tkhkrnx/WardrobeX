import random
from torch.utils.data import Dataset

class PromptDataset(Dataset):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        # 目标生成约1000条示例，结合多种模板提升多样性
        colors = ["红色", "蓝色", "黑色", "白色", "米色", "驼色", "灰色", "粉色", "绿色", "黄色", "橙色", "紫色"]
        items = {
            "连衣裙": ["吊带连衣裙", "蕾丝连衣裙", "雪纺连衣裙", "碎花连衣裙", "针织连衣裙"],
            "风衣": ["短款风衣", "长款风衣", "驼色风衣", "防风风衣"],
            "大衣": ["毛呢大衣", "羊绒大衣", "双排扣大衣", "皮革大衣"],
            "衬衫": ["白衬衫", "丝绸衬衫", "牛津纺衬衫", "条纹衬衫", "印花衬衫"],
            "裤子": ["高腰牛仔裤", "阔腿裤", "西装裤", "运动裤", "九分裤"],
            "半身裙": ["百褶裙", "A字裙", "皮质短裙", "不规则半身裙"],
            "鞋子": ["小白鞋", "马丁靴", "尖头高跟鞋", "短靴", "帆布鞋"],
            "包袋": ["链条包", "托特包", "单肩包", "斜挎包", "手拿包"],
            "配饰": ["丝巾", "腰带", "帽子", "项链", "耳环"]
        }
        scenes = [
            "春季出游", "夏日约会", "秋季通勤", "冬季聚会",
            "晚宴场合", "日常休闲", "商务会议", "校园风",
            "旅行度假", "节日派对", "看展活动", "周末逛街"
        ]
        # 按类别定义输出模板，强调常见互补单品或搭配要素
        output_templates = {
            "连衣裙": [
                lambda c: f"推荐搭配一件简约外套，突出{c}的优雅；配以细高跟鞋和手拿包，整体更显女人味。",
                lambda c: f"可搭配马丁靴和皮质夹克，营造酷感街头风；配饰选择金属饰品更添个性。"
            ],
            "风衣": [
                lambda c: f"风衣内搭高领针织衫和九分裤，配同色系短靴，简洁有型；配饰可选丝巾点缀颈部。",
                lambda c: f"可裹腰带束出腰线，搭配小白鞋和迷你挎包，优雅又休闲。"
            ],
            "大衣": [
                lambda c: f"大衣下身配直筒西裤，脚踩尖头高跟鞋，凸显高级感；配以简约手袋和围巾。",
                lambda c: f"可搭配宽松牛仔裤和马丁靴，形成硬朗对比；配饰选择皮革腰带增加层次。"
            ],
            "衬衫": [
                lambda c: f"白衬衫下搭高腰铅笔裙，脚踩细高跟，优雅知性；配饰选珍珠耳环提升精致度。",
                lambda c: f"可搭配阔腿裤和小白鞋，休闲又平衡廓形；添加细腰带突出腰线。"
            ],
            "裤子": [
                lambda c: f"高腰{c}配修身短款上衣，拉长腿部线条；搭配尖头皮鞋或短靴更显干练。",
                lambda c: f"可配oversize卫衣和运动鞋，舒适兼时尚；配饰选棒球帽或腰包点亮造型。"
            ],
            "半身裙": [
                lambda c: f"A字裙配贴身针织衫和过膝靴，显高显瘦且优雅；配以细链条包。",
                lambda c: f"可选薄毛衣和短靴组合，增加层次；耳饰选择圆形大圈更显活力。"
            ],
            "鞋子": [
                lambda c: f"{c}搭配修身牛仔裤和长款大衣，复古又摩登；配饰可选同色系手袋。",
                lambda c: f"可与连衣裙组合，内搭皮夹克，打造甜酷混搭风；配以复古墨镜。"
            ],
            "包袋": [
                lambda c: f"{c}搭配简约连衣裙或牛仔装，平衡廓形；配以同色系鞋履提升整体感。",
                lambda c: f"可与风衣或大衣搭配，选择中号容量；配饰点缀项链或丝巾增亮。"
            ],
            "配饰": [
                lambda c: f"{c}系于颈部或绑于包带，增加造型亮点；可搭配简约衣着，让配饰更突出。",
                lambda c: f"可与同色系鞋包呼应，或作为对比色，制造层次感；保持其他配饰简洁。"
            ]
        }
        # 多种输入模板
        input_templates = [
            lambda col, itm: f"这是一件{col}{itm}，有什么搭配建议？",
            lambda col, itm: f"我想穿{col}{itm}，请推荐合适的搭配。",
            lambda col, itm: f"{col}{itm}适合搭配哪些单品？",
            lambda col, itm: f"如何用{col}{itm}来打造时尚造型？"
        ]

        self.data = []
        # 单品问答示例
        for cat, item_list in items.items():
            for item in item_list:
                for color in colors:
                    if len(self.data) >= 800:
                        break
                    inp = random.choice(input_templates)(color, item)
                    out = random.choice(output_templates[cat])(color)
                    self.data.append({"input": inp, "output": out})
                if len(self.data) >= 800:
                    break
            if len(self.data) >= 800:
                break
        # 场景问答示例 (补足至1000)
        while len(self.data) < 1000:
            scene = random.choice(scenes)
            cat = random.choice(list(items.keys()))
            item = random.choice(items[cat])
            color = random.choice(colors)
            # 场景输入模板
            scene_inputs = [
                lambda sc, col, itm: f"{sc}穿{col}{itm}，有什么搭配建议？",
                lambda sc, col, itm: f"想在{sc}穿{col}{itm}，如何搭配？",
                lambda sc, col, itm: f"{sc}适合{col}{itm}吗？请推荐搭配。"
            ]
            inp = random.choice(scene_inputs)(scene, color, item)
            out = random.choice(output_templates[cat])(color)
            self.data.append({"input": inp, "output": out})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
