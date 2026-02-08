# parts/models_loader.py
import torch
import argparse
import os
import numpy as np
from pathlib import Path
from hamer.configs import CACHE_DIR_HAMER
from hamer.models import download_models, load_hamer, DEFAULT_CHECKPOINT
from hamer.utils.renderer import Renderer
from vitpose_model import ViTPoseModel
from detectron2.utils.logger import setup_logger
from detectron2.config import LazyConfig
from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy
import hamer

# 引入原本的 global_vars 以获取 hands_rgb 等数据
#from global_vars import hands_rgb, INTRINSICS_HAMER_RENDERER, VIZ_DEMO
from global_config import state, INTRINSICS_HAMER_RENDERER
setup_logger()

# 配置参数
args = argparse.Namespace()
args.checkpoint = DEFAULT_CHECKPOINT
args.batch_size = 1
args.rescale_factor = 2.0
args.body_detector = 'vitdet'

# 初始化图像尺寸
print("Image width: ", state.IM_WIDTH)
print("Image height: ", state.IM_HEIGHT)

INTRINSICS_HAMER_RENDERER[0, 2] = state.IM_WIDTH / 2
INTRINSICS_HAMER_RENDERER[1, 2] = state.IM_HEIGHT / 2

# 下载并加载 HaMeR 模型
download_models(CACHE_DIR_HAMER)
model, model_cfg = load_hamer(args.checkpoint)

device = torch.device('cpu') # 根据需要改为 cuda
model = model.to(device)
model.eval()

# 加载 Detectron2 检测器
if args.body_detector == 'vitdet':
    cfg_path = Path(hamer.__file__).parent/'configs'/'cascade_mask_rcnn_vitdet_h_75ep.py'
    detectron2_cfg = LazyConfig.load(str(cfg_path))
    detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
    for i in range(3):
        detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
    detector = DefaultPredictor_Lazy(detectron2_cfg)
# ... regnety else block omitted for brevity but should be here ...

# 关键点检测器
cpm = ViTPoseModel(device)

# 渲染器
renderer = Renderer(model_cfg, faces=model.mano.faces, load_from_custom_file=True)

print("Existing focal length: ", model_cfg.EXTRA.FOCAL_LENGTH)