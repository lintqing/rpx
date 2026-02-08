# global_vars.py
import numpy as np
import os

# ================= 1. 固定的配置常量 (不随视频变化) =================
VIZ = False 
IS_PRESS_TASK = False 
LOAD_SCENE_DATA_FOR_PROCESSING = False 
VIZ_DEMO = False 
START = 0 
END = 1000000
FRAME_STEP = 5 
MODEL_MANO_PATH = '_DATA/data/mano'

# 相机内参 (如果有多个相机，可以在这里定义好字典，稍后按需取)
INTRINSICS_D455 = np.load("assets/utils/intrinsics_rgb_d455.npy")
# INTRINSICS_HEAD = np.load("assets/utils/head_cam_intrinsic_matrix_aligned_depth.npy")

# 默认使用 D455，稍后可以在 main.py 中修改
INTRINSICS_REAL_CAMERA = INTRINSICS_D455 

# 渲染器内参模板
INTRINSICS_HAMER_RENDERER = np.eye(4)
INTRINSICS_HAMER_RENDERER[0, 0] = 2295.0
INTRINSICS_HAMER_RENDERER[1, 1] = 2295.0
INTRINSICS_HAMER_RENDERER[0, 2] = 320.0 # 默认值，稍后会更新
INTRINSICS_HAMER_RENDERER[1, 2] = 240.0 # 默认值，稍后会更新

T_OPENGL_TO_OPENCV = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
HUMAN_HAND_COLOR = (0.999, 0.6745, 0.4117)
MANO_HAND_IDS = {"wrist": 0, "index_mcp": 1, "index_pip": 2, 
                 "index_dip": 3, "middle_mcp": 4, "middle_pip": 5, 
                 "middle_dip": 6, "pinkie_mcp": 7, "pinkie_pip": 8, 
                 "pinkie_dip": 9, "ring_mcp": 10, "ring_pip": 11, 
                 "ring_dip": 12, "thumb_mcp": 13, "thumb_pip": 14, 
                 "thumb_dip": 15, "thumb_tip": 16, "index_tip": 17, 
                 "middle_tip": 18, "ring_tip": 19, "pinky_tip": 20}
DISTANCE_BETWEEN_GRIPPERS_FINGERS = 0.08507

# ================= 2. 可变的状态容器 =================
class ProcessingState:
    def __init__(self):
        self.IM_WIDTH = 1920  # 默认值
        self.IM_HEIGHT = 1080 # 默认值
        self.SCENE_FILES_FOLDER = "output_debug" # 默认输出路径

# 创建一个全局实例，其他文件都要引用这个实例
state = ProcessingState()