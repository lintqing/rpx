import os
import numpy as np
from dashscope import MultiModalConversation
import dashscope
import json
from typing import List, Dict, Tuple
import glob
from spatial_utils import SpatialMapper, extract_hand_node ,visualize_phase_movement# 导入我们的新工具
from global_config import INTRINSICS_REAL_CAMERA
import cv2
import re
import time

RECORDINGS_ROOT = "assets/recordings"

# 配置 API Key
dashscope.api_key = 'sk-a820ede4abc44f0cb4e5ae5dcd7066a9'

# ==========================================
# 1. 工具函数：处理 VLM 产生的时间幻觉
# ==========================================
def parse_time_to_seconds(time_val):
    """鲁棒地解析时间字符串，防止 0.00.00.00 错误"""
    if isinstance(time_val, (int, float)):
        return float(time_val)
    time_str = str(time_val).strip().lower()
    # 提取第一个合法的数字部分
    match = re.search(r'(\d+\.?\d*)', time_str)
    if match:
        return float(match.group(1))
    return 0.0
# ==========================================
# 2. 核心提取逻辑
# ==========================================
def extract_video_phases(video_path: str, api_key: str, fps: int = 10):
    """通过 CoT 提取视频操作阶段"""
    print(f"    - [VLM] 正在分析视频语义阶段 (Qwen)...")
    messages = [{
        'role': 'user',
        'content': [
            {'video': video_path, "fps": fps},
            {'text': '请提取视频动作阶段。请严格以JSON格式输出，键为"phases"，包含"phase_name", "start_time", "end_time"。注意：时间必须是纯数字秒数。'}
        ]
    }]
    try:
        response = MultiModalConversation.call(model='qwen3-vl-plus', messages=messages, api_key=api_key)
        if response.status_code == 200:
            content = response["output"]["choices"][0]["message"].content[0]["text"]
            content = content.replace("```json", "").replace("```", "").strip()
            return json.loads(content).get("phases", [])
    except Exception as e:
        print(f"      ❌ VLM 调用失败: {e}")
    return []

def get_frame_index(time: float, fps: int) -> int:
    return int(np.ceil(time * fps))

def get_all_npy_files(frame_dir: str) -> Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    all_files = {}
    pose_files = [f for f in os.listdir(frame_dir) if f.startswith("gripper_pose_") and f.endswith(".npy")]
    print(f"    - 发现 {len(pose_files)} 个位姿文件，开始加载...")
    for pose_file in pose_files:
        try:
            frame_idx = int(re.findall(r'\d+', pose_file)[0])
            action_path = os.path.join(frame_dir, f"gripper_actions_{frame_idx}.npy")
            hand_joints_path = os.path.join(frame_dir, f"hand_joints_kpts_{frame_idx}.npy")
            if os.path.exists(action_path) and os.path.exists(hand_joints_path):
                pose_data = np.load(os.path.join(frame_dir, pose_file))
                action_data = np.load(action_path)
                hand_joints_data = np.load(hand_joints_path)
                all_files[frame_idx] = (pose_data, action_data, hand_joints_data)
        except Exception as e:
            print(f"      ⚠️ 读取文件 {pose_file} 出错: {e}")
    return all_files

# ==========================================
# 3. 数据汇总保存逻辑 (汇总全量数据)
# ==========================================

def save_separate_data(all_frames: Dict, output_dir: str):
    """汇总并保存所有帧的位姿、动作和关节点数据"""
    try:
        os.makedirs(output_dir, exist_ok=True)
        # 1. 保存所有pose数据
        with open(os.path.join(output_dir, "all_poses.txt"), 'w', encoding='utf-8') as f:
            f.write("所有帧的姿势数据 (All Poses Data)\n" + "=" * 50 + "\n\n")
            for idx in sorted(all_frames.keys()):
                f.write(f"帧 {idx}: 姿势 (pose): {all_frames[idx][0].tolist()}\n")
        
        # 2. 保存所有action数据
        with open(os.path.join(output_dir, "all_actions.txt"), 'w', encoding='utf-8') as f:
            f.write("所有帧的动作数据 (All Actions Data)\n" + "=" * 50 + "\n\n")
            for idx in sorted(all_frames.keys()):
                action_val = all_frames[idx][1][-1] if len(all_frames[idx][1]) > 0 else 0.0
                f.write(f"帧 {idx}: 动作 (action): {action_val}\n")
        
        # 3. 保存所有hand_joints_kpts数据
        with open(os.path.join(output_dir, "all_hand_joints_kpts.txt"), 'w', encoding='utf-8') as f:
            f.write("所有帧的手部关节点数据 (All Hand Joints Keypoints Data)\n" + "=" * 50 + "\n\n")
            for idx in sorted(all_frames.keys()):
                f.write(f"帧 {idx}: 手部关节点: {all_frames[idx][2].tolist()}\n")

        # 4. 保存特定三个关键点数据 (Thumb Root, Thumb Tip, Index Tip)
        # 初始化映射器用于转换
        mapper = SpatialMapper(intrinsics=INTRINSICS_REAL_CAMERA)

        with open(os.path.join(output_dir, "three_keypoints.txt"), 'w', encoding='utf-8') as f:
            f.write("关键点轨迹 (Grid Coords 60x60x60): [Thumb Root (13), Thumb Tip (16), Index Tip (17)]\n" + "=" * 60 + "\n\n")
            for idx in sorted(all_frames.keys()):
                kpts = all_frames[idx][2] # shape (21, 3)
                if kpts.shape == (21, 3):
                    p_root = kpts[13] # Thumb MCP
                    p_thumb = kpts[16] # Thumb Tip
                    p_index = kpts[17] # Index Tip
                    
                    # 转换为网格坐标
                    g_root = mapper.world_to_grid(p_root)
                    g_thumb = mapper.world_to_grid(p_thumb)
                    g_index = mapper.world_to_grid(p_index)
                    
                    f.write(f"帧 {idx}:\n")
                    f.write(f"  Thumb Root (13): {list(g_root)}\n")
                    f.write(f"  Thumb Tip  (16): {list(g_thumb)}\n")
                    f.write(f"  Index Tip  (17): {list(g_index)}\n")
                    f.write("-" * 20 + "\n")

        print(f"    - [汇总成功] 全量数据已同步至 txt 文件夹")
    except Exception as e:
        print(f"    - [错误] 汇总数据失败: {e}")




def extract_interaction_grids(rgb_path, depth_path, api_key, mapper):
    """
    提取交互物体的网格坐标
    返回: (cup_grid, container_grid)
    * 修改为固定值，由用户决定 (Fixed Value Decision)
    """
    # 用户可在此处修改坐标值 [x, y, z]
    # 作用杯子坐标 (Object Cup)
    FIXED_CUP_GRID = [32, 25, 35] 
    
    # 目标容器坐标 (Target Container)
    FIXED_CONTAINER_GRID = [15, 45, 15]
    
    print(f"    - [Config] 使用固定作用杯子坐标: {FIXED_CUP_GRID}")
    print(f"    - [Config] 使用固定目标容器坐标: {FIXED_CONTAINER_GRID}")
    
    return FIXED_CUP_GRID, FIXED_CONTAINER_GRID

# ==========================================
# 4. 阶段数据保存逻辑 (空间锚点)
# ==========================================

def save_enriched_data(critical_frames: List, output_dir: str, cup_grid=None, container_grid=None):
    """保存阶段性的空间路径边界数据"""
    summary_path = os.path.join(output_dir, "phase_boundary_data.txt") 
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("动作阶段与空间边界数据 (Phase & 60x60x60 Grid Boundary)\n" + "="*50 + "\n")
        
        if cup_grid:
            f.write(f"OBJECT_CUP_GRID: {cup_grid}\n")
        if container_grid:
            f.write(f"TARGET_CONTAINER_GRID: {container_grid}\n")
        
        if cup_grid or container_grid:
            f.write("-" * 30 + "\n")
            
        for cf in critical_frames:
            
            f.write(f"阶段: {cf['phase_index']} - {cf['phase_name']}\n")
            f.write(f"  时间范围: 帧 {cf['start_frame']} -> 帧 {cf['end_frame']}\n")
            
            # Start Coords (Tuple of 3 vectors) - NOW GRIDS
            s_vecs = cf['start_kpts'] # [root, thumb, index] (Grid ints)
            f.write(f"  START_GRIDS:\n")
            f.write(f"    Root:  {list(s_vecs[0])}\n")
            f.write(f"    Thumb: {list(s_vecs[1])}\n")
            f.write(f"    Index: {list(s_vecs[2])}\n")
            
            # End Coords
            e_vecs = cf['end_kpts']
            f.write(f"  END_GRIDS:\n")
            f.write(f"    Root:  {list(e_vecs[0])}\n")
            f.write(f"    Thumb: {list(e_vecs[1])}\n")
            f.write(f"    Index: {list(e_vecs[2])}\n")
            f.write("-" * 30 + "\n")
    print(f"    - [汇总成功] 阶段空间边界已保存至 phase_boundary_data.txt")

# ==========================================
# 5. 主处理函数
# ==========================================

def process_single_sequence(seq_name: str, root_dir: str, api_key: str):
    seq_dir = os.path.join(root_dir, seq_name)
    local_video_path = os.path.join(seq_dir, "gripper_overlayed_video.mp4")
    npy_directory = os.path.join(seq_dir, "scene_files")
    output_directory = os.path.join(seq_dir, "txt")
    
    # Raw Data Paths
    rgb_path = os.path.join(seq_dir, "rgbs.npy")
    depth_path = os.path.join(seq_dir, "depths.npy")
    
    mapper = SpatialMapper(intrinsics=INTRINSICS_REAL_CAMERA) 
    print(f"\n{'='*20} 正在处理序列: {seq_name} {'='*20}")
    
    if not os.path.exists(local_video_path) or not os.path.exists(npy_directory):
        print(f"    [跳过] 必要文件缺失")
        return

    # 1. 物理数据读取与全量汇总
    all_frames = get_all_npy_files(npy_directory)
    if not all_frames: return
    save_separate_data(all_frames, output_directory) # 汇总 all_poses.txt 等

    # 2. 视频阶段分析
    fps = 10 
    phases = extract_video_phases(f"file://{os.path.abspath(local_video_path)}", api_key, fps=fps)
    
    # 2.1 提取交互物体坐标 (Cup & Container)
    cup_grid, container_grid = extract_interaction_grids(rgb_path, depth_path, api_key, mapper)
    
    # 3. 处理阶段锚点
    critical_frames = []
    
    for i, phase in enumerate(phases):
        try:
            # 鲁棒解析时间
            start_time = parse_time_to_seconds(phase["start_time"])
            end_time = parse_time_to_seconds(phase["end_time"])
            
            start_idx = get_frame_index(start_time, fps)
            end_idx = get_frame_index(end_time, fps)
            
            # 寻找最近的可用帧索引
            indices = sorted(all_frames.keys())
            start_idx = min(indices, key=lambda x: abs(x - start_idx))
            end_idx = min(indices, key=lambda x: abs(x - end_idx))
            
            # 提取 Start/End Hand Joints
            _, _, s_hnd = all_frames[start_idx]
            _, _, e_hnd = all_frames[end_idx]
            
            # Extract 3 Keypoints: 13, 16, 17 AND CONVERT TO GRID
            # s_hnd shape (21, 3)
            s_kpts = [
                mapper.world_to_grid(s_hnd[13]), 
                mapper.world_to_grid(s_hnd[16]), 
                mapper.world_to_grid(s_hnd[17])
            ]
            e_kpts = [
                mapper.world_to_grid(e_hnd[13]), 
                mapper.world_to_grid(e_hnd[16]), 
                mapper.world_to_grid(e_hnd[17])
            ]

            critical_frames.append({
                "phase_index": i, "phase_name": phase["phase_name"],
                "start_frame": start_idx, "end_frame": end_idx,
                "start_kpts": s_kpts, "end_kpts": e_kpts
            })
        except Exception as e:
            print(f"    - 处理阶段 {i} 出错: {e}")
            continue

    # 4. 保存阶段锚点数据
    save_enriched_data(critical_frames, output_directory, cup_grid, container_grid)
    print(f"    [完成] 序列 {seq_name} 处理完毕。")

if __name__ == "__main__":
    API_KEY = 'sk-a820ede4abc44f0cb4e5ae5dcd7066a9'
    all_sequences = sorted([d for d in os.listdir(RECORDINGS_ROOT) if d.startswith('seq_')])
    for seq_name in all_sequences:
        process_single_sequence(seq_name, RECORDINGS_ROOT, API_KEY)