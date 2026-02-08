import tqdm
import cv2
import numpy as np
import open3d as o3d
import copy
import os

# 本地引用
from util_functions import *

import global_config
from global_config import state, INTRINSICS_HAMER_RENDERER, INTRINSICS_REAL_CAMERA

# 引入 Pipeline
# 如果你想用新的0-4-8映射，记得在这里导入正确的函数名，例如 get_gripper_pose_from_frame_new
from parts.pipeline import get_gripper_pose_from_frame_new 
from parts.smoothing import GripperSmoother

# ================= 批量处理设置 =================
ASSETS_ROOT = "assets/recordings"  # 你的数据根目录
# 获取所有子文件夹
TASK_DIRS = [d for d in os.listdir(ASSETS_ROOT) if os.path.isdir(os.path.join(ASSETS_ROOT, d)) and d != "utils"]

print(f"Found tasks: {TASK_DIRS}")

# ================= 主循环 =================
for task_name in TASK_DIRS:
    print(f"\n========== Processing Task: {task_name} ==========")
    
    # 1. 构造当前任务的路径
    task_folder = os.path.join(ASSETS_ROOT, task_name)
    rgb_path = os.path.join(task_folder, "rgbs.npy")
    depth_path = os.path.join(task_folder, "depths.npy")

    if not os.path.exists(rgb_path) or not os.path.exists(depth_path):
        print(f"Skipping {task_name}: Data files not found.")
        continue
    
    # 2. 加载数据
    print("Loading data...")
    hands_rgb = np.load(rgb_path)
    hands_depth = np.load(depth_path)
    
    # 截取范围
    start_idx = global_config.START
    end_idx = min(global_config.END, len(hands_rgb))
    hands_rgb = hands_rgb[start_idx:end_idx]
    hands_depth = hands_depth[start_idx:end_idx]

    # 3. 【关键】更新全局状态 (State)
    state.IM_HEIGHT = hands_rgb[0].shape[0]
    state.IM_WIDTH = hands_rgb[0].shape[1]
    state.SCENE_FILES_FOLDER = os.path.join(task_folder, "scene_files")
    
    # 确保输出目录存在
    os.makedirs(state.SCENE_FILES_FOLDER, exist_ok=True)

    # 更新相机内参中心点
    INTRINSICS_HAMER_RENDERER[0, 2] = state.IM_WIDTH / 2
    INTRINSICS_HAMER_RENDERER[1, 2] = state.IM_HEIGHT / 2

    print(f"Image Size: {state.IM_WIDTH}x{state.IM_HEIGHT}")
    print(f"Saving to: {state.SCENE_FILES_FOLDER}")

    # 4. 初始化存储列表 (必须在循环内初始化，否则会把上一个视频的数据带过来)
    gripper_poses = []
    gripper_pcds = []
    live_image_pcds = []
    meshes = []
    joint_actions = [] # 如果需要
    gripper_actions = [] # 如果需要c

    # 5. 处理该视频的每一帧
    frame_step = global_config.FRAME_STEP
    #初始化滤波器
    smoother = GripperSmoother()
    
    # 【修正】注意这里的缩进，必须在 task 循环内部
    for idx in tqdm.tqdm(range(0, hands_rgb.shape[0], frame_step), desc=f"Processing {task_name}"):
        try:
            rgb = hands_rgb[idx]
            depth = hands_depth[idx] / 1000 # 假设单位转换

            ret = get_gripper_pose_from_frame_new(
                rgb, depth, 
                vizualize=global_config.VIZ, 
                press_task=global_config.IS_PRESS_TASK
            )
            
            # 解包返回值
            gripper_scaled_to_hand_pcd, live_image_pcd, gripper_pose, gripper_opening, hand_mesh, hand_kpts = ret
            
            #滤波处理
            smoth_pose,smoth_action = smoother.update(gripper_pose,gripper_opening, idx)

            # 数据收集
            #gripper_poses.append(gripper_pose)
            gripper_poses.append(smoth_pose)
            gripper_pcds.append(gripper_scaled_to_hand_pcd) # 用于后续投影
            live_image_pcds.append(live_image_pcd)
            #gripper_actions.append(gripper_opening)
            gripper_actions.append(smoth_action)
            meshes.append(hand_mesh)
            joint_actions.append(hand_kpts)

            # 保存中间文件 (使用 state.SCENE_FILES_FOLDER)
            o3d.io.write_point_cloud(f'{state.SCENE_FILES_FOLDER}/gripper_pcd_{idx}.ply', gripper_scaled_to_hand_pcd)
            print("\n gripper_pose 数据生成ing")
            np.save(f'{state.SCENE_FILES_FOLDER}/gripper_pose_{idx}.npy', gripper_pose)
            print("\n gripper_actions 数据生成ing")
            np.save(f'{state.SCENE_FILES_FOLDER}/gripper_actions_{idx}.npy', gripper_actions)
            print("\n hand_joints 数据生成ing")
            np.save(f'{state.SCENE_FILES_FOLDER}/hand_joints_kpts_{idx}.npy', hand_kpts)

        except Exception as e:
            print(f"Error processing frame {idx} in {task_name}: {e}")
            continue
    


    # 6. 后处理 (必须在 task 循环内部)
    if len(gripper_poses) == 0:
        print(f"No poses found for {task_name}, skipping post-processing.")
        continue

    print(f"Finished processing {task_name}. Interpolating and saving...")
    
    # 插值
    interpolated_gripper_poses = interpolate_pose_sequence(gripper_poses, frame_step)
    np.save(f'{state.SCENE_FILES_FOLDER}/interpolated_gripper_poses.npy', interpolated_gripper_poses)
    
    # 平滑处理
    interpolated_gripper_poses_xyz = []
    for pose in interpolated_gripper_poses:
        interpolated_gripper_poses_xyz.append(pose[:3, 3])
    interpolated_gripper_poses_xyz = np.array(interpolated_gripper_poses_xyz)

    window_size = frame_step
    interpolated_gripper_poses_filtered = copy.deepcopy(interpolated_gripper_poses_xyz)
    if len(interpolated_gripper_poses_xyz) > 2 * window_size:
        for i in range(window_size, interpolated_gripper_poses_xyz.shape[0] - window_size):
            interpolated_gripper_poses_filtered[i] = np.mean(interpolated_gripper_poses_xyz[i - window_size: i + window_size], axis=0)
    
    # 生成插值后的点云
    gripper_pcd0  = gripper_pcds[0]
    gripper_pose0 = gripper_poses[0]
    gripper_pose0_inv = np.linalg.inv(gripper_pose0)

    interpolated_gripper_pcds = []
    for pose in interpolated_gripper_poses:
        interpolated_gripper_pcds.append(copy.deepcopy(gripper_pcd0).transform(pose @ gripper_pose0_inv))

    # 投影到 2D 图像
    print("Projecting gripper pcds to 2D image...")
    gripper_projections = []
    # 使用 INTRINSICS_REAL_CAMERA (确保已从 global_config 导入)
    for i in range(len(gripper_pcds)):
        gripper_pcd_np = np.asarray(gripper_pcds[i].points)
        gripper_pcd_depth_im = point_cloud_to_depth_image(gripper_pcd_np,
                                                          INTRINSICS_REAL_CAMERA[0, 0],
                                                          INTRINSICS_REAL_CAMERA[1, 1],
                                                          INTRINSICS_REAL_CAMERA[0, 2],
                                                          INTRINSICS_REAL_CAMERA[1, 2],
                                                          width=state.IM_WIDTH,   # 【修正】使用 state
                                                          height=state.IM_HEIGHT) # 【修正】使用 state
        gripper_projections.append(gripper_pcd_depth_im)

    interpolated_gripper_projections = []
    for i in range(len(interpolated_gripper_pcds)):
        gripper_pcd_np = np.asarray(interpolated_gripper_pcds[i].points)
        gripper_pcd_depth_im = point_cloud_to_depth_image(gripper_pcd_np,
                                                          INTRINSICS_REAL_CAMERA[0, 0],
                                                          INTRINSICS_REAL_CAMERA[1, 1],
                                                          INTRINSICS_REAL_CAMERA[0, 2],
                                                          INTRINSICS_REAL_CAMERA[1, 2],
                                                          width=state.IM_WIDTH,   # 【修正】使用 state
                                                          height=state.IM_HEIGHT) # 【修正】使用 state
        interpolated_gripper_projections.append(gripper_pcd_depth_im)

    # 生成视频
    # 1. 先初始化 VideoWriter，不再创建 video_ims 列表
    video_path = os.path.join(state.SCENE_FILES_FOLDER, "..", "gripper_overlayed_video.mp4")
    # 获取尺寸 (假设 hands_rgb 不为空)
    video_height, video_width = hands_rgb[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # 或者 'DIVX'
    out = cv2.VideoWriter(video_path, fourcc, 10, (video_width, video_height))

    cnt = 0
    cnt_gripper_proj = 0

    # 2. 边生成边写入，不占用额外内存
    # 使用 tqdm 显示进度
    for i in tqdm.tqdm(range(0, hands_rgb.shape[0]), desc="Saving Video"):
        # 安全检查：防止索引越界
        if cnt >= len(interpolated_gripper_projections) or cnt_gripper_proj >= len(gripper_projections):
            break
            
        gripper_proj = interpolated_gripper_projections[cnt][:, :, np.newaxis] 
        gripper_proj[gripper_proj > 0] = 1
        
        # 图像融合
        # 注意：hands_rgb[i] 可能是 float 也可能是 uint8，这里最好统一一下
        bg_img = hands_rgb[i] / 255.0 if hands_rgb[i].max() > 1.0 else hands_rgb[i]
        
        im = .5 * gripper_proj + 1 * bg_img + 0.5 * gripper_projections[cnt_gripper_proj][:, :, np.newaxis].repeat(3, axis=2) * np.array([0, 1, 0])
        
        # 限制数值范围并转为 uint8
        im = np.clip(im, 0, 1) * 255
        im = im.astype(np.uint8)
        
        # RGB 转 BGR (OpenCV 需要)
        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        
        # 3. 立即写入硬盘，释放内存
        out.write(im)
        
        # 更新计数器
        if cnt % frame_step == 0:
            cnt_gripper_proj += 1
        cnt += 1

    out.release()
    print(f"Video saved to {video_path}")

    # 4. 强制垃圾回收，释放显存和内存，为下一个视频做准备
    import gc
    import torch
    del interpolated_gripper_projections
    del gripper_projections
    del gripper_pcds
    gc.collect()
    torch.cuda.empty_cache()
print("All tasks processed!")