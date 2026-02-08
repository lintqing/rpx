# parts/pipeline.py
import numpy as np
import open3d as o3d
import copy
from global_config import MANO_HAND_IDS, DISTANCE_BETWEEN_GRIPPERS_FINGERS
from util_functions import np_to_o3d, rotation_matrix # 假设

# 引入子模块
from parts.detection import get_hand_and_rendered_depth, get_hand_mask_from_detectron
from parts.keypoints import get_joints_of_hand_mesh, get_hand_keypoints_from_mediapipe, get_hand_keypoints_from_mano_model
from parts.geometry import align_gripper_palm,get_hand_pcd_in_scene_from_rendered_model, mesh_and_joints_to_world_metric_space, align_gripper_to_hand, align_gripper_with_hand_fingers, align_hand_to_gripper_press, align_gripper_with_triangle

#从单帧图像计算夹爪位姿
def get_gripper_pose_from_frame(rgb, depth, use_mediapipe_for_hand_kpts=False, vizualize=False, press_task=False, scale_depth_image=False):
    depth = np.array(depth).astype(np.float32)
    rgb_im, rgb_hand_only, rend_depth_front_view, det_out, hamer_output, hand_mesh_params, all_mesh_params = get_hand_and_rendered_depth(rgb)
    depth_im = depth
    human_mask = get_hand_mask_from_detectron(det_out)
    all_meshes, all_cameras, all_vertices = all_mesh_params[0], all_mesh_params[1], all_mesh_params[2]
    rgb_hand_only = rgb_hand_only[:,:,:3]
    joint_meshes, joints_coords = get_joints_of_hand_mesh(copy.deepcopy(all_meshes[0]), copy.deepcopy(all_vertices[0]), copy.deepcopy(all_cameras[0]))
    if use_mediapipe_for_hand_kpts: 
        middle_x, middle_y, index_finger_1_x, index_finger_1_y, index_finger_2_x, index_finger_2_y, index_finger_3_x, index_finger_3_y, thumb_1_x, thumb_1_y, thumb_2_x, thumb_2_y, thumb_3_x, thumb_3_y = get_hand_keypoints_from_mediapipe(rgb_im, viz_keypoints=vizualize)
    else:
        middle_x, middle_y, index_finger_1_x, index_finger_1_y, index_finger_2_x, index_finger_2_y, index_finger_3_x, index_finger_3_y, thumb_1_x, thumb_1_y, thumb_2_x, thumb_2_y, thumb_3_x, thumb_3_y = get_hand_keypoints_from_mano_model(joints_coords, rgb_im=rgb_im, vizualize=vizualize)
    scaling_factor = 1
    if scale_depth_image:
        # generally no need to use this
        d_val = -1
        base_value_finger = 'index' # index, middle, thumb
        base_val = copy.deepcopy(rend_depth_front_view[index_finger_1_y, index_finger_1_x])
        idx_x, idx_y = index_finger_1_x, index_finger_1_y
        if base_val == 0 or depth_im[idx_y, idx_x] == 0:
            base_value_finger = 'middle' # index, middle, thumb
            base_val = copy.deepcopy(rend_depth_front_view[middle_y, middle_x])
            idx_x, idx_y = middle_x, middle_y
            if base_val == 0 or depth_im[idx_y, idx_x] == 0:
                base_value_finger = 'thumb'
                base_val = copy.deepcopy(rend_depth_front_view[thumb_1_y, thumb_1_x])
                idx_x, idx_y = thumb_1_x, thumb_1_y
        d_val = copy.deepcopy(depth_im[idx_y, idx_x])
        percentage_rend_depth_front_view = rend_depth_front_view / base_val
        scaled_depth_image = percentage_rend_depth_front_view * d_val
        scaling_factor = d_val / base_val
        print(f'Scaling factor {base_value_finger}: {scaling_factor}')
    else:
        scaled_depth_image = copy.deepcopy(rend_depth_front_view)
    hand_point_cloud, hand_point_cloud_full, image_point_cloud, hand_pcd_as_o3d, live_image_pcd_as_o3d, T_mesh_to_live = get_hand_pcd_in_scene_from_rendered_model(rgb_im, 
                                                                                                                                                depth_im, 
                                                                                                                                                rend_depth_front_view, 
                                                                                                                                                scaled_depth_image, 
                                                                                                                                                human_mask, 
                                                                                                                                                vizualize=vizualize) 
    mesh_live_np, joints_live, hand_mesh_in_live_metric_space = mesh_and_joints_to_world_metric_space(copy.deepcopy(all_meshes[0]), 
                                                               copy.deepcopy(joints_coords), 
                                                               T_mesh_to_live, 
                                                               scaling_factor, 
                                                               live_image_pcd=live_image_pcd_as_o3d, 
                                                               vizualize=vizualize)   
    gripper_pcd = np.load('assets/utils/gripper_point_cloud_dense.npy')
    gripper_pcd = gripper_pcd / 1000 # scale down by 1000
    gripper_pcd_as_o3d = o3d.geometry.PointCloud()
    gripper_pcd_as_o3d.points = o3d.utility.Vector3dVector(gripper_pcd)
    gripper_aligned_to_hand_pcd_as_o3d = align_gripper_to_hand(hand_point_cloud, 
                                                            hand_pcd_as_o3d, 
                                                            gripper_pcd, 
                                                            gripper_pcd_as_o3d, 
                                                            vizualize=vizualize)

    key_fingers_points = np.array([joints_live[MANO_HAND_IDS["index_tip"]],
                                joints_live[MANO_HAND_IDS["index_dip"]],
                                joints_live[MANO_HAND_IDS["index_pip"]],
                                joints_live[MANO_HAND_IDS["thumb_tip"]], 
                                joints_live[MANO_HAND_IDS["thumb_dip"]], 
                                joints_live[MANO_HAND_IDS["thumb_pip"]], 
                                joints_live[MANO_HAND_IDS["wrist"]],
                                (joints_live[MANO_HAND_IDS["index_mcp"]] + joints_live[MANO_HAND_IDS["thumb_dip"]]) / 2])
    key_fingers_points = key_fingers_points.reshape(-1, 3)
    gripper_scaled_to_hand_pcd = copy.deepcopy(gripper_aligned_to_hand_pcd_as_o3d) # replace, so no scale applied. Comment out if scaling - although likely you don't need to
    gripper_scaled_to_hand_pcd.paint_uniform_color([0, 0, 1])
    bias_T  = np.eye(4)
    bias_T[2, 3] = 0.01 # NOTE: This is explicit to the robotiq CAD model gripper we had
    if not press_task:
        gripper_scaled_to_hand_pcd, gripper_pose, distance_hand_fingers = align_gripper_with_hand_fingers(gripper_scaled_to_hand_pcd, 
                                                                    np_to_o3d(mesh_live_np), 
                                                                    key_fingers_points, 
                                                                    gripper_aligned_to_hand_pcd_as_o3d,
                                                                    gripper_pcd_as_o3d, 
                                                                    use_only_thumb_keypoints=False,
                                                                    use_only_index_keypoints=False,
                                                                    rescale_gripper_to_hand_opening=False,
                                                                    rescale_hand_to_gripper_opening=False,
                                                                    bias_transformation=bias_T,
                                                                    vizualize=vizualize)
        gripper_opening_percent = np.min([distance_hand_fingers / DISTANCE_BETWEEN_GRIPPERS_FINGERS, 1])
    else:
        gripper_scaled_to_hand_pcd, gripper_pose = align_hand_to_gripper_press(gripper_scaled_to_hand_pcd, 
                                                                                np.array([key_fingers_points[0], key_fingers_points[1], key_fingers_points[2]]), 
                                                                                vizualize=vizualize,
                                                                                bias_transformation=bias_T)
        gripper_opening_percent = 0
        
    return gripper_scaled_to_hand_pcd, live_image_pcd_as_o3d, gripper_pose, gripper_opening_percent, hand_mesh_in_live_metric_space, joints_live

#从单帧图像计算夹爪位姿
def get_gripper_pose_from_frame_new(rgb, depth, use_mediapipe_for_hand_kpts=False, vizualize=False, press_task=False, scale_depth_image=False):
    # ---------------- 1. 基础数据准备 (保持不变) ----------------
    depth = np.array(depth).astype(np.float32)
    rgb_im, rgb_hand_only, rend_depth_front_view, det_out, hamer_output, hand_mesh_params, all_mesh_params = get_hand_and_rendered_depth(rgb)
    depth_im = depth
    human_mask = get_hand_mask_from_detectron(det_out)
    all_meshes, all_cameras, all_vertices = all_mesh_params[0], all_mesh_params[1], all_mesh_params[2]
    rgb_hand_only = rgb_hand_only[:,:,:3]
    joint_meshes, joints_coords = get_joints_of_hand_mesh(copy.deepcopy(all_meshes[0]), copy.deepcopy(all_vertices[0]), copy.deepcopy(all_cameras[0]))
    if use_mediapipe_for_hand_kpts: 
        middle_x, middle_y, index_finger_1_x, index_finger_1_y, index_finger_2_x, index_finger_2_y, index_finger_3_x, index_finger_3_y, thumb_1_x, thumb_1_y, thumb_2_x, thumb_2_y, thumb_3_x, thumb_3_y = get_hand_keypoints_from_mediapipe(rgb_im, viz_keypoints=vizualize)
    else:
        middle_x, middle_y, index_finger_1_x, index_finger_1_y, index_finger_2_x, index_finger_2_y, index_finger_3_x, index_finger_3_y, thumb_1_x, thumb_1_y, thumb_2_x, thumb_2_y, thumb_3_x, thumb_3_y = get_hand_keypoints_from_mano_model(joints_coords, rgb_im=rgb_im, vizualize=vizualize)
    scaling_factor = 1
    if scale_depth_image:
        # generally no need to use this
        d_val = -1
        base_value_finger = 'index' # index, middle, thumb
        base_val = copy.deepcopy(rend_depth_front_view[index_finger_1_y, index_finger_1_x])
        idx_x, idx_y = index_finger_1_x, index_finger_1_y
        if base_val == 0 or depth_im[idx_y, idx_x] == 0:
            base_value_finger = 'middle' # index, middle, thumb
            base_val = copy.deepcopy(rend_depth_front_view[middle_y, middle_x])
            idx_x, idx_y = middle_x, middle_y
            if base_val == 0 or depth_im[idx_y, idx_x] == 0:
                base_value_finger = 'thumb'
                base_val = copy.deepcopy(rend_depth_front_view[thumb_1_y, thumb_1_x])
                idx_x, idx_y = thumb_1_x, thumb_1_y
        d_val = copy.deepcopy(depth_im[idx_y, idx_x])
        percentage_rend_depth_front_view = rend_depth_front_view / base_val
        scaled_depth_image = percentage_rend_depth_front_view * d_val
        scaling_factor = d_val / base_val
        print(f'Scaling factor {base_value_finger}: {scaling_factor}')
    else:
        scaled_depth_image = copy.deepcopy(rend_depth_front_view)
    
    hand_point_cloud, hand_point_cloud_full, image_point_cloud, hand_pcd_as_o3d, live_image_pcd_as_o3d, T_mesh_to_live = get_hand_pcd_in_scene_from_rendered_model(
        rgb_im, depth_im, rend_depth_front_view, scaled_depth_image, human_mask, vizualize=vizualize
    ) 
    
    mesh_live_np, joints_live, hand_mesh_in_live_metric_space = mesh_and_joints_to_world_metric_space(
        copy.deepcopy(all_meshes[0]), 
        copy.deepcopy(joints_coords), 
        T_mesh_to_live, 
        scaling_factor, 
        live_image_pcd=live_image_pcd_as_o3d, 
        vizualize=vizualize
    )    
    # ---------------- 3. 加载夹爪模型 (提到前面来！) ----------------
    # 必须先加载，才能传给函数
    gripper_pcd = np.load('assets/utils/gripper_point_cloud_dense.npy')
    gripper_pcd = gripper_pcd / 1000 # scale down by 1000
    gripper_pcd_as_o3d = o3d.geometry.PointCloud()
    gripper_pcd_as_o3d.points = o3d.utility.Vector3dVector(gripper_pcd)

    
    # 调用 geometry.py 中的新函数 align_gripper_with_triangle
    gripper_scaled_to_hand_pcd, gripper_pose, gripper_opening_percent = align_gripper_with_triangle(
        gripper_pcd_as_o3d, # 原始夹爪
        joints_live,     # 21 个手部关键点
        vizualize=vizualize
    )
    
    # 给夹爪上色，方便可视化
    gripper_scaled_to_hand_pcd.paint_uniform_color([0, 0, 1])

    # ---------------- 5. 返回结果 ----------------
    
    return gripper_scaled_to_hand_pcd, live_image_pcd_as_o3d, gripper_pose, gripper_opening_percent, hand_mesh_in_live_metric_space, joints_live