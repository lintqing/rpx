# parts/keypoints.py
import mediapipe as mp
import cv2 as cv
import numpy as np
import copy
import matplotlib.pyplot as plt
import torch
import pickle
import os
import os.path as osp
import trimesh
from mano import lbs
from mano.joints_info import TIP_IDS
from mano.utils import Mesh, Struct, colors, to_np, to_tensor
from global_config import T_OPENGL_TO_OPENCV, MODEL_MANO_PATH, MANO_HAND_IDS

# 导入配置
from parts.models_loader import INTRINSICS_HAMER_RENDERER
from util_functions import *

#检测手部21个关键点，返回手指的关键点坐标
def get_hand_keypoints_from_mediapipe(rgb_im, viz_keypoints=False):
    # # Initialize MediaPipe Hands.
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)
    color_image = copy.deepcopy((rgb_im * 255.0).astype(np.uint8))
    if viz_keypoints:
        plt.imshow(rgb_im)
        plt.axis('off')
        plt.show()
    # Process the image to find hand landmarks.
    results = hands.process(color_image)
    d_val = -1
    # Draw the hand landmarks on the image.
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                # Draw a circle on the image.
                cx, cy = int(landmark.x * color_image.shape[1]), int(landmark.y * color_image.shape[0])
                cv.circle(color_image, (cx, cy), 5, (0, 255, 0), -1)
    mp_drawing = mp.solutions.drawing_utils
    image = color_image
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            # Example: Get the tip of the index finger
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            # print(f"Index Finger Tip Coordinates: (x: {index_tip.x}, y: {index_tip.y}, z: {index_tip.z})")
            index_x, index_y = int(index_tip.x * image.shape[1]), int(index_tip.y * image.shape[0])
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            # print(f"Thumb Tip Coordinates: (x: {thumb_tip.x}, y: {thumb_tip.y}, z: {thumb_tip.z})")
            thumb_x, thumb_y = int(thumb_tip.x * image.shape[1]), int(thumb_tip.y * image.shape[0])
            middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            # print(f"Middle Finger Tip Coordinates: (x: {middle_tip.x}, y: {middle_tip.y}, z: {middle_tip.z})")
            middle_x, middle_y = int(middle_tip.x * image.shape[1]), int(middle_tip.y * image.shape[0])
    # Extract a hand mesh from the hand landmarks and display it.
    hand_mesh = mp_hands.HAND_CONNECTIONS
    hand_mesh = [list(pair) for pair in hand_mesh]
    # Draw the hand mesh on the image.
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for connection in hand_mesh:
                # Get the coordinates of the two points.
                start = (int(hand_landmarks.landmark[connection[0]].x * image.shape[1]),
                        int(hand_landmarks.landmark[connection[0]].y * image.shape[0]))
                end = (int(hand_landmarks.landmark[connection[1]].x * image.shape[1]),
                    int(hand_landmarks.landmark[connection[1]].y * image.shape[0]))
                # Draw a line connecting the two points.
                cv.line(image, start, end, (0, 255, 0), 2)
        
    # Draw a cirlce on the pixel of the index finger tip using matplotlib.
    thumb_cmc = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC]
    thumb_cmc_x, thumb_cmc_y = int(thumb_cmc.x * image.shape[1]), int(thumb_cmc.y * image.shape[0])

    thumb_IP = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
    thumb_IP_x, thumb_IP_y = int(thumb_IP.x * image.shape[1]), int(thumb_IP.y * image.shape[0])

    thumb_MCP = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP]
    thumb_MCP_x, thumb_MCP_y = int(thumb_MCP.x * image.shape[1]), int(thumb_MCP.y * image.shape[0])

    index_mc = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    index_mc_x, index_mc_y = int(index_mc.x * image.shape[1]), int(index_mc.y * image.shape[0])

    index_pip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]
    index_pip_x, index_pip_y = int(index_pip.x * image.shape[1]), int(index_pip.y * image.shape[0])

    index_dip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP]
    index_dip_x, index_dip_y = int(index_dip.x * image.shape[1]), int(index_dip.y * image.shape[0])
    
    if viz_keypoints:
        plt.imshow(rgb_im)
        plt.scatter(index_x, index_y, color='blue', s=10)
        plt.scatter(index_dip_x, index_dip_y, color='orange', s=10)

        plt.scatter(index_pip_x, index_pip_y, color='purple', s=10)
        plt.scatter(thumb_x, thumb_y, color='blue', s=10)
        plt.scatter(thumb_IP_x, thumb_IP_y, color='orange', s=10)
        plt.scatter(thumb_MCP_x, thumb_MCP_y, color='purple', s=10)

    index_finger_1_x, index_finger_1_y = index_x, index_y
    index_finger_2_x, index_finger_2_y = index_dip_x, index_dip_y
    index_finger_3_x, index_finger_3_y = index_pip_x, index_pip_y

    thumb_1_x, thumb_1_y = thumb_x, thumb_y
    thumb_2_x, thumb_2_y = thumb_IP_x, thumb_IP_y
    thumb_3_x, thumb_3_y = thumb_MCP_x, thumb_MCP_y
    return middle_x, middle_y, index_finger_1_x, index_finger_1_y, index_finger_2_x, index_finger_2_y, index_finger_3_x, index_finger_3_y, thumb_1_x, thumb_1_y, thumb_2_x, thumb_2_y, thumb_3_x, thumb_3_y


#从MANO手部网格顶点计算关节位置
def get_mano_hand_joints_from_vertices(vertices, camera_translation, model_path = MODEL_MANO_PATH):
    is_rhand = True
    ext='pkl'
    data_struct = None
    if data_struct is None:
            # Load the model
            if osp.isdir(model_path):
                model_fn = 'MANO_{}.{ext}'.format('RIGHT' if is_rhand else 'LEFT', ext=ext)
                mano_path = os.path.join(model_path, model_fn)
            else:
                mano_path = model_path
                is_rhand = True if 'RIGHT' in os.path.basename(model_path) else False
            assert osp.exists(mano_path), 'Path {} does not exist!'.format(
                mano_path)
            if ext == 'pkl':
                with open(mano_path, 'rb') as mano_file:
                    model_data = pickle.load(mano_file, encoding='latin1')
            elif ext == 'npz':
                model_data = np.load(mano_path, allow_pickle=True)
            else:
                raise ValueError('Unknown extension: {}'.format(ext))
            data_struct = Struct(**model_data)
            
    def add_joints(vertices,joints, joint_ids = None):
      tip_ids = TIP_IDS['mano']
      dev = vertices.device
      if joint_ids is None:
          joint_ids = to_tensor(list(tip_ids.values()),
                                dtype=torch.long).to(dev)
      extra_joints = torch.index_select(vertices, 1, joint_ids)
      joints = torch.cat([joints, extra_joints], dim=1)
      return joints
    
    data_struct.J_regressor = torch.from_numpy(data_struct.J_regressor.todense()).float()
    joints_predicted = lbs.vertices2joints(data_struct.J_regressor, torch.tensor(vertices).unsqueeze(0))
    joints_predicted = add_joints(torch.tensor(vertices).unsqueeze(0), joints_predicted)

    hand_camera_translation_torch = torch.tensor(camera_translation).unsqueeze(0).unsqueeze(0)
    # the below comes from renderer.py (ln: 400) to match how the actual hand mesh is
    # transformed before rendering
    rot_axis=[1,0,0]
    rot_angle= 180
    rot = trimesh.transformations.rotation_matrix(np.radians(rot_angle), rot_axis)
    joints_predicted_rotated = joints_predicted @ rot[:3, :3].T
    joints_predicted_translated = joints_predicted_rotated - hand_camera_translation_torch 

    return joints_predicted_translated

#将关节点坐标转化为可视化的球体网格
def joints_np_joint_meshes(joints, radius=.005, vc=colors['green']):
    joints = to_np(joints)
    if joints.ndim <3:
        joints = joints.reshape(1,-1,3)
    meshes = []
    for j in joints:
        joint_mesh = Mesh(vertices=j, radius=radius, vc=vc)
        meshes.append(joint_mesh)
    return  meshes

#获取手部网格的关节
def get_joints_of_hand_mesh(mesh, vertices, camera_translation):
    joints = get_mano_hand_joints_from_vertices(vertices, camera_translation)
    joint_meshes = joints_np_joint_meshes(joints)
    joint_meshes[0].vertices = joint_meshes[0].vertices + mesh.vertices.mean(0) - joint_meshes[0].vertices.mean(0) # align joints with the hand mesh
    joints = joints.squeeze(0)
    joints = joints + mesh.vertices.mean(0) - torch.mean(joints, axis=0) # align joints with the hand mesh
    return joint_meshes, joints

#从MANO模型获取手部关键点
def get_hand_keypoints_from_mano_model(joints, rgb_im=None, vizualize=False):

    joint_mesh = np.linalg.inv(T_OPENGL_TO_OPENCV) @ np.vstack((joints.T, np.ones((1, joints.shape[0]))))
    joint_mesh = joint_mesh[:3, :].T
    jm_X, jm_Y, jm_Z = joint_mesh[:, 0], joint_mesh[:, 1], joint_mesh[:, 2]
    x_pixel = (jm_X * INTRINSICS_HAMER_RENDERER[0, 0] / jm_Z) + INTRINSICS_HAMER_RENDERER[0, 2]
    y_pixel = (jm_Y * INTRINSICS_HAMER_RENDERER[1, 1] / jm_Z) + INTRINSICS_HAMER_RENDERER[1, 2]
    joint_mesh_2d = np.vstack((x_pixel, y_pixel)).T
    joint_mesh_2d = joint_mesh_2d.astype(int)
    middle_x, middle_y = joint_mesh_2d[MANO_HAND_IDS["middle_tip"]]
    index_finger_1_x, index_finger_1_y = joint_mesh_2d[MANO_HAND_IDS["index_tip"]]
    index_finger_2_x, index_finger_2_y = joint_mesh_2d[MANO_HAND_IDS["index_pip"]] 
    index_finger_3_x, index_finger_3_y = joint_mesh_2d[MANO_HAND_IDS["index_dip"]]
    thumb_1_x, thumb_1_y = joint_mesh_2d[MANO_HAND_IDS["thumb_tip"]]
    thumb_2_x, thumb_2_y = joint_mesh_2d[MANO_HAND_IDS["thumb_pip"]]
    thumb_3_x, thumb_3_y = joint_mesh_2d[MANO_HAND_IDS["thumb_dip"]]
    if vizualize:
        assert rgb_im is not None, "Must provide RGB to vizualize"

        joint_mesh_depth_im = point_cloud_to_depth_image(copy.deepcopy(joint_mesh), 
                                                      fx=INTRINSICS_HAMER_RENDERER[0, 0],
                                                      fy=INTRINSICS_HAMER_RENDERER[1, 1],
                                                      cx=INTRINSICS_HAMER_RENDERER[0, 2],
                                                      cy=INTRINSICS_HAMER_RENDERER[1, 2],
                                                      width=int(INTRINSICS_HAMER_RENDERER[0, 2] * 2),
                                                      height=int(INTRINSICS_HAMER_RENDERER[1, 2] * 2))
        joint_mesh_depth_im = np.asarray(joint_mesh_depth_im)[..., np.newaxis]
        plt.imshow(joint_mesh_depth_im)
        plt.show()
  
        plt.imshow(rgb_im)
        plt.scatter(index_finger_1_x, index_finger_1_y, color='blue', s=10)
        plt.scatter(index_finger_2_x, index_finger_2_y, color='orange', s=10)
        plt.scatter(index_finger_3_x, index_finger_3_y, color='purple', s=10)
        plt.scatter(thumb_1_x, thumb_1_y, color='blue', s=10)
        plt.scatter(thumb_2_x, thumb_2_y, color='orange', s=10)
        plt.scatter(thumb_3_x, thumb_3_y, color='purple', s=10)
        plt.show()    
        
    return middle_x, middle_y, index_finger_1_x, index_finger_1_y, index_finger_2_x, index_finger_2_y, \
        index_finger_3_x, index_finger_3_y, thumb_1_x, thumb_1_y, thumb_2_x, thumb_2_y, thumb_3_x, thumb_3_y 
