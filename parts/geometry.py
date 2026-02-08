# parts/geometry.py
import numpy as np
import open3d as o3d
import copy
import matplotlib.pyplot as plt
from util_functions import * # 假设这里有 compute_principal_axis, find_scaled_transformation 等
# from global_vars import INTRINSICS_REAL_CAMERA, INTRINSICS_HAMER_RENDERER, T_OPENGL_TO_OPENCV
from global_config import INTRINSICS_REAL_CAMERA, INTRINSICS_HAMER_RENDERER, T_OPENGL_TO_OPENCV,state


#夹爪坐标系变化到手部坐标系
def align_gripper_to_hand(hand_point_cloud, hand_pcd_as_o3d, gripper_pcd, gripper_pcd_as_o3d, vizualize=False):

    # Compute the principal axis of the point cloud
    principal_axis_h, second_axis_h = compute_principal_axis(hand_point_cloud)
    principal_axis_g, second_axis_g = compute_principal_axis(gripper_pcd, switch_principal_axis=True)
    middle_point_gripper = np.mean(gripper_pcd, axis=0)
    middle_point_hand = np.mean(hand_point_cloud, axis=0)
    if vizualize:
        # Plot a vector in the direction of the principal axis in open3d
        line_set_h1 = o3d.geometry.LineSet()
        line_set_h1.points = o3d.utility.Vector3dVector([middle_point_hand, middle_point_hand + principal_axis_h])
        line_set_h1.lines = o3d.utility.Vector2iVector([[0, 1]])
        # make line red
        line_set_h1.colors = o3d.utility.Vector3dVector([[1, 0, 0]])
        line_set_h2 = o3d.geometry.LineSet()
        line_set_h2.points = o3d.utility.Vector3dVector([middle_point_hand, middle_point_hand - principal_axis_h])
        line_set_h2.lines = o3d.utility.Vector2iVector([[0, 1]])
        # make line red
        line_set_h2.colors = o3d.utility.Vector3dVector([[1, 0, 0]])
        line_set_h3 = o3d.geometry.LineSet()
        line_set_h3.points = o3d.utility.Vector3dVector([middle_point_hand, middle_point_hand + second_axis_h])
        line_set_h3.lines = o3d.utility.Vector2iVector([[0, 1]])
        # make line green
        line_set_h3.colors = o3d.utility.Vector3dVector([[0, 1, 0]])
        line_set_h4 = o3d.geometry.LineSet()
        line_set_h4.points = o3d.utility.Vector3dVector([middle_point_hand, middle_point_hand - second_axis_h])
        line_set_h4.lines = o3d.utility.Vector2iVector([[0, 1]])
        # make line green
        line_set_h4.colors = o3d.utility.Vector3dVector([[0, 1, 0]])
        line_set_g1 = o3d.geometry.LineSet()
        line_set_g1.points = o3d.utility.Vector3dVector([middle_point_gripper, middle_point_gripper + principal_axis_g])
        line_set_g1.lines = o3d.utility.Vector2iVector([[0, 1]])
        # make line red
        line_set_g1.colors = o3d.utility.Vector3dVector([[1, 0, 0]])
        line_set_g2 = o3d.geometry.LineSet()
        line_set_g2.points = o3d.utility.Vector3dVector([middle_point_gripper, middle_point_gripper - principal_axis_g])
        line_set_g2.lines = o3d.utility.Vector2iVector([[0, 1]])
        # make line red
        line_set_g2.colors = o3d.utility.Vector3dVector([[1, 0, 0]])
        line_set_g3 = o3d.geometry.LineSet()
        line_set_g3.points = o3d.utility.Vector3dVector([middle_point_gripper, middle_point_gripper + second_axis_g])
        line_set_g3.lines = o3d.utility.Vector2iVector([[0, 1]])
        # make line green
        line_set_g3.colors = o3d.utility.Vector3dVector([[0, 1, 0]])
        line_set_g4 = o3d.geometry.LineSet()
        line_set_g4.points = o3d.utility.Vector3dVector([middle_point_gripper, middle_point_gripper - second_axis_g])
        line_set_g4.lines = o3d.utility.Vector2iVector([[0, 1]])
        # make line green
        line_set_g4.colors = o3d.utility.Vector3dVector([[0, 1, 0]])
        # Plot the coordinate frame of the gripper in open3d
        gripper_coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=.100, origin=middle_point_gripper)
        T_gripper_coord = np.eye(4)
        T_gripper_coord[3, :3] = middle_point_gripper
        # Show in open3d
        o3d.visualization.draw_geometries([hand_pcd_as_o3d, gripper_pcd_as_o3d,  line_set_h2, line_set_h3, line_set_h4, line_set_g2, line_set_g3, line_set_g4, gripper_coord])
    # Extract points on each pair of principal axes to compute the relative transformation
    # Extract points on the principal axis of the hand
    q = np.array([middle_point_hand, 
                hand_point_cloud[0] + principal_axis_h, 
                #   middle_point_hand - principal_axis_h, 
                middle_point_hand + second_axis_h,
                middle_point_hand - second_axis_h])
    p = np.array([middle_point_gripper,
                    # middle_point_gripper + principal_axis_g,
                    gripper_pcd[0] - principal_axis_g,
                    middle_point_gripper + second_axis_g,
                    middle_point_gripper - second_axis_g])
    # Compute the relative transformation between the two pairs of principal axes
    T = find_scaled_transformation(p, q, use_scale=True)
    # Apply the transformation matrix to the point cloud
    gripper_aligned_to_hand_pcd = apply_transformation(gripper_pcd, T)
    # Show in open3d
    gripper_aligned_to_hand_pcd_as_o3d = o3d.geometry.PointCloud()
    gripper_aligned_to_hand_pcd_as_o3d.points = o3d.utility.Vector3dVector(gripper_aligned_to_hand_pcd)
    gripper_aligned_to_hand_pcd_as_o3d.paint_uniform_color([0, 1, 0])

    if vizualize:
        o3d.visualization.draw_geometries([hand_pcd_as_o3d, gripper_pcd_as_o3d, gripper_aligned_to_hand_pcd_as_o3d])
    return gripper_aligned_to_hand_pcd_as_o3d

#渲染的手部模型映射到真实场景坐标系
def get_hand_pcd_in_scene_from_rendered_model(rgb_im, depth_im, rend_depth_front_view, new_depth_image, human_mask, vizualize=False):

    point_cloud_camera = depth_to_point_cloud(depth_im, INTRINSICS_REAL_CAMERA[0, 0], INTRINSICS_REAL_CAMERA[1, 1], INTRINSICS_REAL_CAMERA[0, 2], INTRINSICS_REAL_CAMERA[1, 2]) # Camera intrinsics, depth at real scale, although the shape is not accurate
    point_cloud_camera = point_cloud_camera.reshape(-1, 3)
    point_cloud_rescaled_depth = depth_to_point_cloud(new_depth_image, INTRINSICS_HAMER_RENDERER[0, 0], INTRINSICS_HAMER_RENDERER[1, 1], INTRINSICS_HAMER_RENDERER[0, 2],  INTRINSICS_HAMER_RENDERER[1, 2]) # Hamer intrinsics, depth at real scale, shape does not match real
    point_cloud_rescaled_depth = point_cloud_rescaled_depth.reshape(-1, 3)
    pcd_camera = o3d.geometry.PointCloud()
    pcd_camera.points = o3d.utility.Vector3dVector(point_cloud_camera)
    pcd_rescaled_depth = o3d.geometry.PointCloud()
    pcd_rescaled_depth.points = o3d.utility.Vector3dVector(point_cloud_rescaled_depth)

    show_scaled_up_pcd = False
    remove_outliers=True # does not work that well, a big hacky
    # print("point cloud camera shape: ", point_cloud_camera.shape)
    point_cloud_camera = point_cloud_camera.reshape(state.IM_HEIGHT, state.IM_WIDTH, 3)
    point_cloud_rescaled_depth = point_cloud_rescaled_depth.reshape(state.IM_HEIGHT, state.IM_WIDTH, 3)
    # Extract common points between rendered model and live image
    common_points_binary = np.zeros_like(new_depth_image)
    common_points_binary[new_depth_image > 0] = 1 # binary mask of points in the rendered model depth

    depth_im_hand =  human_mask * depth_im # extract human hand from the live image
    depth_im_hand = common_points_binary * depth_im_hand # extract common points between rendered depth model and live image
    if vizualize:
        print(depth_im_hand)
        plt.imshow(depth_im_hand)
        plt.title("Depth image of hand after human mask")
        plt.show()

    
    if remove_outliers: # a bit hacky, does not do much, if not cause problems
        mean_depth = np.median(new_depth_image[new_depth_image > 0])
        depth_im_hand[np.abs(depth_im_hand) > mean_depth + 0.2] = 0 # remove all points 2m away from the camera

    common_points = np.zeros_like(depth_im_hand)
    common_points[depth_im_hand > 0] = 1 # Binary mask of common points between rendered model depth and live image

    if vizualize:
        plt.imshow(common_points)
        plt.title("Common points")
        plt.show()

    common_pts_indices = np.argwhere(common_points > 0) # extract common points indices
    q = point_cloud_camera[common_pts_indices[:, 0], common_pts_indices[:, 1]]
    p = point_cloud_rescaled_depth[common_pts_indices[:, 0], common_pts_indices[:, 1]]

    T = find_scaled_transformation(p, q, use_scale=False) # Compute scale and rotation matrix Tp = q
    pcd_rendered_hand_to_live_hand = apply_transformation(point_cloud_rescaled_depth.reshape(-1, 3), T)
    pcd_rendered_hand_to_live_hand = pcd_rendered_hand_to_live_hand.reshape(-1, 3)
    pcd_rendered_hand_to_live_hand_full = copy.deepcopy(pcd_rendered_hand_to_live_hand)
    # pcd_rendered_hand_to_live_hand = pcd_rendered_hand_to_live_hand[pcd_rendered_hand_to_live_hand[:, 2] > .400]

    # Show the point cloud in open3d
    pcd_hand_to_scale = o3d.geometry.PointCloud()
    pcd_hand_to_scale.points = o3d.utility.Vector3dVector(pcd_rendered_hand_to_live_hand)
    hand_point_cloud = pcd_rendered_hand_to_live_hand # from above
    hand_point_cloud_full = pcd_rendered_hand_to_live_hand_full
    use_scaled_up_hand = True
    use_both_projected_hands = False
    clean_live_image_background = True
    if clean_live_image_background:
        mean_depth_hand = np.mean(new_depth_image[new_depth_image > 0])
        std_depth_hand = np.std(new_depth_image[new_depth_image > 0])
        depth_im[depth_im > mean_depth_hand + 1000] = 0
        depth_im[depth_im < mean_depth_hand - 1000] = 0
    if not use_scaled_up_hand and not use_both_projected_hands:
        hand_point_cloud = depth_to_point_cloud(new_depth_image, INTRINSICS_REAL_CAMERA[0, 0], INTRINSICS_REAL_CAMERA[1, 1], INTRINSICS_REAL_CAMERA[0, 2], INTRINSICS_REAL_CAMERA[1, 2])
        hand_point_cloud = hand_point_cloud.reshape(-1, 3)
        hand_point_cloud_full = hand_point_cloud
    if use_both_projected_hands:
        hand_point_cloud_via_intrinsics = depth_to_point_cloud(new_depth_image, INTRINSICS_REAL_CAMERA[0, 0], INTRINSICS_REAL_CAMERA[1, 1], INTRINSICS_REAL_CAMERA[0, 2], INTRINSICS_REAL_CAMERA[1, 2])
        hand_point_cloud_via_intrinsics = hand_point_cloud_via_intrinsics.reshape(-1, 3)
    image_point_cloud = depth_to_point_cloud(depth_im, INTRINSICS_REAL_CAMERA[0, 0], INTRINSICS_REAL_CAMERA[1, 1], INTRINSICS_REAL_CAMERA[0, 2], INTRINSICS_REAL_CAMERA[1, 2])
    image_point_cloud = image_point_cloud.reshape(-1, 3)
    # Point cloud to open3d point cloud
    pcd_hand_to_scale = o3d.geometry.PointCloud()
    pcd_hand_to_scale.points = o3d.utility.Vector3dVector(hand_point_cloud[hand_point_cloud[:, 2] > 0])
    # Set pcd color to red
    pcd_hand_to_scale.paint_uniform_color([1, 0, 0])
    pcd_image = o3d.geometry.PointCloud()
    pcd_image.points = o3d.utility.Vector3dVector(image_point_cloud)
    # Set color to RGB
    pcd_image.colors = o3d.utility.Vector3dVector(rgb_im.reshape(-1, 3))
    # Set pcd color to blue
    if use_both_projected_hands and vizualize:
        pcd_via_intrinsics = o3d.geometry.PointCloud()
        pcd_via_intrinsics.points = o3d.utility.Vector3dVector(hand_point_cloud_via_intrinsics[hand_point_cloud_via_intrinsics[:, 2] > 0])
        pcd_via_intrinsics.paint_uniform_color([0, 1, 0])
        pcd_hand_to_scale.paint_uniform_color([1, 0, 0])

    return hand_point_cloud, hand_point_cloud_full, image_point_cloud, pcd_hand_to_scale, pcd_image, T

def align_gripper_to_hand(hand_point_cloud, hand_pcd_as_o3d, gripper_pcd, gripper_pcd_as_o3d, vizualize=False):

    # Compute the principal axis of the point cloud
    principal_axis_h, second_axis_h = compute_principal_axis(hand_point_cloud)
    principal_axis_g, second_axis_g = compute_principal_axis(gripper_pcd, switch_principal_axis=True)
    middle_point_gripper = np.mean(gripper_pcd, axis=0)
    middle_point_hand = np.mean(hand_point_cloud, axis=0)
    if vizualize:
        # Plot a vector in the direction of the principal axis in open3d
        line_set_h1 = o3d.geometry.LineSet()
        line_set_h1.points = o3d.utility.Vector3dVector([middle_point_hand, middle_point_hand + principal_axis_h])
        line_set_h1.lines = o3d.utility.Vector2iVector([[0, 1]])
        # make line red
        line_set_h1.colors = o3d.utility.Vector3dVector([[1, 0, 0]])
        line_set_h2 = o3d.geometry.LineSet()
        line_set_h2.points = o3d.utility.Vector3dVector([middle_point_hand, middle_point_hand - principal_axis_h])
        line_set_h2.lines = o3d.utility.Vector2iVector([[0, 1]])
        # make line red
        line_set_h2.colors = o3d.utility.Vector3dVector([[1, 0, 0]])
        line_set_h3 = o3d.geometry.LineSet()
        line_set_h3.points = o3d.utility.Vector3dVector([middle_point_hand, middle_point_hand + second_axis_h])
        line_set_h3.lines = o3d.utility.Vector2iVector([[0, 1]])
        # make line green
        line_set_h3.colors = o3d.utility.Vector3dVector([[0, 1, 0]])
        line_set_h4 = o3d.geometry.LineSet()
        line_set_h4.points = o3d.utility.Vector3dVector([middle_point_hand, middle_point_hand - second_axis_h])
        line_set_h4.lines = o3d.utility.Vector2iVector([[0, 1]])
        # make line green
        line_set_h4.colors = o3d.utility.Vector3dVector([[0, 1, 0]])
        line_set_g1 = o3d.geometry.LineSet()
        line_set_g1.points = o3d.utility.Vector3dVector([middle_point_gripper, middle_point_gripper + principal_axis_g])
        line_set_g1.lines = o3d.utility.Vector2iVector([[0, 1]])
        # make line red
        line_set_g1.colors = o3d.utility.Vector3dVector([[1, 0, 0]])
        line_set_g2 = o3d.geometry.LineSet()
        line_set_g2.points = o3d.utility.Vector3dVector([middle_point_gripper, middle_point_gripper - principal_axis_g])
        line_set_g2.lines = o3d.utility.Vector2iVector([[0, 1]])
        # make line red
        line_set_g2.colors = o3d.utility.Vector3dVector([[1, 0, 0]])
        line_set_g3 = o3d.geometry.LineSet()
        line_set_g3.points = o3d.utility.Vector3dVector([middle_point_gripper, middle_point_gripper + second_axis_g])
        line_set_g3.lines = o3d.utility.Vector2iVector([[0, 1]])
        # make line green
        line_set_g3.colors = o3d.utility.Vector3dVector([[0, 1, 0]])
        line_set_g4 = o3d.geometry.LineSet()
        line_set_g4.points = o3d.utility.Vector3dVector([middle_point_gripper, middle_point_gripper - second_axis_g])
        line_set_g4.lines = o3d.utility.Vector2iVector([[0, 1]])
        # make line green
        line_set_g4.colors = o3d.utility.Vector3dVector([[0, 1, 0]])
        # Plot the coordinate frame of the gripper in open3d
        gripper_coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=.100, origin=middle_point_gripper)
        T_gripper_coord = np.eye(4)
        T_gripper_coord[3, :3] = middle_point_gripper
        # Show in open3d
        o3d.visualization.draw_geometries([hand_pcd_as_o3d, gripper_pcd_as_o3d,  line_set_h2, line_set_h3, line_set_h4, line_set_g2, line_set_g3, line_set_g4, gripper_coord])
    # Extract points on each pair of principal axes to compute the relative transformation
    # Extract points on the principal axis of the hand
    q = np.array([middle_point_hand, 
                hand_point_cloud[0] + principal_axis_h, 
                #   middle_point_hand - principal_axis_h, 
                middle_point_hand + second_axis_h,
                middle_point_hand - second_axis_h])
    p = np.array([middle_point_gripper,
                    # middle_point_gripper + principal_axis_g,
                    gripper_pcd[0] - principal_axis_g,
                    middle_point_gripper + second_axis_g,
                    middle_point_gripper - second_axis_g])
    # Compute the relative transformation between the two pairs of principal axes
    T = find_scaled_transformation(p, q, use_scale=True)
    # Apply the transformation matrix to the point cloud
    gripper_aligned_to_hand_pcd = apply_transformation(gripper_pcd, T)
    # Show in open3d
    gripper_aligned_to_hand_pcd_as_o3d = o3d.geometry.PointCloud()
    gripper_aligned_to_hand_pcd_as_o3d.points = o3d.utility.Vector3dVector(gripper_aligned_to_hand_pcd)
    gripper_aligned_to_hand_pcd_as_o3d.paint_uniform_color([0, 1, 0])

    if vizualize:
        o3d.visualization.draw_geometries([hand_pcd_as_o3d, gripper_pcd_as_o3d, gripper_aligned_to_hand_pcd_as_o3d])
    return gripper_aligned_to_hand_pcd_as_o3d

#基于手指关节点对齐夹爪
def align_gripper_with_hand_fingers(gripper_scaled_to_hand_pcd, 
                                    hand_pcd_as_o3d, 
                                    key_fingers_points, 
                                    gripper_aligned_to_hand_pcd_as_o3d,
                                    gripper_pcd_dense_mesh,  
                                    use_only_thumb_keypoints=False,
                                    use_only_index_keypoints=False,
                                    rescale_gripper_to_hand_opening=False, 
                                    rescale_hand_to_gripper_opening=False,
                                    vizualize=False,
                                    bias_transformation=np.eye(4)):
    assert (use_only_index_keypoints == use_only_thumb_keypoints) or use_only_index_keypoints == False , f'Either ONLY index {use_only_thumb_keypoints} or thumb keypoints {use_only_index_keypoints} can be used. Both False=Use both'
    assert (rescale_gripper_to_hand_opening == rescale_hand_to_gripper_opening) or rescale_gripper_to_hand_opening == False, f'Either rescale gripper to hand opening {rescale_gripper_to_hand_opening} or rescale hand to gripper opening {rescale_hand_to_gripper_opening} can be used. Both False=Do not rescale'

    dense_pcd_kpts = {"index_front": 517980, "thumb_front": 248802, "wrist": 246448}
    gripper_fingers = np.array([gripper_scaled_to_hand_pcd.points[dense_pcd_kpts["index_front"]], 
                                gripper_scaled_to_hand_pcd.points[dense_pcd_kpts["index_front"]],
                                gripper_scaled_to_hand_pcd.points[dense_pcd_kpts["index_front"]],
                                gripper_scaled_to_hand_pcd.points[dense_pcd_kpts["thumb_front"]],
                                gripper_scaled_to_hand_pcd.points[dense_pcd_kpts["thumb_front"]],
                                gripper_scaled_to_hand_pcd.points[dense_pcd_kpts["thumb_front"]],
                                gripper_scaled_to_hand_pcd.points[dense_pcd_kpts["wrist"]]])
    
    if rescale_gripper_to_hand_opening:
        distance_between_thumb_and_index = np.linalg.norm(key_fingers_points[0] - key_fingers_points[3])
        distance_between_thumb_and_index_gripper = np.linalg.norm(gripper_fingers[0] - gripper_fingers[3])
        scaling_factor = distance_between_thumb_and_index / distance_between_thumb_and_index_gripper
        center_gripper = gripper_scaled_to_hand_pcd.get_center()
        gripper_scaled_to_hand_pcd_np = (np.asarray(gripper_scaled_to_hand_pcd.points) - center_gripper) * np.array([scaling_factor, scaling_factor, scaling_factor]) + center_gripper
        gripper_scaled_to_hand_pcd.points = o3d.utility.Vector3dVector(gripper_scaled_to_hand_pcd_np)

    if rescale_hand_to_gripper_opening:
        distance_between_thumb_and_index = np.linalg.norm(key_fingers_points[0] - key_fingers_points[3])
        distance_between_thumb_and_index_gripper = np.linalg.norm(gripper_fingers[0] - gripper_fingers[3])
        scaling_factor = distance_between_thumb_and_index_gripper / distance_between_thumb_and_index
        center_hand = hand_pcd_as_o3d.get_center()
        hand_pcd_as_o3d_np = (np.asarray(hand_pcd_as_o3d.points) - center_hand) * np.array([scaling_factor, scaling_factor, scaling_factor]) + center_hand
        hand_pcd_as_o3d.points = o3d.utility.Vector3dVector(hand_pcd_as_o3d_np)
        # rescale the key fingers points
        key_finger_points_center = np.mean(key_fingers_points, axis=0)
        key_fingers_points = (key_fingers_points - key_finger_points_center) * np.array([scaling_factor, scaling_factor, scaling_factor]) + key_finger_points_center
        

    # determine a line that goes through the index and thumb of the hand
    kpt_o3d_sphere = []
    count = 0
    # key_fingers_points_4pts = np.array([key_fingers_points[0], key_fingers_points[1], key_fingers_points[3], key_fingers_points[4]])
    key_fingers_points_4pts = np.array([key_fingers_points[0], key_fingers_points[3], key_fingers_points[-1]])

    key_fingers_points = np.array(key_fingers_points_4pts)

    alpha = 1
    line_point = key_fingers_points[0] + alpha * (key_fingers_points[1] - key_fingers_points[0])
    unit_vec_difference = (key_fingers_points[1] - key_fingers_points[0]) / np.linalg.norm(key_fingers_points[1] - key_fingers_points[0])
    distance_gripper_fingers = np.linalg.norm(gripper_fingers[0] - gripper_fingers[4])
    distance_key_fingers = np.linalg.norm(key_fingers_points[0] - key_fingers_points[1])
    difference_half = np.abs(distance_gripper_fingers - distance_key_fingers)/2
    pt1 = key_fingers_points[0] - unit_vec_difference * difference_half
    pt2 = key_fingers_points[1] + unit_vec_difference * difference_half
    middle_finger_point = key_fingers_points[0] + unit_vec_difference * distance_key_fingers/2
    distance_middle_griper_middle_finger = np.linalg.norm(gripper_fingers[-1] - middle_finger_point)
    unit_difference_between_middle_finger_point_and_key_fingers_last = (middle_finger_point - key_fingers_points[-1]) / np.linalg.norm(middle_finger_point - key_fingers_points[-1])

    new_hand_point = middle_finger_point - unit_difference_between_middle_finger_point_and_key_fingers_last * distance_middle_griper_middle_finger
    distance_pt1_pt2 = np.linalg.norm(pt1 - pt2)
    print(f"Distance between pt1 and pt2: {distance_pt1_pt2}")
    print(f"Distance between gripper fingers: {distance_gripper_fingers}")
    key_fingers_points = np.array([pt1, pt2, key_fingers_points[-1]])
    # key_fingers_points = np.array([pt1, pt2])

    for kpt in key_fingers_points:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=.003)
        sphere.compute_vertex_normals()
        count += 1
        if count % 3 == 0:
            red, green, blue = 1, 0, 1
        elif count % 3 == 1:
            red, green, blue = 0,1,0
        else:
            red, green, blue = 1, 0.5, 0
        sphere.paint_uniform_color([red, green, blue])
        sphere.translate(kpt)
        kpt_o3d_sphere.append(sphere)
    
    # add middle finger point
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=.003)
    sphere.compute_vertex_normals()
    sphere.paint_uniform_color([1, .5, .777])
    sphere.translate(middle_finger_point)
    kpt_o3d_sphere.append(sphere)

    # gripper_fingers_4pts = np.array([gripper_fingers[1], gripper_fingers[2], gripper_fingers[4], gripper_fingers[5]])
    gripper_fingers_4pts = np.array([gripper_fingers[0], gripper_fingers[4], gripper_fingers[-1]])
    # gripper_fingers_4pts = np.array([gripper_fingers[0], gripper_fingers[4]])


    gripper_fingers = np.array(gripper_fingers_4pts)
    gripper_fingers_o3d = []
    count = 0
    # Create vizualizer to sequentilaly add spheres to the gripper fingers
    for kpt in gripper_fingers:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=.003)
        sphere.compute_vertex_normals()
        count += 1
        if count % 3 == 0:
            red, green, blue = 1, 0, 1 # color: purple
        elif count % 3 == 1:
            red, green, blue = 0, 1, 0
        else:
            red, green, blue = 1, 0.5, 0
        sphere.paint_uniform_color([red, green, blue])
        sphere.translate(kpt)
        gripper_fingers_o3d.append(sphere)
    
    if use_only_thumb_keypoints:
        key_fingers_points = key_fingers_points[2:]
        gripper_fingers = gripper_fingers[2:]
        gripper_fingers_o3d = gripper_fingers_o3d[2:]
        kpt_o3d_sphere = kpt_o3d_sphere[2:]

    if use_only_index_keypoints:
        key_fingers_points = key_fingers_points[:4]
        gripper_fingers = gripper_fingers[:4]
        gripper_fingers_o3d = gripper_fingers_o3d[:4]
        kpt_o3d_sphere = kpt_o3d_sphere[:4]


    T = find_scaled_transformation(gripper_fingers[:2], key_fingers_points[:2], use_scale=False)
    # transform the gripper_fingers_o3d to the hand frame
    for sphere in gripper_fingers_o3d:
        sphere.transform(T)

    gripper_pcd_before_transform = copy.deepcopy(gripper_scaled_to_hand_pcd)
    gripper_scaled_to_hand_pcd.transform(T)
    # Assume R is the rotation matrix you've computed and t is the translation
    # Transform z2
    R = T[:3, :3]
    t = T[:3, 3]
    z1 = key_fingers_points[-1]
    x1 = key_fingers_points[0]
    y1 = key_fingers_points[1]
    z2 = gripper_fingers[-1]
    x2 = gripper_fingers[0]
    y2 = gripper_fingers[1]

    z2_transformed = R @ z2 + t
    # Compute rotation axis (using x2 and y2 after transformation)
    x2_transformed = R @ x2 + t
    y2_transformed = R @ y2 + t
    rotation_axis = y2_transformed - x2_transformed #np.cross(x2_transformed - y2_transformed, z2_transformed - y2_transformed)


    # find theta that bring z2 as closest as possible to z2 while keeping the rotation axis the same
    distance = 10e10
    rotation_theta = None
    for theta in np.linspace(0, 2 * np.pi, 1000):
        R_additional = rotation_matrix(rotation_axis, theta)
        z2_final = (z2_transformed -  (y2_transformed + x2_transformed) / 2) @ R_additional.T + (y2_transformed + x2_transformed) / 2
        distance_temp = np.linalg.norm(z2_final - z1)
        if distance_temp < distance:
            distance = distance_temp
            rotation_theta = theta

    # Apply rotation about the axis
    R_additional = rotation_matrix(rotation_axis, rotation_theta)
    z2_final = (z2_transformed -  (y2_transformed + x2_transformed) / 2) @ R_additional.T + (y2_transformed + x2_transformed) / 2

    T2 = np.eye(4)
    T2[:3, :3] = R_additional
    gripper_scaled_to_hand_pcd_points = np.asarray(gripper_scaled_to_hand_pcd.points)
    gripper_scaled_to_hand_pcd_points = (gripper_scaled_to_hand_pcd_points - (y2_transformed + x2_transformed) / 2) @ R_additional.T + (y2_transformed + x2_transformed) / 2
    gripper_scaled_to_hand_pcd.points = o3d.utility.Vector3dVector(gripper_scaled_to_hand_pcd_points)
    # gripper_scaled_to_hand_pcd.transform(T2)
    gripper_aligned_to_hand_pcd_as_o3d.paint_uniform_color([.1, 1, 1])

    # z2_final = gripper_scaled_to_hand_pcd.points[dense_pcd_kpts["wrist"]]
    # add z2_final to sphere
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=.003)
    sphere.compute_vertex_normals()
    sphere.paint_uniform_color([0, 0, 0])
    sphere.translate(z2_final)
    kpt_o3d_sphere.append(sphere)

    if bias_transformation is None:
        bias_transformation = np.eye(4)
    # apply bias transformation in the gripper frame
    gripper_pose, gripper_zero_mean = get_gripper_transform_in_camera_frame(gripper_scaled_to_hand_pcd, 
                                                                            gripper_pcd_dense_mesh, 
                                                                            return_zero_meaned_gripper=True,
                                                                            vizualize=vizualize)
    gripper_pose = gripper_pose @ bias_transformation
    gripper_zero_mean.transform(gripper_pose)
    gripper_scaled_to_hand_pcd = copy.deepcopy(gripper_zero_mean)
        
    if vizualize:
            # o3d.visualization.draw_geometries([pcd_image, gripper_scaled_to_hand_pcd])
        line_o3d = o3d.geometry.LineSet()
        line_o3d.points = o3d.utility.Vector3dVector([key_fingers_points[0], key_fingers_points[0] + unit_vec_difference * 3])
        line_o3d.lines = o3d.utility.Vector2iVector([[0, 1]])

        line_o3d_2 = o3d.geometry.LineSet()
        line_o3d_2.points = o3d.utility.Vector3dVector([key_fingers_points[0], key_fingers_points[0] - unit_vec_difference * 3])
        line_o3d_2.lines = o3d.utility.Vector2iVector([[0, 1]])

        line_o3d_3 = o3d.geometry.LineSet()
        line_o3d_3.points = o3d.utility.Vector3dVector([middle_finger_point, key_fingers_points[1] + unit_difference_between_middle_finger_point_and_key_fingers_last * 3])
        line_o3d_3.lines = o3d.utility.Vector2iVector([[0, 1]])

        line_o3d_4 = o3d.geometry.LineSet()
        line_o3d_4.points = o3d.utility.Vector3dVector([middle_finger_point, key_fingers_points[1] - unit_difference_between_middle_finger_point_and_key_fingers_last * 3])
        line_o3d_4.lines = o3d.utility.Vector2iVector([[0, 1]])


        line_o3d_rotation_axis = o3d.geometry.LineSet()
        line_o3d_rotation_axis.points = o3d.utility.Vector3dVector([x2_transformed, x2_transformed + 10 * rotation_axis])
        line_o3d_rotation_axis.lines = o3d.utility.Vector2iVector([[0, 1]])

        line_o3d_rotation_axis_2 = o3d.geometry.LineSet()
        line_o3d_rotation_axis_2.points = o3d.utility.Vector3dVector([y2_transformed, y2_transformed - 10 * rotation_axis])
        line_o3d_rotation_axis_2.lines = o3d.utility.Vector2iVector([[0, 1]])

        gripper_frame_coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=.100, origin=[0, 0, 0])
        gripper_frame_coord.transform(gripper_pose)
        gripper_scaled_to_hand_pcd.paint_uniform_color([0, 1, 0])
        hand_pcd_as_o3d.paint_uniform_color([1, 0, 0])
        o3d.visualization.draw_geometries([hand_pcd_as_o3d, gripper_scaled_to_hand_pcd, gripper_frame_coord] + gripper_fingers_o3d + kpt_o3d_sphere + [line_o3d, line_o3d_2, line_o3d_3, line_o3d_4, line_o3d_rotation_axis, line_o3d_rotation_axis_2])
        # o3d.visualization.draw_geometries([hand_pcd_as_o3d, gripper_scaled_to_hand_pcd] + gripper_fingers_o3d + kpt_o3d_sphere + [line_o3d_rotation_axis])

    
    return gripper_scaled_to_hand_pcd, gripper_pose, distance_key_fingers

#专为按压任务设计，对齐到夹爪
def align_hand_to_gripper_press(gripper_pcd, 
                            actions, 
                            vizualize=False,
                            bias_transformation=np.eye(4)):
    
    gripper_pcd_original_mesh = copy.deepcopy(gripper_pcd)
    dense_pcd_kpts = {"index_front": 517980, 
                      "index_middle": 231197, 
                      "index_bottom":335530, 
                      "thumb_front": 248802, 
                      "thumb_middle":71859, 
                      "thumb_bottom":523328, 
                      "wrist": 246448}
    
    gripper_fingers = np.array([gripper_pcd.points[dense_pcd_kpts["index_front"]], 
                                gripper_pcd.points[dense_pcd_kpts["index_middle"]],
                                gripper_pcd.points[dense_pcd_kpts["index_bottom"]],
                                gripper_pcd.points[dense_pcd_kpts["thumb_front"]],
                                gripper_pcd.points[dense_pcd_kpts["thumb_middle"]],
                                gripper_pcd.points[dense_pcd_kpts["thumb_bottom"]],
                                gripper_pcd.points[dense_pcd_kpts["wrist"]]])
    
    gripper_fingers[0] = gripper_fingers[0] - (gripper_fingers[0] - gripper_fingers[3]) / 2 
    gripper_fingers[4] = gripper_fingers[1] - (gripper_fingers[1] - gripper_fingers[4]) / 2
    gripper_fingers[-1] = gripper_fingers[2] - (gripper_fingers[2] - gripper_fingers[5]) / 2
    key_fingers_points = actions

    kpt_o3d_sphere = []
    count = 0

    for kpt in key_fingers_points:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=.003)
        sphere.compute_vertex_normals()
        count += 1
        if count % 3 == 0:
            red, green, blue = 1, 0, 1
        elif count % 3 == 1:
            red, green, blue = 0,1,0
        else:
            red, green, blue = 1, 0.5, 0
        sphere.paint_uniform_color([red, green, blue])
        sphere.translate(kpt)
        kpt_o3d_sphere.append(sphere)
    

    gripper_fingers_4pts = np.array([gripper_fingers[0], gripper_fingers[4], gripper_fingers[-1]])
    gripper_fingers = np.array(gripper_fingers_4pts)
    gripper_fingers_o3d = []
    count = 0
    # Create vizualizer to sequentilaly add spheres to the gripper fingers
    for kpt in gripper_fingers:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=.003)
        sphere.compute_vertex_normals()
        count += 1
        if count % 3 == 0:
            red, green, blue = 1, 0, 1 # color: purple
        elif count % 3 == 1:
            red, green, blue = 0, 1, 0
        else:
            red, green, blue = 1, 0.5, 0
        sphere.paint_uniform_color([red, green, blue])
        sphere.translate(kpt)
        gripper_fingers_o3d.append(sphere)

    # o3d.visualization.draw_geometries([gripper_pcd] + gripper_fingers_o3d )


    T = find_scaled_transformation(gripper_fingers, key_fingers_points, use_scale=False)
    # transform the gripper_fingers_o3d to the hand frame
    for sphere in gripper_fingers_o3d:
        sphere.transform(T)
    gripper_pcd.transform(T)


    if bias_transformation is None:
        bias_transformation = np.eye(4)
    # apply bias transformation in the gripper frame
    gripper_pose, gripper_zero_mean = get_gripper_transform_in_camera_frame(gripper_pcd, 
                                                                            gripper_pcd_original_mesh, 
                                                                            return_zero_meaned_gripper=True,
                                                                            vizualize=vizualize)
    gripper_pose = gripper_pose @ bias_transformation
    gripper_zero_mean.transform(gripper_pose)
    gripper_pcd = copy.deepcopy(gripper_zero_mean)
        
    if vizualize:
            # o3d.visualization.draw_geometries([pcd_image, gripper_scaled_to_hand_pcd])

        gripper_frame_coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=.100, origin=[0, 0, 0])
        gripper_frame_coord.transform(gripper_pose)
        gripper_pcd.paint_uniform_color([0.0,0.0,0.0])
        o3d.visualization.draw_geometries([gripper_pcd] + kpt_o3d_sphere )
        # o3d.visualization.draw_geometries([hand_pcd_as_o3d, gripper_scaled_to_hand_pcd] + gripper_fingers_o3d + kpt_o3d_sphere + [line_o3d_rotation_axis])

    
    return gripper_pcd, gripper_pose

#计算夹爪相对于相机坐标系的变换矩阵
def get_gripper_transform_in_camera_frame(gripper_scaled_to_hand_pcd, original_hand_pcd, vizualize=False, return_zero_meaned_gripper=False):
    # Add a world frame
    world_coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=.100, origin=[0, 0, 0])
    # Show in open3d
    # o3d.visualization.draw_geometries([gripper_scaled_to_hand_pcd, sphere])
    gripper_zero_origin = copy.deepcopy(np.asarray(original_hand_pcd.points))
    # zero mean the z-axis
    gripper_zero_origin[:, 2] = gripper_zero_origin[:, 2] - np.mean(gripper_zero_origin[:, 2])
    # rotate 90 degrees around the x-axis
    R = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    gripper_zero_origin = np.dot(R, gripper_zero_origin.T).T
    gripper_zero_origin_o3d = o3d.geometry.PointCloud()
    gripper_zero_origin_o3d.points = o3d.utility.Vector3dVector(gripper_zero_origin)
    p = np.asarray(gripper_zero_origin_o3d.points)
    q = np.asarray(gripper_scaled_to_hand_pcd.points)
    T = find_scaled_transformation(p, q, use_scale=False)
    gripper_coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=.100, origin=[0, 0, 0])
    gripper_coord.transform(T)

    if vizualize:
        o3d.visualization.draw_geometries([gripper_scaled_to_hand_pcd, gripper_zero_origin_o3d, gripper_coord, world_coord])
    if return_zero_meaned_gripper:
        return T, gripper_zero_origin_o3d
    return T

#应用变换矩阵将手部模型转换到真实世界坐标系
def mesh_and_joints_to_world_metric_space(mesh, joints, T_mesh_to_live, scaling_rendered_to_live, live_image_pcd = None, vizualize=False):
    # Visualize alignment of partial point cloud in the world frame
    points_mesh = mesh.sample(10000)
    # points_mesh in camera coordinates
    points_mesh = np.linalg.inv(T_OPENGL_TO_OPENCV) @ np.vstack((points_mesh.T, np.ones((1, points_mesh.shape[0]))))
    points_mesh = points_mesh[:3, :].T
    # points_mesh = points_mesh * scaling_rendered_to_live
    points_mesh = apply_transformation(points_mesh, T_mesh_to_live)

    joint_mesh = np.linalg.inv(T_OPENGL_TO_OPENCV) @ np.vstack((joints.T, np.ones((1, joints.shape[0]))))
    joint_mesh = joint_mesh[:3, :].T
    # joint_mesh = joint_mesh * scaling_rendered_to_live
    joint_mesh = apply_transformation(joint_mesh, T_mesh_to_live)

    mesh.apply_transform(np.linalg.inv(T_OPENGL_TO_OPENCV)) # probably could have done this before points_mesh to save some lines, but yeah it is what it is
    mesh.apply_scale(scaling_rendered_to_live)
    mesh.apply_transform(T_mesh_to_live)
    mesh_o3d = o3d.geometry.TriangleMesh()
    mesh_o3d.vertices = o3d.utility.Vector3dVector(mesh.vertices)
    mesh_o3d.triangles = o3d.utility.Vector3iVector(mesh.faces)
    mesh_o3d.compute_vertex_normals()
    mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(mesh.visual.vertex_colors[:, :3] / 255.0)

    if vizualize:
        assert live_image_pcd is not None, "Need to provide the live image point cloud to vizualize mesh and joints in live image space"


        pcd_mesh = o3d.geometry.PointCloud()
        pcd_mesh.points = o3d.utility.Vector3dVector(points_mesh)
        # pcd joints as speheres
        pcd_joints = o3d.geometry.TriangleMesh.create_coordinate_frame(size=.100, origin=joint_mesh[0])
        for j in joint_mesh:
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=.005)
            sphere.compute_vertex_normals()
            sphere.paint_uniform_color([1, 0, 0])
            sphere.translate(j)
            pcd_joints += sphere
        o3d.visualization.draw_geometries([pcd_mesh, pcd_joints, live_image_pcd, mesh_o3d])
    return points_mesh, joint_mesh, mesh_o3d


#0-4-8映射
def align_gripper_048(gripper_pcd, hand_joints_048, vizualize=False):
    """
    使用 0-4-8 映射对齐夹爪
    hand_joints_048: shape (3, 3) -> [Wrist, Thumb_Tip, Index_Tip]
    """
    
    # 1. 定义夹爪模型上的三个对应点 (单位: 米)
    # 注意：这些坐标需要根据你的 gripper_pcd 模型原点来调整
    # 假设夹爪原点在底部中心，Y轴向前，X轴向右
    # 这里只是示例，你需要测量你的夹爪模型的实际尺寸
    gripper_base = np.array([0.0, 0.0, 0.0])       # 对应手腕
    gripper_l_finger = np.array([0.04, -0.1, 0.16]) # 对应拇指 (左宽, 前后, 高度)
    gripper_r_finger = np.array([-0.06, -0.1, 0.16])  # 对应食指
    
    source_points = np.stack([gripper_base, gripper_l_finger, gripper_r_finger])
    target_points = hand_joints_048 # 真实手部的三个点

    # 2. 计算变换矩阵 (SVD/Umeyama算法)
    # 这会找到最佳的旋转和平移，把夹爪的这3个点对齐到手的这3个点
    T = find_scaled_transformation(source_points, target_points, use_scale=False)
    
    # 3. 应用变换
    gripper_aligned = copy.deepcopy(gripper_pcd)
    gripper_aligned.transform(T)
    
    # 4. 计算开合度 (距离比率)
    # 真实手部拇指和食指的距离
    hand_opening = np.linalg.norm(target_points[1] - target_points[2])
    # 夹爪最大物理开合度 (例如 Robotiq 2F-85 是 0.085m)
    max_gripper_width = 0.085 
    
    # 归一化到 0-1
    opening_percent = np.clip(hand_opening / max_gripper_width, 0, 1)

    return gripper_aligned, T, opening_percent

def align_gripper_palm(gripper_pcd, hand_joints, vizualize=False):
    """
    改进版：基于手掌平面 (Palm-Based) 对齐夹爪
    hand_joints: 包含所有关键点的数组，通常 shape 为 (21, 3)
    {"wrist": 0, "index_mcp": 1, "index_pip": 2, 
                 "index_dip": 3, "middle_mcp": 4, "middle_pip": 5, 
                 "middle_dip": 6, "pinkie_mcp": 7, "pinkie_pip": 8, 
                 "pinkie_dip": 9, "ring_mcp": 10, "ring_pip": 11, 
                 "ring_dip": 12, "thumb_mcp": 13, "thumb_pip": 14, 
                 "thumb_dip": 15, "thumb_tip": 16, "index_tip": 17, 
                 "middle_tip": 18, "ring_tip": 19, "pinky_tip": 20}
    """

    # ==========================================
    # 1. 提取用于定姿态的“刚性”关键点 (手掌三角)
    # ==========================================
    # MANO 索引: 0=Wrist, 5=Thump MCP (食指根), 17=Middle MCP (小指根)
    p_wrist = hand_joints[13]
    p_right_mcp = hand_joints[16]
    p_left_mcp = hand_joints[17]
    
    target_points = np.stack([p_wrist, p_right_mcp, p_left_mcp])

    # ==========================================
    # 2. 定义夹爪上对应的虚拟锚点
    # ==========================================
    # 这需要根据你的夹爪模型尺寸来定。
    # 假设夹爪原点在底部，Z轴向前，Y轴向上(垂直手掌)，X轴向右
    
    # 手掌的几何中心大概在手腕前方 5-8cm 处
    # 我们需要构建一个与 (Wrist, IndexMCP, PinkyMCP) 形状相似的虚拟三角形
    
    # 调试建议：你可以把这三个点画出来看看
    
    # 锚点 A: 对应手腕 (Base)
    src_base = np.array([0.0, 0.0, -0.05])  # 往后缩一点，让夹爪中心落在掌心
    
    # 锚点 B: 对应食指根部 (右前方)
    # 假设手掌宽 8cm, 长 8cm
    src_index = np.array([0.04, 0.0, 0.03]) 
    
    # 锚点 C: 对应小指根部 (左前方)
    src_pinky = np.array([-0.04, 0.0, 0.03])

    source_points = np.stack([src_base, src_index, src_pinky])

    # ==========================================
    # 3. 计算刚性变换 (SVD) - 确定 Pose
    # ==========================================
    # 这里的 T 只负责把夹爪“贴”在手掌上，不会受手指弯曲影响
    T = find_scaled_transformation(source_points, target_points, use_scale=False)
    
    # 应用变换
    gripper_aligned = copy.deepcopy(gripper_pcd)
    gripper_aligned.transform(T)
    gripper_pose = T

    # ==========================================
    # 4. 独立计算动作 (Action)
    # ==========================================
    # 使用指尖距离来计算开合
    p_thumb_tip = hand_joints[4]
    p_index_tip = hand_joints[8]
    
    hand_opening = np.linalg.norm(p_thumb_tip - p_index_tip)
    max_gripper_width = 0.085 # 8.5cm
    opening_percent = np.clip(hand_opening / max_gripper_width, 0, 1)

    return gripper_aligned, gripper_pose, opening_percent

def find_scaled_transformation(source_points, target_points, use_scale=False):
    """
    辅助函数：计算 SVD 变换
    """
    assert source_points.shape == target_points.shape
    
    # 1. 计算质心
    centroid_src = np.mean(source_points, axis=0)
    centroid_tgt = np.mean(target_points, axis=0)
    
    # 2. 去中心化
    src_centered = source_points - centroid_src
    tgt_centered = target_points - centroid_tgt
    
    # 3. 计算旋转 (R)
    H = np.dot(src_centered.T, tgt_centered)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)
    
    # 处理反射情况 (det(R) = -1)
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = np.dot(Vt.T, U.T)
        
    # 4. 计算缩放 (Scale) - 刚性物体通常设为 False
    scale = 1.0
    if use_scale:
        var_src = np.var(src_centered, axis=0).sum()
        scale = 1.0 / var_src * np.sum(S)
        
    # 5. 计算平移 (t)
    t = centroid_tgt - scale * np.dot(R, centroid_src)
    
    # 6. 构建 4x4 矩阵
    T = np.eye(4)
    T[:3, :3] = scale * R
    T[:3, 3] = t
    
    return T

def align_gripper_with_triangle(gripper_pcd, hand_joints, vizualize=False):
    """
    "wrist": 0, "index_mcp": 1, "index_pip": 2, 
                 "index_dip": 3, "middle_mcp": 4, "middle_pip": 5, 
                 "middle_dip": 6, "pinkie_mcp": 7, "pinkie_pip": 8, 
                 "pinkie_dip": 9, "ring_mcp": 10, "ring_pip": 11, 
                 "ring_dip": 12, "thumb_mcp": 13, "thumb_pip": 14, 
                 "thumb_dip": 15, "thumb_tip": 16, "index_tip": 17, 
                 "middle_tip": 18, "ring_tip": 19, "pinky_tip": 20
    根据用户自定义三角形对齐夹爪：
    三角形顶点：大拇指根部 (Thumb Root), 大拇指指尖 (Thumb Tip), 食指指尖 (Index Tip)
    
    假设 hand_joints 为标准 21 点定义 (MediaPipe/OpenPose):
    2: Thumb_MCP  (大拇指根部)
    4: Thumb_Tip  (大拇指指尖)
    8: Index_Tip  (食指指尖)
    """
    
    # 1. 提取手部关键点 (Target Points)
    # 索引定义
    idx_thumb_root = 13 # Thumb MCP
    idx_thumb_tip = 16   # Thumb Tip
    idx_index_tip = 17   # Index Tip
    
    p_thumb_root = hand_joints[idx_thumb_root]
    p_thumb_tip = hand_joints[idx_thumb_tip]
    p_index_tip = hand_joints[idx_index_tip]
    
    target_points = np.stack([p_thumb_root, p_thumb_tip, p_index_tip])
    
    # 2. 定义夹爪上的对应点 (Source Points)
    # 调整说明：
    # 这里的 source_points 必须在形状上尽可能接近 target_points (手部三角形)，否则 SVD 会产生错误的旋转。
    # 手部三角形 (Root->ThumbTip->IndexTip) 通常接近一个直角三角形（拇指根到指尖是一条边，指尖连线是另一条边）。
    # 之前的“等腰三角形”（根部在中心）会导致夹爪旋转偏移。
    
    # 定义虚拟指尖位置 (根据夹爪实际开合宽度设定)
    # 假设 Z 轴向前 (Approach)，X 轴为开合方向
    # 增加 Z 值可以让夹爪模型相对于手部整体"后退" (解决在指尖前方的问题)
    tip_z_offset = 0.12  # 增大此值，夹爪模型会向后移动
    half_width = 0.04    # 4cm 半宽 -> 8cm 开合
    
    # 对应 Thumb Tip (假设为正 X 侧)
    gripper_thumb_tip = np.array([half_width, 0.0, tip_z_offset]) 
    
    # 对应 Index Tip (假设为负 X 侧)
    gripper_index_tip = np.array([-half_width, 0.0, tip_z_offset])
    
    # 对应 Thumb MCP (虚拟根部)
    # 拇指根部应该位于拇指指尖的"后方"且偏向同侧
    # 下面的坐标表示：在 ThumbTip 的位置，沿着 -Z 方向退回 thumb_len 距离，稍微偏移一点 Y/X
    thumb_len = 0.06 # 拇指长度估计 6cm
    
    # [X, Y, Z]
    # 我们把 Root 设为跟 ThumbTip 同侧 (half_width)，但在 Z 轴上靠后
    gripper_virtual_root = np.array([half_width, -0.02, tip_z_offset - thumb_len])
    
    source_points = np.stack([gripper_virtual_root, gripper_thumb_tip, gripper_index_tip])
    
    # 3. 计算刚性变换 (SVD)
    T = find_scaled_transformation(source_points, target_points, use_scale=False)
    
    # 4. 应用变换
    gripper_aligned = copy.deepcopy(gripper_pcd)
    gripper_aligned.transform(T)
    gripper_pose = T
    
    # 5. 计算开合度 (Opening Degree)
    # 基于指尖距离
    dist_hand = np.linalg.norm(p_thumb_tip - p_index_tip)
    max_width = 0.085 # 夹爪最大开合度 (m)
    opening_percent = np.clip(dist_hand / max_width, 0, 1)
    
    return gripper_aligned, gripper_pose, opening_percent