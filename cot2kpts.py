
import os
import numpy as np
import sys

# Ensure we can import from parts/
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from spatial_utils import SpatialMapper
from parts.geometry import find_scaled_transformation

def parse_grid_trajectory(file_path):
    """
    Parses headers from seq_004_predicted_grid_trajectory.txt
    Each line: [x1, y1, z1, x2, y2, z2, x3, y3, z3]
    Points: Thumb Root, Thumb Tip, Index Tip
    """
    traj = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('['): 
                # Handle standard list format [1, 2, ...]
                try:
                    # Remove brackets if they exist strictly for parsing
                    clean_line = line.replace('[', '').replace(']', '').replace(',', ' ')
                    vals = list(map(float, clean_line.split()))
                    if len(vals) == 9:
                        traj.append(np.array(vals).reshape(3, 3))
                except:
                    continue
    return traj

def compute_pose_and_opening(keypoints_world):
    """
    Recover 4x4 Pose and Opening from 3 points:
    P1: Thumb Root (Proxy for Base/Palm anchor)
    P2: Thumb Tip
    P3: Index Tip
    """
    p_root, p_thumb, p_index = keypoints_world[0], keypoints_world[1], keypoints_world[2]
    
    # 1. Calculate Opening
    width = np.linalg.norm(p_thumb - p_index)
    
    # 2. Define Source Points in Canonical Gripper Frame
    # We define a "Idea" gripper in local frame
    # Origin at Root
    # Fingers pointing forward (Z+)
    # Opening along X
    
    # Let's verify standard gripper dimensions from geometry.py
    # But simpler: We just want a consistent frame.
    # Let's assume Root is at (0,0,0)
    # Thumb Tip is at (width/2, 0, length)
    # Index Tip is at (-width/2, 0, length)
    # But length varies? 
    # Better: Use the actual computed width and a fixed length to define Source structure?
    # NO: The Rigid Transform (SVD) will align best fit.
    # We should define a CONSTANT source shape if we want a consistent Frame Reference.
    # If the gripper deforms (fingers execute opening), the triangle shape changes.
    
    # Strategy: 
    # The Pose should represent the "Base" of the hand.
    # P_root is physically attached to the base.
    # P_thumb and P_index move relative to P_root.
    
    # If we align to [Root, Thumb, Index], the orientation will fluctuate as fingers open/close.
    # Ideally, we align to [Root, Knuckles...] but we only have 3 points.
    
    # Robust Frame Construction:
    # Origin = P_Root
    # Z_axis (Forward) = Vector from Root to Midpoint(Thumb, Index)
    # Y_axis (Up) = Cross(Using Normal of triangle)
    # X_axis = Cross(Y, Z)
    
    origin = p_root
    mid_fingers = (p_thumb + p_index) / 2.0
    
    z_vec = mid_fingers - origin
    z_vec /= (np.linalg.norm(z_vec) + 1e-8)
    
    # Temporary X axis: Vector from Thumb to Index (or Index to Thumb)
    # Right-hand rule logic
    temp_vec = p_index - p_thumb 
    
    # Y axis = Z cross Temp produces Up
    y_vec = np.cross(z_vec, temp_vec)
    y_vec /= (np.linalg.norm(y_vec) + 1e-8)
    
    # Recompute X to be orthogonal
    x_vec = np.cross(y_vec, z_vec)
    x_vec /= (np.linalg.norm(x_vec) + 1e-8)
    
    # Construct Rotation Matrix
    R = np.column_stack((x_vec, y_vec, z_vec))
    
    # Construct 4x4 Matrix
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = origin
    
    return T, width

def main():
    root_dir = "assets/recordings"
    seq_name = "seq_004"
    input_file = f"{root_dir}/{seq_name}/txt/{seq_name}_predicted_grid_trajectory.txt"
    output_file = f"{root_dir}/{seq_name}/txt/{seq_name}_trajectory_se3.txt"
    
    if not os.path.exists(input_file):
        # Fallback to current dir if running locally or not stored yet
        if os.path.exists(f"{seq_name}_predicted_grid_trajectory.txt"):
            input_file = f"{seq_name}_predicted_grid_trajectory.txt"
        else:
            print(f"Error: {input_file} not found.")
            return

    # Init Mapper
    # Use default params matching cot_kat.py assumptions
    mapper = SpatialMapper() 
    
    grid_traj = parse_grid_trajectory(input_file)
    print(f"Loaded {len(grid_traj)} frames.")
    
    poses = []
    
    for i, frame_grid in enumerate(grid_traj):
        # frame_grid is 3x3 array of grid coords
        
        # 1. Grid -> World
        world_pts = []
        for pt in frame_grid: # 3 pts
            # pt is [col, row, depth]? check spatial_utils
            # grid_coords are (i, j, k)
            wp = mapper.grid_to_world(tuple(pt.astype(int)))
            world_pts.append(wp)
        
        world_pts = np.array(world_pts)
        
        # 2. Compute Pose
        T, width = compute_pose_and_opening(world_pts)
        
        # Flatten for saving
        # Format: r11 r12 r13 tx ... width
        # Or standard 4x4 flat
        flat_pose = T.flatten().tolist()
        poses.append(flat_pose + [width])
        
    # Save
    with open(output_file, 'w') as f:
        for i, (wp, _) in enumerate(zip(grid_traj, poses)):
            # Recompute proper pose loop logic or just use stored
            # We stored flattened poses, let's just reconstruct or grab from loop
            pass 
        
        # Rewriting this block properly
        pass

    with open(output_file, 'w') as f:
        for p in poses:
            # p is flat 16 + 1
            matrix_flat = p[:16]
            width = p[16]
            # Reshape to 4x4 for "Array Form" if implied, or just list
            # User said "Transform matrix use array form", usually implies structure
            matrix_4x4 = np.array(matrix_flat).reshape(4,4).tolist()
            f.write(f"{matrix_4x4} {width}\n")
            
    print(f"Saved SE3 trajectory to {output_file}")
    
    # Visualize verification?
    # Simple check of first frame
    if len(poses) > 0:
        print("First Frame Pose:")
        print(np.array(poses[0][:-1]).reshape(4,4))
        print(f"Width: {poses[0][-1]}")

if __name__ == "__main__":
    main()
