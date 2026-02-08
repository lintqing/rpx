import numpy as np
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from global_config import INTRINSICS_REAL_CAMERA

class SpatialMapper:
    def __init__(self, 
                 intrinsics: np.ndarray = None, 
                 workspace_center: Tuple[float, float, float] = (0.0, 0.0, 0.4), # 中心调整至0.4m (符合数据统计)
                 cube_size: float = 0.4): # 缩小至0.4m立方体以提高分辨率 (4cm/格)
        """
        初始化空间映射器
        :param intrinsics: 相机内参矩阵 3x3
        :param workspace_center: 1m^3 立方体在相机坐标系下的中心位置 (X, Y, Z)
        :param cube_size: 立方体边长
        """
        # 如果没有传入内参，使用一个通用的近似值（建议从 global_config 获取）
        if intrinsics is None:
            self.fx, self.fy = INTRINSICS_REAL_CAMERA[0, 0], INTRINSICS_REAL_CAMERA[1, 1]
            self.cx, self.cy = INTRINSICS_REAL_CAMERA[0, 2], INTRINSICS_REAL_CAMERA[1, 2]
        else:
            self.fx = intrinsics[0, 0]
            self.fy = intrinsics[1, 1]
            self.cx = intrinsics[0, 2]
            self.cy = intrinsics[1, 2]

        # 定义 1m^3 的物理边界
        half_size = cube_size / 2.0
        self.x_range = (workspace_center[0] - half_size, workspace_center[0] + half_size)
        self.y_range = (workspace_center[1] - half_size, workspace_center[1] + half_size)
        self.z_range = (workspace_center[2] - half_size, workspace_center[2] + half_size)

    def pixel_to_world(self, u: float, v: float, depth: float) -> np.ndarray:
        """将像素坐标和深度转换为相机坐标系下的 3D 坐标 (X, Y, Z)"""
        if depth <= 0:
            return np.array([0, 0, 0])
        z = depth
        x = (u - self.cx) * z / self.fx
        y = (v - self.cy) * z / self.fy
        return np.array([x, y, z])

    def world_to_pixel(self, world_coord: np.ndarray) -> Tuple[int, int]:
        """将相机坐标系下的 3D 坐标 (X, Y, Z) 投影回像素坐标 (u, v)"""
        x, y, z = world_coord
        if z <= 0:
            return (-1, -1) # 无效深度
            
        u = x * self.fx / z + self.cx
        v = y * self.fy / z + self.cy
        return int(u), int(v)

    def world_to_grid(self, world_coord: np.ndarray) -> Tuple[int, int, int]:
        """将 3D 物理坐标转换为 60*60*60 的网格索引 (0-59)"""
        x, y, z = world_coord
        
        # 线性映射并限位在 0-59
        i = int(np.clip((x - self.x_range[0]) / (self.x_range[1] - self.x_range[0]) * 60, 0, 59))
        j = int(np.clip((y - self.y_range[0]) / (self.y_range[1] - self.y_range[0]) * 60, 0, 59))
        k = int(np.clip((z - self.z_range[0]) / (self.z_range[1] - self.z_range[0]) * 60, 0, 59))
        
        return (i, j, k)

    def grid_to_world(self, grid_coord: Tuple[int, int, int]) -> np.ndarray:
        """将网格索引还原为物理坐标（取体素中心点），用于 KAT 后处理"""
        i, j, k = grid_coord
        # 动态计算分辨率: 范围宽度 / 格子数
        res_x = (self.x_range[1] - self.x_range[0]) / 60.0
        res_y = (self.y_range[1] - self.y_range[0]) / 60.0
        res_z = (self.z_range[1] - self.z_range[0]) / 60.0
        
        x = self.x_range[0] + (i + 0.5) * res_x
        y = self.y_range[0] + (j + 0.5) * res_y
        z = self.z_range[0] + (k + 0.5) * res_z
        return np.array([x, y, z])

def extract_hand_node(hand_kpts: np.ndarray) -> np.ndarray:
    """
    从 MANO 21 个点中提取抽象节点
    根据讨论，取：腕部(0), 拇指根部(1), 食指根部(5) 的平均值
    :param hand_kpts: (21, 3) 的 3D 坐标
    """
    # 这里的 3D 坐标应该是已经由像素+深度转换后的物理坐标
    selected_indices = [0, 1, 5]
    node = np.mean(hand_kpts[selected_indices], axis=0)
    return node

def generate_discrete_trajectory(start_grid: Tuple[int, int, int], 
                                 end_grid: Tuple[int, int, int]) -> List[Tuple[int, int, int]]:
    """
    在两个 60*60*60 网格点之间生成直线离散路径 (用于填充 KAT 的输入序列)
    使用简单的插值算法
    """
    start = np.array(start_grid)
    end = np.array(end_grid)
    
    # 计算需要生成的步数（取三个轴向最大差值）
    steps = max(np.abs(end - start))
    if steps == 0:
        return [start_grid]
    
    trajectory = []
    for s in range(steps + 1):
        # 线性插值
        point = start + (end - start) * (s / steps)
        # 取整得到网格索引
        grid_point = tuple(np.round(point).astype(int))
        if not trajectory or grid_point != trajectory[-1]:
            trajectory.append(grid_point)
            
    return trajectory

def visualize_phase_movement(img_start, img_end, grid_start, grid_end, phase_name, save_path):
    """
    在一张图里展示阶段的起始、结束状态及其位移
    :param img_start: 阶段开始时的 RGB 图像
    :param img_end: 阶段结束时的 RGB 图像
    :param grid_start: 起始 (I, J, K)
    :param grid_end: 结束 (I, J, K)
    """
    fig = plt.figure(figsize=(18, 6))
    
    # 1. 左侧：起始帧
    ax1 = fig.add_subplot(131)
    ax1.imshow(img_start)
    ax1.set_title(f"START: {grid_start}")
    ax1.axis('off')
    
    # 2. 中间：结束帧
    ax2 = fig.add_subplot(132)
    ax2.imshow(img_end)
    ax2.set_title(f"END: {grid_end}")
    ax2.axis('off')
    
    # 3. 右侧：3D 位移网格
    ax3 = fig.add_subplot(133, projection='3d')
    
    # 提取坐标
    x1, y1, z1 = grid_start
    x2, y2, z2 = grid_end
    
    # 绘制起始点 (蓝色) 和 结束点 (红色)
    ax3.scatter([x1], [y1], [z1], color='blue', s=250, marker='o', label='Start', edgecolors='black', alpha=0.8)
    ax3.scatter([x2], [y2], [z2], color='red', s=300, marker='s', label='End', edgecolors='black', alpha=0.9)
    
    # 绘制带箭头的位移向量
    # quiver(x, y, z, u, v, w) -> u,v,w 是向量方向
    ax3.quiver(x1, y1, z1, x2-x1, y2-y1, z2-z1, 
               color='green', arrow_length_ratio=0.2, linewidth=3, label='Movement')

    # 装饰网格
    ax3.set_xlim(0, 59)
    ax3.set_ylim(0, 59)
    ax3.set_zlim(0, 59)
    ax3.set_xticks(range(0, 61, 5))
    ax3.set_yticks(range(0, 61, 5))
    ax3.set_zticks(range(0, 61, 5))
    
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    ax3.set_title(f"Phase Movement: {phase_name}")
    ax3.legend()
    
    # 调整视角让 3D 感更强
    ax3.view_init(elev=20, azim=45)

    plt.suptitle(f"Action Phase Visualization: {phase_name}", fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# --- 测试代码 ---
if __name__ == "__main__":
    # 模拟内参
    intrinsics = np.array([[600, 0, 320], [0, 600, 240], [0, 0, 1]])
    mapper = SpatialMapper(intrinsics=intrinsics)
    
    # 模拟一个在相机前 0.6m 的像素点
    test_world = mapper.pixel_to_world(320, 240, 0.6)
    print(f"物理坐标: {test_world}")
    
    test_grid = mapper.world_to_grid(test_world)
    print(f"网格索引: {test_grid}")
    
    # 测试路径填充
    path = generate_discrete_trajectory((2,2,2), (5,8,4))
    print(f"生成路径序列: {path}")