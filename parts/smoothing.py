import numpy as np
import time

class OneEuroFilter:
    def __init__(self, t0, x0, dx0=0.0, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        """
        min_cutoff: 最小截止频率，越小越平滑，但延迟越高。建议 0.1 ~ 1.0
        beta: 速度系数，越大越能响应高速运动（减少延迟），但会有噪音。建议 0.001 ~ 0.1
        """
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev = np.array(x0, dtype=np.float64)
        self.dx_prev = np.array(dx0, dtype=np.float64)
        self.t_prev = float(t0)

    def smoothing_factor(self, t_e, cutoff):
        r = 2 * np.pi * cutoff * t_e
        return r / (r + 1)

    def exponential_smoothing(self, a, x, x_prev):
        return a * x + (1 - a) * x_prev

    def __call__(self, t, x):
        """
        t: 当前时间戳
        x: 当前观测值 (Pose 矩阵的平移部分 or 旋转四元数)
        """
        t_e = t - self.t_prev
        
        # 防止时间戳重复导致的除零错误
        if t_e <= 0.0:
            return self.x_prev

        a_d = self.smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = self.exponential_smoothing(a_d, dx, self.dx_prev)

        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = self.smoothing_factor(t_e, cutoff)
        x_hat = self.exponential_smoothing(a, x, self.x_prev)

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat

class GripperSmoother:
    def __init__(self):
        # 对位置 (Translation) 进行平滑
        self.pos_filter = None
        # 对旋转矩阵 (Rotation) 我们简化处理，也可以转四元数平滑，
        # 但简单的逐元素平滑在小角度变化时也凑合
        self.rot_filter = None 
        self.action_filter = None
        self.start_time = time.time()

    def update(self, pose_matrix, action_val, timestamp):
        current_time = timestamp
        
        # 分离位置和旋转
        current_pos = pose_matrix[:3, 3]
        current_rot = pose_matrix[:3, :3].flatten() # 简化：展平旋转矩阵
        
        # 初始化滤波器
        if self.pos_filter is None:
            # min_cutoff=0.5 (越小越稳), beta=0.05 (越大响应越快)
            self.pos_filter = OneEuroFilter(current_time, current_pos, min_cutoff=0.1, beta=0.05)
            self.rot_filter = OneEuroFilter(current_time, current_rot, min_cutoff=0.1, beta=0.05)
            self.action_filter = OneEuroFilter(current_time, action_val, min_cutoff=1.0, beta=0.0)
            return pose_matrix, action_val

        # 更新滤波器
        smooth_pos = self.pos_filter(current_time, current_pos)
        smooth_rot_flat = self.rot_filter(current_time, current_rot)
        smooth_action = self.action_filter(current_time, action_val)

        # 重组矩阵
        smooth_pose = np.eye(4)
        smooth_pose[:3, 3] = smooth_pos
        
        # 旋转矩阵正交化 (防止平滑导致矩阵变形)
        rot_mat = smooth_rot_flat.reshape(3, 3)
        u, _, vt = np.linalg.svd(rot_mat)
        clean_rot = np.dot(u, vt)
        smooth_pose[:3, :3] = clean_rot

        return smooth_pose, smooth_action