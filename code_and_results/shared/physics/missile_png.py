"""
三阶段分段制导导弹模型 — Baseline3 动态导弹版

升级特性：
  1. 支持动态初始位置和速度（从威胁生成器加载）
  2. 可选历史记录：减少内存分配开销
  3. 简化RK4：减少中间变量
"""

import numpy as np
from shared.config.params import PNG_PARAMS, REAL_TARGET


class PNGMissileFast:
    """
    加速版三阶段分段制导导弹（支持动态配置）
    """

    __slots__ = [
        'missile_num', 'pos', 'vel', 'target_pos',
        'N', 'n_max', 'a_max', 'drag_coeff', 'g_vec', 'R_safe',
        'FOV_max_rad', 'break_lock', 't', 'phase', 'landed', 'blind_time',
        'break_lock_time', 'break_lock_angle', '_record_history',
        'trajectory', 'velocity_history', 'phase_history', 'time_history'
    ]

    def __init__(self, missile_num, init_pos=None, init_vel=None, params=None, record_history=False):
        """
        Args:
            missile_num: 导弹编号（可以是任意整数）
            init_pos: 初始位置 [x, y, z]，如果为None则使用MISSILE_PARAMS
            init_vel: 初始速度 [vx, vy, vz]，如果为None则自动计算指向原点
            params: 参数字典
            record_history: 是否记录轨迹历史（默认False加速）
        """
        p = {**PNG_PARAMS, **(params or {})}

        self.missile_num = missile_num
        self.target_pos = REAL_TARGET["center"].copy()

        # 支持动态初始位置
        if init_pos is not None:
            self.pos = np.array(init_pos, dtype=np.float64)
        else:
            # 回退到静态配置（兼容Baseline2）
            from config.params import MISSILE_PARAMS
            self.pos = MISSILE_PARAMS["init_pos"][missile_num].astype(np.float64).copy()

        # 支持动态初始速度
        if init_vel is not None:
            self.vel = np.array(init_vel, dtype=np.float64)
        else:
            # 自动计算指向原点的速度
            init_dir = self.target_pos - self.pos
            from config.params import MISSILE_SPEED
            self.vel = (init_dir / np.linalg.norm(init_dir)) * MISSILE_SPEED

        # 制导参数
        self.N = p["N"]
        self.n_max = p["n_max"]
        self.a_max = self.n_max * p["g_scalar"]

        # 阻力系数 (预计算)
        self.drag_coeff = 0.5 * p["rho"] * p["S_ref"] * p["C_d"] / p["mass"]
        self.g_vec = p["g_vec"].copy()
        self.R_safe = p["R_base"] + p["R_kill"]

        # FOV
        self.FOV_max_rad = p["FOV_max_rad"]
        self.break_lock = False

        # 状态
        self.t = 0.0
        self.phase = 1
        self.landed = False
        self.blind_time = 0.0
        self.break_lock_time = None
        self.break_lock_angle = None

        # 历史记录（可选）
        self._record_history = record_history
        if record_history:
            self.trajectory = [self.pos.copy()]
            self.velocity_history = [self.vel.copy()]
            self.phase_history = [1]
            self.time_history = [0.0]
        else:
            self.trajectory = None
            self.velocity_history = None
            self.phase_history = None
            self.time_history = None

    def _compute_png_command(self, R_MT, vel):
        """PNG制导指令（内联优化）"""
        R_norm = np.linalg.norm(R_MT)
        V_norm = np.linalg.norm(vel)

        if R_norm < 1e-3 or V_norm < 1e-3:
            return np.zeros(3)

        R_unit = R_MT / R_norm
        Vc = np.dot(vel, R_unit)

        if Vc < 1e-3:
            return np.zeros(3)

        # omega_LOS = cross(R_MT, -vel) / R_norm^2
        omega_LOS = np.cross(R_MT, -vel) / (R_norm ** 2)
        u_V = vel / V_norm

        return self.N * Vc * np.cross(omega_LOS, u_V)

    def _check_fov_break_lock(self, vel, pos):
        """FOV脱锁检查"""
        R_LOS = self.target_pos - pos
        R_norm = np.linalg.norm(R_LOS)
        V_norm = np.linalg.norm(vel)

        if V_norm < 1e-3 or R_norm < 1e-3:
            return False, 0.0

        vel_dir = vel / V_norm
        los_dir = R_LOS / R_norm

        cos_angle = np.clip(np.dot(vel_dir, los_dir), -1.0, 1.0)
        angle_rad = np.arccos(cos_angle)

        return angle_rad > self.FOV_max_rad, np.degrees(angle_rad)

    def step(self, dt, is_blinded):
        """单步RK4积分"""
        if self.landed:
            return True

        # 阶段转换
        if is_blinded:
            if self.phase == 1 or self.phase == 3:
                self.phase = 2
        else:
            if self.phase == 2 and not self.break_lock:
                is_break, angle_deg = self._check_fov_break_lock(self.vel, self.pos)
                if is_break:
                    self.break_lock = True
                    self.break_lock_time = self.t
                    self.break_lock_angle = angle_deg
                else:
                    self.phase = 3

        effective_blinded = is_blinded or self.break_lock

        # RK4积分（简化版）
        pos, vel = self.pos, self.vel
        g = self.g_vec

        # k1
        a1 = self._compute_accel(pos, vel, effective_blinded)
        k1v, k1p = a1 * dt, vel * dt

        # k2
        pos2, vel2 = pos + 0.5 * k1p, vel + 0.5 * k1v
        a2 = self._compute_accel(pos2, vel2, effective_blinded)
        k2v, k2p = a2 * dt, vel2 * dt

        # k3
        pos3, vel3 = pos + 0.5 * k2p, vel + 0.5 * k2v
        a3 = self._compute_accel(pos3, vel3, effective_blinded)
        k3v, k3p = a3 * dt, vel3 * dt

        # k4
        pos4, vel4 = pos + k3p, vel + k3v
        a4 = self._compute_accel(pos4, vel4, effective_blinded)
        k4v, k4p = a4 * dt, vel4 * dt

        # 更新
        self.vel = vel + (k1v + 2*k2v + 2*k3v + k4v) / 6.0
        self.pos = pos + (k1p + 2*k2p + 2*k3p + k4p) / 6.0
        self.t += dt

        if is_blinded:
            self.blind_time += dt

        # 落地检测
        if self.pos[2] <= 0:
            self.pos[2] = 0.0
            self.landed = True

        # 记录历史
        if self._record_history:
            self.trajectory.append(self.pos.copy())
            self.velocity_history.append(self.vel.copy())
            self.phase_history.append(self.phase)
            self.time_history.append(self.t)

        return self.landed

    def _compute_accel(self, pos, vel, is_blinded):
        """计算总加速度"""
        # 重力
        a_gravity = self.g_vec

        # 阻力
        speed = np.linalg.norm(vel)
        a_drag = -self.drag_coeff * speed * vel if speed > 1e-6 else np.zeros(3)

        # 制导
        if is_blinded:
            a_autopilot = np.zeros(3)
        else:
            R_MT = self.target_pos - pos
            a_cmd = self._compute_png_command(R_MT, vel)
            a_autopilot_desired = a_cmd - a_gravity - a_drag

            # 饱和
            a_mag = np.linalg.norm(a_autopilot_desired)
            if a_mag > self.a_max and a_mag > 1e-6:
                a_autopilot = self.a_max * a_autopilot_desired / a_mag
            else:
                a_autopilot = a_autopilot_desired

        return a_autopilot + a_gravity + a_drag

    def simulate(self, smoke_checker=None, dt=None, max_time=120.0):
        """
        完整弹道仿真

        Args:
            smoke_checker: 烟幕遮蔽检查函数 (missile_num, t, pos) -> bool
            dt: 时间步长
            max_time: 最大仿真时间
        """
        if dt is None:
            dt = PNG_PARAMS["dt"]

        while self.t < max_time and not self.landed:
            is_blinded = False
            if smoke_checker is not None:
                is_blinded = smoke_checker(self.missile_num, self.t, self.pos)

            self.step(dt, is_blinded)

            # 提前终止：FOV脱锁
            if self.break_lock:
                return {
                    "miss_distance": 99999.0,
                    "total_time": self.t,
                    "blind_time": self.blind_time,
                    "break_lock": True,
                    "defense_success": True,
                }

        return self.get_result()

    def get_result(self):
        """返回仿真结果"""
        miss_distance = np.sqrt(self.pos[0]**2 + self.pos[1]**2)

        result = {
            "missile_num": self.missile_num,
            "landing_pos": self.pos.copy(),
            "miss_distance": miss_distance,
            "defense_success": miss_distance > self.R_safe,
            "blind_time": self.blind_time,
            "total_time": self.t,
            "break_lock": self.break_lock,
            "break_lock_time": self.break_lock_time,
            "break_lock_angle": self.break_lock_angle,
        }

        if self._record_history:
            result["trajectory"] = np.array(self.trajectory)
            result["velocities"] = np.array(self.velocity_history)

        return result


# 兼容性别名
PNGMissile = PNGMissileFast
