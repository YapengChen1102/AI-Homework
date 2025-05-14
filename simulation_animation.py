import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# 定义 PID 控制器
class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = 0
        self.integral = 0

    def compute(self, setpoint, current_value, dt):
        error = setpoint - current_value
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        self.prev_error = error
        return self.kp * error + self.ki * self.integral + self.kd * derivative

# 模拟倒立摆
def simulate_inverted_pendulum(pid, theta0, omega0, setpoint, t, dt):
    theta = theta0
    omega = omega0
    theta_history = []
    for time in t:
        tau = pid.compute(setpoint, theta, dt)  # 计算控制力矩
        domega_dt = -9.8 * np.sin(theta) + tau  # 简化的倒立摆动力学
        omega += domega_dt * dt
        theta += omega * dt
        theta_history.append(theta)
    return theta_history

# 动画函数
def animate_pendulum(theta_history, t, save_path=None):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect('equal')
    ax.set_title('Inverted Pendulum Simulation with PID Control')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.grid(alpha=0.3)

    # 绘制摆的支点和杆
    pivot, = ax.plot(0, 0, 'ko')  # 支点
    rod, = ax.plot([], [], 'r-', lw=2)  # 杆

    # 更新动画帧
    def update(frame):
        theta = theta_history[frame]
        x = np.sin(theta)  # 杆末端的 x 坐标
        y = np.cos(theta)  # 杆末端的 y 坐标
        rod.set_data([0, x], [0, y])  # 更新杆的位置
        return rod,

    # 创建动画
    anim = FuncAnimation(fig, update, frames=len(theta_history), interval=20, blit=True)
    
    # 保存动画
    '''
    writer = PillowWriter(fps=30)
    anim.save('inverted_pendulum.gif', writer=writer)
    '''
    
    # 显示动画
    plt.show()

# 主程序
if __name__ == "__main__":
    # 时间参数
    t = np.linspace(0, 10, 1000)  # 模拟 10 秒
    dt = t[1] - t[0]

    # 初始条件
    theta0 = 0.1  # 初始角度 (弧度)
    omega0 = 0.0  # 初始角速度
    setpoint = 0.0  # 目标角度 (弧度)

    # 初始化 PID 控制器
    pid = PIDController(kp=52.074, ki=6.0893, kd=9.1494)

    # 模拟倒立摆
    theta_history = simulate_inverted_pendulum(pid, theta0, omega0, setpoint, t, dt)

    # 动画可视化
    animate_pendulum(theta_history, t)