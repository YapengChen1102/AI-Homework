import numpy as np
import matplotlib.pyplot as plt
from test import inverted_pendulum
from joblib import load

# PID 控制器
class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = 0
        self.integral = 0

    def compute(self, setpoint, current, dt):
        error = setpoint - current
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        self.prev_error = error
        return self.kp * error + self.ki * self.integral + self.kd * derivative

# 模拟真实系统的动力学
def simulate_real_system(pid, theta0, omega0, setpoint, t, dt, g, l, b, m):
    theta = theta0
    omega = omega0
    theta_history = []
    for time in t:
        tau = pid.compute(setpoint, theta, dt)  # 计算控制力矩
        domega_dt = (g / l) * np.sin(theta) - \
        (b / m) * omega + tau / (m * l**2)  # 使用真实系统的动力学方程  
        omega += domega_dt * dt
        theta += omega * dt
        theta_history.append(theta)
    return theta_history

# 模拟基于神经网络模型的动力学
def simulate_model_system(pid, model, theta0, omega0, setpoint, t, dt):
    theta = theta0
    omega = omega0
    theta_history = []
    for time in t:
        tau = pid.compute(setpoint, theta, dt)  # 计算控制力矩
        domega_dt = model.predict([[theta, omega, tau]])[0]  # 使用神经网络预测角加速度
        omega += domega_dt * dt
        theta += omega * dt
        theta_history.append(theta)
    return theta_history

# 主程序
if __name__ == "__main__":
    # 时间参数
    t = np.linspace(0, 10, 1000)  # 模拟 10 秒
    dt = t[1] - t[0]

    # 系统参数
    g, l, b, m = 9.8, 1.0, 0.1, 1.0 
    
    # 初始条件
    theta0 = 0.1  # 初始角度 (弧度)
    omega0 = 0.0  # 初始角速度
    setpoint = 0.0  # 目标角度 (弧度)

    # 初始化 PID 控制器
    pid = PIDController(kp=29, ki=0.5, kd=6.25)

    # 加载神经网络模型
    try:
        model = load(r'c:\Users\Administrator\36-AI\mlp_model.joblib')
    except FileNotFoundError:
        print("神经网络模型未找到，请检查路径！")
        exit()

    # 模拟真实系统
    theta_history_real = simulate_real_system(pid, theta0, omega0, setpoint, t, dt, g, l, b, m)

    # 模拟基于神经网络模型的系统
    theta_history_model = simulate_model_system(pid, model, theta0, omega0, setpoint, t, dt)

    # 计算绝对误差
    absolute_error = np.abs(np.array(theta_history_real) - np.array(theta_history_model))

    # 计算平均绝对误差 (MAE)
    mean_absolute_error = np.mean(absolute_error)
    print(f"Mean Absolute Error (MAE): {mean_absolute_error:.4f}")

    # 可视化
    plt.figure(figsize=(12, 8))

    # 子图 1: 真实系统 vs 神经网络模型的角度对比
    plt.subplot(2, 1, 1)
    plt.plot(t, theta_history_real, label='Real System (PID)', color='blue', linestyle='-')
    plt.plot(t, theta_history_model, label='Model System (NN + PID)', color='red', linestyle='--')
    plt.xlabel('Time (s)')
    plt.ylabel('Angle (rad)')
    plt.title('Angle Comparison: Real System vs Model System')
    plt.legend()
    plt.grid(alpha=0.3)

    # 子图 2: 误差曲线
    error = np.array(theta_history_real) - np.array(theta_history_model)
    plt.subplot(2, 1, 2)
    plt.plot(t, error, label='Error (Real - Model)', color='green')
    plt.xlabel('Time (s)')
    plt.ylabel('Error (rad)')
    plt.title('Prediction Error')
    plt.legend()
    plt.grid(alpha=0.3)

    # 显示图像
    plt.tight_layout()
    plt.show()