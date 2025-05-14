import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from sklearn.metrics import mean_squared_error
from joblib import load

# 倒立摆动力学方程
def inverted_pendulum(state, t, g, l, b, m, tau):
    theta, omega = state
    dtheta_dt = omega
    domega_dt = (g / l) * np.sin(theta) - \
        (b / m) * omega + tau / (m * l**2)
    return [dtheta_dt, domega_dt]

# 数据生成
def generate_data():
    g, l, b, m = 9.8, 1.0, 0.1, 1.0  # 参数
    t = np.linspace(0, 10, 1000)  # 时间
    tau = 0.0     # 控制力矩
    y0 = [0.0, 1]  # 初始条件 [初始角度， 初始角速度]
    omega0 = y0[1] # 初始角速度
    data = odeint(inverted_pendulum, y0, t, args=(g, l, b, m, tau))
    theta, omega = data[:, 0], data[:, 1]
    dtheta_dt = omega
    domega_dt = (g / l) * np.sin(theta) - (b / m) * omega\
        + tau / (m * l**2)
    X = np.column_stack((theta, omega, tau * np.ones_like(theta)))  # 输入
    y = domega_dt  # 输出
    return X, y, y0, t, tau, data

# 主程序
if __name__ == "__main__":
    
    X, y, y0, t, tau, data = generate_data()
    model = load(r'c:\Users\Administrator\36-AI\mlp_model.joblib')
    
    # 初始化状态
    time = 0
    omega0 = 0

    #记录演化过程
    theta_history = []
    omega_history = []
    time_history = []
    theta, omega, tau = y0[0], y0[1], tau
    # 预测演化
    for _ in range(1000):
        # 构造输入特征 [theta, omega, tau]
        X = np.array([[theta, omega, tau]])
        
        # 使用神经网络预测角加速度
        domega_dt = model.predict(X)[0]
        
        # 更新状态 (Euler法)
        omega += domega_dt * 0.01
        theta += omega * 0.01
        time += 0.01
        
        # 记录当前状态
        theta_history.append(theta)
        omega_history.append(omega)
        time_history.append(time)
    
    # 评估模型
    mse = mean_squared_error(data[:,0], theta_history)
    print(f"Mean Squared Error: {mse}")
    
    
    # 可视化
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(t, data[:, 0], label='Real Angle', color='blue')
    plt.plot(time_history, theta_history, label='Predict Angle', color='red', linestyle='--')
    plt.legend()
    plt.title('Angle Comparison')
    
    plt.subplot(2, 1, 2)
    plt.plot(t, data[:, 1], label='Real Angular Velocity', color='blue')
    plt.plot(time_history, omega_history, label='Predict Angular Velocity', color='red', linestyle='--')
    plt.legend()
    plt.title('Angular Velocity Comparison')
    
    plt.xlabel('Time (s)')
    plt.ylabel('Angle/Angle Velocity')
    plt.grid(alpha=0.3)
    plt.show()
    
    