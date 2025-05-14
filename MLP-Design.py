import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from joblib import dump
import multiprocessing as mp

# 倒立摆动力学方程
def inverted_pendulum(state, t, g, l, b, m, tau):
    theta, omega = state
    dtheta_dt = omega
    domega_dt = (g / l) * np.sin(theta) - (b / m) * omega + tau / (m * l**2)
    return [dtheta_dt, domega_dt]

# 单组初始条件的数据生成
def generate_single_condition(theta0,omega0, tau, t, g, l, b, m):
    y0 = [theta0, omega0]  # 初始条件 [初始角度，初始角速度]
    data = odeint(inverted_pendulum, y0, t, args=(g, l, b, m, tau))
    theta, omega = data[:, 0], data[:, 1]
    dtheta_dt = omega
    domega_dt = (g / l) * np.sin(theta) - (b / m) * omega + tau / (m * l**2)
    
    # 构造输入和输出
    X = np.column_stack((theta, omega, tau * np.ones_like(theta)))  # 输入
    y = domega_dt  # 输出
    return X, y

# 数据生成（并行版本）
def generate_data():
    g, l, b, m = 9.8, 1.0, 0.1, 1.0  # 参数
    t = np.linspace(0, 10, 1000)  # 时间

    # 定义所有初始条件的组合
    conditions = [(theta0,omega0,tau) for theta0 in [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1]
                  for tau in [0, 0.1, 0.5, 1.0, 5, 10, 20] for omega0 in [0, 0.1, 0.5]]

    # 使用多进程并行计算
    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.starmap(generate_single_condition, [(theta0,omega0, tau, t, g, l, b, m) for theta0,omega0, tau in conditions])

    # 合并所有结果
    X_list, y_list = zip(*results)
    X = np.vstack(X_list)
    y = np.hstack(y_list)
    return X, y, t

# 主程序
if __name__ == "__main__":
    # 生成数据
    X, y, t = generate_data()

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=0.2, random_state=42)

    # 神经网络建模
    model = MLPRegressor(hidden_layer_sizes=(256,128,64,32),\
        activation='relu', max_iter=5000, random_state=42)
    model.fit(X_train, y_train)

    # 保存模型
    dump(model, r'c:\Users\Administrator\36-AI\mlp_model.joblib')
    print("模型已保存到文件 mlp_model.joblib")

    # 预测
    y_pred = model.predict(X_test)
    
    # 评估
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")