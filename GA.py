import numpy as np
from joblib import load
import random

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

# 使用神经网络模拟倒立摆系统
def simulate_inverted_pendulum_with_model(pid, model, theta0, omega0, setpoint, t, dt):
    theta = theta0
    omega = omega0
    total_error = 0
    for time in t:
        tau = pid.compute(setpoint, theta, dt)  # 计算控制力矩
        # 使用神经网络预测角加速度
        domega_dt = model.predict([[theta, omega, tau]])[0]  # 确保取出第一个预测结果
        omega += domega_dt * dt
        theta += omega * dt
        total_error += abs(setpoint - theta)  # 积分绝对误差 (IAE)
    return total_error

# 锦标赛选择
def tournament_selection(population, fitness, tournament_size=3):
    selected = []
    for _ in range(len(population)):
        # 随机选择若干个体
        indices = random.sample(range(len(population)), tournament_size)
        # 找到适应度最小的个体（误差最小）
        best_index = min(indices, key=lambda i: fitness[i])
        selected.append(population[best_index])
    return selected

# 遗传算法优化 PID 参数
def genetic_algorithm_with_model(model, setpoint, theta0, omega0, t, dt, population_size=20, generations=50, tournament_size=3):
    # 初始化种群 (随机生成 Kp, Ki, Kd 参数，范围为 [0, 100], [0, 10], [0, 10])
    population = [np.array([np.random.uniform(0, 30),  # Kp
                            np.random.uniform(0, 10),   # Ki
                            np.random.uniform(0, 10)])  # Kd
                  for _ in range(population_size)]

    for generation in range(generations):
        # 计算适应度 (目标函数值)
        fitness = []
        for individual in population:
            kp, ki, kd = individual
            pid = PIDController(kp, ki, kd)
            error = simulate_inverted_pendulum_with_model(pid, model, theta0, omega0, setpoint, t, dt)
            fitness.append(error)

        # 锦标赛选择
        selected_population = tournament_selection(population, fitness, tournament_size)

        # 交叉 (生成新个体)
        offspring = []
        while len(offspring) < population_size:
            parent1, parent2 = random.sample(selected_population, 2)
            crossover_point = random.randint(1, 2)
            child = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
            offspring.append(child)

        # 变异 (随机调整参数，确保范围限制)
        for individual in offspring:
            if random.random() < 0.1:  # 10% 概率变异
                individual += np.random.uniform([-5, -1, -1], [5, 1, 1], 3)
                individual[0] = np.clip(individual[0], 0, 100)  # 限制 Kp 在 [0, 100]
                individual[1] = np.clip(individual[1], 0, 10)   # 限制 Ki 在 [0, 10]
                individual[2] = np.clip(individual[2], 0, 10)   # 限制 Kd 在 [0, 10]

        # 更新种群
        population = offspring

        # 输出当前代的最佳适应度
        best_fitness = min(fitness)
        best_individual = population[fitness.index(best_fitness)]
        print(f"Generation {generation + 1}: Best Fitness = {best_fitness}, Best PID = {best_individual}")

    # 返回最优参数
    return best_individual

# 主程序
if __name__ == "__main__":
    
    # 载入模型
    try:
        model = load(r'c:\Users\Administrator\36-AI\mlp_model.joblib')
    except FileNotFoundError:
        print("Error: Model file not found. Please check the file path.")
        exit()
        
    # 时间参数
    t = np.linspace(0, 10, 1000)  # 模拟 10 秒
    dt = t[1] - t[0]

    # 初始条件
    theta0 = 0.1  # 初始角度 (弧度)
    omega0 = 0.0  # 初始角速度
    setpoint = 0.0  # 目标角度 (弧度)

    # 使用遗传算法优化 PID 参数
    best_pid = genetic_algorithm_with_model(model, setpoint, theta0, omega0, t, dt)
    print(f"Optimal PID Parameters: Kp = {best_pid[0]}, Ki = {best_pid[1]}, Kd = {best_pid[2]}")