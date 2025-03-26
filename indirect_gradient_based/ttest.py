import gurobipy as gp

# 创建模型
model = gp.Model()

# 获取 Gurobi 版本
gurobi_version = gp.gurobi.version()

# 获取使用的 CPU 核心数
num_cores = model.Params.Threads

# 获取处理器速度（需要借助外部库）
import psutil
processor_speed = psutil.cpu_freq().current

print(f"Gurobi Version: {gurobi_version}")
print(f"Number of CPU Cores Used: {num_cores}")
print(f"Processor Speed: {processor_speed} MHz")