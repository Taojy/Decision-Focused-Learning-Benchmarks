import numpy as np
import pandas as pd
import statsmodels.api as sm
import time

# 在代码开始处记录开始时间
start_time = time.time()

# 数据加载
X_train = pd.read_csv('X_train.csv')
y_train = pd.read_csv('Y_train.csv').squeeze()
X_test = pd.read_csv('X_test.csv')
y_test = pd.read_csv('Y_test.csv').squeeze()

# 确保 X_train 和 X_test 是 DataFrame
if not isinstance(X_train, pd.DataFrame):
    X_train = pd.DataFrame(X_train)
if not isinstance(X_test, pd.DataFrame):
    X_test = pd.DataFrame(X_test)

# 添加常数项
X_train = sm.add_constant(X_train, has_constant='add')
X_test = sm.add_constant(X_test, has_constant='add')

# 检查特征维度
print("X_train 形状:", X_train.shape)  # 应该是 (n_samples, 15)
print("X_test 形状:", X_test.shape)    # 应该是 (m_samples, 15)

# 设置迭代次数和分位数
iter_num = 1000
q = [0.15 + 0.05 * i for i in range(1,14,1)]

# 训练模型并进行预测
quan_models = [sm.QuantReg(y_train, X_train).fit(q_i, max_iter=iter_num) for q_i in q]
y_predict = [model.predict(X_test) for model in quan_models]

# 将预测结果转换为DataFrame
y_predict_df = pd.DataFrame(np.column_stack(y_predict), columns=[f'q_{q_i:.1f}' for q_i in q])

# 将预测结果保存到CSV文件
y_predict_df.to_csv('predictions.csv', index=False)

print("预测结果已保存到 predictions.csv")

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib.pyplot as plt
import math
from sklearn.metrics import mean_squared_error,mean_absolute_percentage_error

mpc_bus = [
    [1, 51], [2, 20], [3, 39], [4, 39], [5, 0], [6, 52], [7, 19], [8, 28], [9, 0], [10, 0],
    [11, 70], [12, 47], [13, 34], [14, 14], [15, 90], [16, 25], [17, 11], [18, 60], [19, 45], [20, 18],
    [21, 14], [22, 10], [23, 7], [24, 13], [25, 0], [26, 0], [27, 71], [28, 17], [29, 24], [30, 0],
    [31, 43], [32, 59], [33, 23], [34, 59], [35, 33], [36, 31], [37, 0], [38, 0], [39, 27], [40, 66],
    [41, 37], [42, 96], [43, 18], [44, 16], [45, 53], [46, 28], [47, 34], [48, 20], [49, 87], [50, 17],
    [51, 17], [52, 18], [53, 23], [54, 113], [55, 63], [56, 84], [57, 12], [58, 12], [59, 277], [60, 78],
    [61, 0], [62, 77], [63, 0], [64, 0], [65, 0], [66, 39], [67, 28], [68, 0], [69, 0], [70, 66],
    [71, 0], [72, 12], [73, 6], [74, 68], [75, 47], [76, 68], [77, 61], [78, 71], [79, 39], [80, 130],
    [81, 0], [82, 54], [83, 20], [84, 11], [85, 24], [86, 21], [87, 0], [88, 48], [89, 0], [90, 163],
    [91, 10], [92, 65], [93, 12], [94, 30], [95, 42], [96, 38], [97, 15], [98, 34], [99, 42], [100, 37],
    [101, 22], [102, 5], [103, 23], [104, 38], [105, 31], [106, 43], [107, 50], [108, 2], [109, 8], [110, 39],
    [111, 0], [112, 68], [113, 6], [114, 8], [115, 22], [116, 184], [117, 20], [118, 33]
]
ratio_bus = np.zeros(118)
for i in range(118):
    ratio_bus[i] = mpc_bus[i][1]/sum(mpc_bus[j][1] for j in range(118))
print(ratio_bus)
print(sum(ratio_bus[i] for i in range(118)))
print(sum(mpc_bus[j][1] for j in range(118)))

# 读取系统参数
topo = pd.read_excel('118nodes_system.xlsx', sheet_name='topology', index_col=None, header=None) # 拓扑结构

unit = pd.read_excel('118nodes_system.xlsx', sheet_name='unit', header=0) # 每个节点机组容量、电价、上/下行备用，上/下行价格

reserve_up = 150
reserve_down = 150
nodes = 118
test_Y_splits = np.array(pd.read_csv('predictions.csv', header=0))
real_Y_splits = np.array(pd.read_csv('Y_test.csv', header=0))
print(unit.shape[0])
s = test_Y_splits.shape[0]
n = len(q)
print(real_Y_splits[100][0]*ratio_bus[1])

# 建立模型
model = gp.Model()
P_c = model.addVars(unit.shape[0], s, lb=0, vtype=GRB.CONTINUOUS, name='P_c') # 火电机组出力
R_U = model.addVars(unit.shape[0], s, lb=0, vtype=GRB.CONTINUOUS, name='R_U') # 火电机组向上备用容量
R_D = model.addVars(unit.shape[0], s, lb=0, vtype=GRB.CONTINUOUS, name='R_D') # 火电机组向下备用容量
r_U = model.addVars(unit.shape[0], s, n, lb=0, vtype=GRB.CONTINUOUS, name='r_U') # 火电机组上调功率
r_D = model.addVars(unit.shape[0], s, n, lb=0, vtype=GRB.CONTINUOUS, name='r_D') # 火电机组下调功率
P_lsh = model.addVars(topo.shape[0], s, n, lb=0, vtype=GRB.CONTINUOUS, name='P_lsh') # 切负荷
delta_DA = model.addVars(topo.shape[0], s, n, lb=-180, ub=180, vtype=GRB.CONTINUOUS, name='delta_DA')
delta_RT = model.addVars(topo.shape[0], s, n, lb=-180, ub=180, vtype=GRB.CONTINUOUS, name='delta_RT')
ss = model.addVars(topo.shape[0], s, n, lb=0, vtype=GRB.CONTINUOUS, name='ss') 

c = 9000 #切负荷成本
c_s = 1750 

# 设置目标函数
model.setObjective(gp.quicksum(R_U[i,j]*unit.iloc[i,5] + R_D[i,j]*unit.iloc[i,6] + P_c[i,j]*unit.iloc[i,2] for i in range(unit.shape[0]) for j in range(s))
                + gp.quicksum((r_U[i,j,l]-r_D[i,j,l])*unit.iloc[i,2] +(P_lsh[i,j,l] * c) + (ss[i,j,l]*c_s) for i in range(topo.shape[0]) for j in range(s) for l in range(n))/n, GRB.MINIMIZE) 

for k in range(s):
    # 设置约束
        # 备用需求约束
    model.addConstr(gp.quicksum(R_U[i,k] for i in range(unit.shape[0])) == reserve_up)
    model.addConstr(gp.quicksum(R_D[i,k] for i in range(unit.shape[0])) == reserve_down)
        # 机组上下行备用约束
    for i in range(unit.shape[0]):
        model.addConstr(R_U[i,k] <= unit.iloc[i,3])
        model.addConstr(R_D[i,k] <= unit.iloc[i,4])
        # 日前：每个节点的功率平衡约束
    for l in range(n):
        for t in range(topo.shape[0]):
            model.addConstr(P_c[t,k]-test_Y_splits[k][l]*ratio_bus[t]
            -gp.quicksum(topo.iloc[t,j]*(delta_DA[t,k,l]-delta_DA[j,k,l]) for j in range(topo.shape[0])) >= 0)       
        model.addConstr(delta_DA[0,k,l] == 0)
        # 日前：机组容量约束
    for i in range(unit.shape[0]):
        model.addConstr(P_c[i,k] <= unit.iloc[i,1]-R_U[i,k])
        model.addConstr(P_c[i,k] >= R_D[i,k])
        # 实时：每个节点的功率平衡约束
    for l in range(n):
        for t in range(topo.shape[0]):
            model.addConstr(P_c[t,k] + r_U[t,k,l] - r_D[t,k,l] - ss[t,k,l] - test_Y_splits[k][l]*ratio_bus[t]  + P_lsh[t,k,l]
            -gp.quicksum(topo.iloc[t,j]*(delta_RT[t,k,l]-delta_RT[j,k,l]) for j in range(topo.shape[0]))==0)
        # 实时：备用容量约束
        for i in range(unit.shape[0]):
            model.addConstr(r_U[i,k,l] <= R_U[i,k])
            model.addConstr(r_D[i,k,l] <= R_D[i,k])
        model.addConstr(delta_RT[0,k,l] == 0)  
        # 添加切负荷量大于等于0的约束
        for t in range(topo.shape[0]):
            model.addConstr(P_lsh[t,k,l] >= 0)  # 确保切负荷量大于等于0
        # 线路容量约束
        for i in range(nodes):
            for j in range(nodes):
                if i != j and topo.iloc[i,j] < -0.1:
                    model.addConstr(topo.iloc[i,j] * (delta_DA[i,k,l] - delta_DA[j,k,l]) <= 175)
                    model.addConstr(topo.iloc[i,j] * (delta_DA[i,k,l] - delta_DA[j,k,l]) >= -175)
        # 线路容量约束
        for i in range(nodes):
            for j in range(nodes):
                if i != j and topo.iloc[i,j] < -0.1:
                    model.addConstr(topo.iloc[i,j] * (delta_RT[i,k,l] - delta_RT[j,k,l]) <= 175)
                    model.addConstr(topo.iloc[i,j] * (delta_RT[i,k,l] - delta_RT[j,k,l]) >= -175)

# model.setParam("FeasibilityTol", 1e-9) # 设置可行性容忍度
# model.setParam("OptimalityTol", 1e-9) # 设置最优性容忍度
# model.setParam("MIPGap", 1e-18) # 设置目标间隙
# model.setParam("NumericFocus", 3) # 设置更高的数值精度    
model.optimize()
# model.computeIIS()
# model.write("model1.ilp")

# 存储
import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB


def save_optimization_results(model, model2, time_steps, nodes, unit, topo, c=9000, c_s=1750,
                              filename='optimization_results_GB_test.xlsx'):
    # Create DataFrames to store the results
    da_results = []
    rt_results = []
    objective_results = []

    # Day-ahead results
    for k in range(time_steps):
        for i in range(nodes):
            P_c = model.getVarByName(f'P_c[{i},{k}]').X
            R_U = model.getVarByName(f'R_U[{i},{k}]').X
            R_D = model.getVarByName(f'R_D[{i},{k}]').X
            delta_DA = model.getVarByName(f'delta_DA[{i},{k}]').X

            da_results.append([
                'DA', k, i, P_c, R_U, R_D, delta_DA
            ])

    # Real-time results
    for k in range(time_steps):
        for i in range(unit.shape[0]):
            r_U = model2.getVarByName(f'r_U[{i},{k}]').X
            r_D = model2.getVarByName(f'r_D[{i},{k}]').X
            P_lsh = model2.getVarByName(f'P_lsh[{i},{k}]').X
            delta_RT = model2.getVarByName(f'delta_RT[{i},{k}]').X
            ss = model2.getVarByName(f'ss[{i},{k}]').X

            rt_results.append([
                'RT', k, i, r_U, r_D, P_lsh, delta_RT, ss
            ])

    # Objective function results
    for k in range(time_steps):
        # Day-ahead cost
        obj_DA = sum(
            model.getVarByName(f'R_U[{i},{k}]').X * unit.iloc[i, 5] +
            model.getVarByName(f'R_D[{i},{k}]').X * unit.iloc[i, 6] +
            model.getVarByName(f'P_c[{i},{k}]').X * unit.iloc[i, 2]
            for i in range(nodes)
        )
        # Real-time cost
        obj_RT = sum(
            (model2.getVarByName(f'r_U[{i},{k}]').X - model2.getVarByName(f'r_D[{i},{k}]').X) * unit.iloc[i, 2] +
            model2.getVarByName(f'P_lsh[{i},{k}]').X * c +
            model2.getVarByName(f'ss[{i},{k}]').X * c_s
            for i in range(unit.shape[0])
        )
        objective_results.append([k, obj_DA, obj_RT])

    # Convert to DataFrames
    da_columns = ['Stage', 'TimeStep', 'Node', 'P_c', 'R_U', 'R_D', 'delta_DA']
    rt_columns = ['Stage', 'TimeStep', 'Node', 'r_U', 'r_D', 'P_lsh', 'delta_RT', 'ss']
    objective_columns = ['TimeStep', 'Objective_DA', 'Objective_RT']

    df_da = pd.DataFrame(da_results, columns=da_columns)
    df_rt = pd.DataFrame(rt_results, columns=rt_columns)
    df_objective = pd.DataFrame(objective_results, columns=objective_columns)

    # Save to Excel with multiple sheets
    with pd.ExcelWriter(filename) as writer:
        df_da.to_excel(writer, sheet_name='DA_Results', index=False)
        df_rt.to_excel(writer, sheet_name='RT_Results', index=False)
        df_objective.to_excel(writer, sheet_name='Objective_Results', index=False)


# 建立模型
model2 = gp.Model()
r_U = model2.addVars(unit.shape[0], s, lb=0, vtype=GRB.CONTINUOUS, name='r_U')  # 火电机组上调功率
r_D = model2.addVars(unit.shape[0], s, lb=0, vtype=GRB.CONTINUOUS, name='r_D')  # 火电机组下调功率
P_lsh = model2.addVars(topo.shape[0], s, lb=0, vtype=GRB.CONTINUOUS, name='P_lsh')  # 切负荷
delta_RT = model2.addVars(topo.shape[0], s, lb=-180, ub=180, vtype=GRB.CONTINUOUS, name='delta_RT')
ss = model2.addVars(topo.shape[0], s, lb=0, vtype=GRB.CONTINUOUS, name='ss')
c = 9000  # 切负荷成本
c_s = 1750

# 设置目标函数
model2.setObjective(
    gp.quicksum(R_U[i, j].x * unit.iloc[i, 5] + R_D[i, j].x * unit.iloc[i, 6] + P_c[i, j].x * unit.iloc[i, 2] for i in
                range(unit.shape[0]) for j in range(s))
    + gp.quicksum(
        (r_U[i, j] - r_D[i, j]) * unit.iloc[i, 2] + (P_lsh[i, j] * c) + (ss[i, j] * c_s) for i in range(topo.shape[0])
        for j in range(s)),
    GRB.MINIMIZE
)

# 设置约束
for k in range(s):
    # 实时：每个节点的功率平衡约束
    for t in range(topo.shape[0]):
        model2.addConstr(
            P_c[t, k].x + r_U[t, k] - r_D[t, k] - ss[t, k] - real_Y_splits[k][0] * ratio_bus[t] + P_lsh[t, k]
            - gp.quicksum(topo.iloc[t, j] * (delta_RT[t, k] - delta_RT[j, k]) for j in range(topo.shape[0])) == 0
        )
    # 实时：备用容量约束
    for i in range(unit.shape[0]):
        model2.addConstr(r_U[i, k] <= R_U[i, k].x)
        model2.addConstr(r_D[i, k] <= R_D[i, k].x)
    model2.addConstr(delta_RT[0, k] == 0)
    # 添加切负荷量大于等于0的约束
    for t in range(topo.shape[0]):
        model2.addConstr(P_lsh[t, k] >= 0)  # 确保切负荷量大于等于0
    # 线路容量约束
    for i in range(nodes):
        for j in range(nodes):
            if i != j and topo.iloc[i, j] < -0.1:
                model2.addConstr(topo.iloc[i, j] * (delta_RT[i, k] - delta_RT[j, k]) <= 175)
                model2.addConstr(topo.iloc[i, j] * (delta_RT[i, k] - delta_RT[j, k]) >= -175)

# 优化模型
model2.optimize()

# 保存优化结果
save_optimization_results(model, model2, s, nodes, unit, topo, c, c_s, 'optimization_results_GB_test.xlsx')

if model2.status == GRB.Status.OPTIMAL:
    # 输出目标函数值
    objective_value = model2.ObjVal
    print(f"优化完成，目标函数的最优值为: {objective_value}")

objective_value = model2.ObjVal
# 在代码结束处记录结束时间
end_time = time.time()

# 计算运行时间
run_time = end_time - start_time

# 将运行时间添加到数据中
data = {'Objective Value': [objective_value], 'Run Time (seconds)': [run_time]}
df = pd.DataFrame(data)
df.to_csv('objective_value.csv',index=False)

