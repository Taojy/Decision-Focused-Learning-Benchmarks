import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib.pyplot as plt
import math
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

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

# Read system parameters
topo = pd.read_excel('118nodes_system.xlsx', sheet_name='topology', index_col=None, header=None)  # Topology
unit = pd.read_excel('118nodes_system.xlsx', sheet_name='unit', header=0)  # Unit capacity, price, up/down reserves and reserve prices per node

reserve_up = 150
reserve_down = 150
nodes = 118
train_X_splits = np.array(pd.read_csv('X_train.csv', header=0))
test_X_splits = np.array(pd.read_csv('X_test.csv', header=0))
train_Y_splits = np.array(pd.read_csv('Y_train.csv', header=0))
test_Y_splits = np.array(pd.read_csv('Y_test.csv', header=0))
s = train_X_splits.shape[0]

# Create model
model = gp.Model()
P_c = model.addVars(unit.shape[0], s, lb=0, vtype=GRB.CONTINUOUS, name='P_c')  # Thermal generator output
P_l = model.addVars(s, lb=0, vtype=GRB.CONTINUOUS, name='P_w')  # Node load demand
R_U = model.addVars(unit.shape[0], s, lb=0, vtype=GRB.CONTINUOUS, name='R_U')  # Upward reserve capacity of thermal units
R_D = model.addVars(unit.shape[0], s, lb=0, vtype=GRB.CONTINUOUS, name='R_D')  # Downward reserve capacity of thermal units
r_U = model.addVars(unit.shape[0], s, lb=0, vtype=GRB.CONTINUOUS, name='r_U')  # Upward regulation
r_D = model.addVars(unit.shape[0], s, lb=0, vtype=GRB.CONTINUOUS, name='r_D')  # Downward regulation
P_lsh = model.addVars(topo.shape[0], s, lb=0, vtype=GRB.CONTINUOUS, name='P_lsh')  # Load shedding
theta_pl = model.addVars(train_X_splits.shape[1], lb=-1000, vtype=GRB.CONTINUOUS, name='theta_pl')  # Load forecast coefficients
delta_DA = model.addVars(topo.shape[0], s, lb=-180, ub=180, vtype=GRB.CONTINUOUS, name='delta_DA')
delta_RT = model.addVars(topo.shape[0], s, lb=-180, ub=180, vtype=GRB.CONTINUOUS, name='delta_RT')
ss = model.addVars(topo.shape[0], s, lb=0, vtype=GRB.CONTINUOUS, name='ss') 
b = model.addVar(lb=-10000, vtype=GRB.CONTINUOUS, name='b')

c = 9000  # Load shedding cost
c_s = 1750 

# Set objective function
model.setObjective(gp.quicksum(R_U[i,j]*unit.iloc[i,5] + R_D[i,j]*unit.iloc[i,6] + P_c[i,j]*unit.iloc[i,2] 
                + (r_U[i,j]-r_D[i,j])*unit.iloc[i,2] for i in range(unit.shape[0]) for j in range(s))
                +gp.quicksum(P_lsh[i,j] * c for i in range(topo.shape[0]) for j in range(s))
                +gp.quicksum(ss[i,j]*c_s for i in range(topo.shape[0]) for j in range(s)))

for k in range(s):
    # Set constraints
        # Reserve requirement constraints
    model.addConstr(gp.quicksum(R_U[i,k] for i in range(unit.shape[0])) == reserve_up)
    model.addConstr(gp.quicksum(R_D[i,k] for i in range(unit.shape[0])) == reserve_down)
        # Generator up/down reserve constraints
    for i in range(unit.shape[0]):
        model.addConstr(R_U[i,k] <= unit.iloc[i,3])
        model.addConstr(R_D[i,k] <= unit.iloc[i,4])
        # Day-ahead: power balance at each node
    for t in range(topo.shape[0]):
        model.addConstr(P_c[t,k]-P_l[k]*ratio_bus[t]
        -gp.quicksum(topo.iloc[t,j]*(delta_DA[t,k]-delta_DA[j,k]) for j in range(topo.shape[0])) == 0)       
    model.addConstr(delta_DA[0,k] == 0)  
        # Day-ahead: generator capacity constraints
    for i in range(unit.shape[0]):
        model.addConstr(P_c[i,k] <= unit.iloc[i,1]-R_U[i,k])
        model.addConstr(P_c[i,k] >= R_D[i,k])
        # Day-ahead: load demand constraint
    model.addConstr(P_l[k] == gp.quicksum(theta_pl[r] * train_X_splits[k][r] for r in range(train_X_splits.shape[1])) + b)
    model.addConstr(P_l[k] >= 0)
        # Real-time: power balance at each node
    for t in range(topo.shape[0]):
        model.addConstr(r_U[t,k]-r_D[t,k] - ss[t,k] - train_Y_splits[k][0]*ratio_bus[t] + P_l[k]*ratio_bus[t] + P_lsh[t,k]
        -gp.quicksum(topo.iloc[t,j]*(delta_RT[t,k]-delta_RT[j,k]) - topo.iloc[t,j]*(delta_DA[t,k]-delta_DA[j,k]) for j in range(topo.shape[0]))==0)
        # Real-time: reserve capacity constraints
    for i in range(unit.shape[0]):
        model.addConstr(r_U[i,k] <= R_U[i,k])
        model.addConstr(r_D[i,k] <= R_D[i,k])
    model.addConstr(delta_RT[0,k] == 0)  
        # Ensure load shedding is non-negative
    for t in range(topo.shape[0]):
        model.addConstr(P_lsh[t, k] >= 0)
        # Transmission line capacity constraints (day-ahead)
    for i in range(nodes):
        for j in range(nodes):
            if i != j and topo.iloc[i,j] < -0.1:
                model.addConstr(topo.iloc[i,j] * (delta_DA[i,k] - delta_DA[j,k]) <= 175)
                model.addConstr(topo.iloc[i,j] * (delta_DA[i,k] - delta_DA[j,k]) >= -175)
        # Transmission line capacity constraints (real-time)
    for i in range(nodes):
        for j in range(nodes):
            if i != j and topo.iloc[i,j] < -0.1:
                model.addConstr(topo.iloc[i,j] * (delta_RT[i,k] - delta_RT[j,k]) <= 175)
                model.addConstr(topo.iloc[i,j] * (delta_RT[i,k] - delta_RT[j,k]) >= -175)

model.optimize()

theta_result = np.zeros(train_X_splits.shape[1])
for i in range(train_X_splits.shape[1]):
    theta_result[i] = theta_pl[i].X

test_Y_splits = pd.read_csv('Y_test.csv', header=0)

pre_Y = np.zeros(len(test_Y_splits))
for i in range(len(test_Y_splits)):
    pre_Y[i] = sum(theta_result[j]*test_X_splits[i][j] for j in range(len(theta_result)))

# Combine pre_Y and test_Y_splits into a DataFrame
result_df = pd.DataFrame({
    'Actual': test_Y_splits.iloc[:, 0],  # Assuming test_Y_splits has only one column
    'Predicted': pre_Y
})

# Save results to CSV file
result_df.to_csv('测试集.csv', index=False)

print("Prediction results have been saved to predictions.csv")
