import numpy as np
import pandas as pd
import os

# 定义路径
fit_results_path = './global_piecewise_linear_fit_results_continuous.csv'  # 分段线性拟合结果
smooth_results_path = './smooth_breakpoints_results.csv'  # 平滑化结果
output_dir = './predictions'  # 预测结果输出文件夹
model_params_dir = './model_params'  # 模型参数保存文件夹
epsilon_dir = './epsilon_values'  # 保存迭代过程中的 epsilon 文件夹

# 创建输出文件夹
os.makedirs(output_dir, exist_ok=True)
os.makedirs(model_params_dir, exist_ok=True)
os.makedirs(epsilon_dir, exist_ok=True)

# 读取分段线性拟合和平滑化结果
fit_results_df = pd.read_csv(fit_results_path)
smooth_results_df = pd.read_csv(smooth_results_path)

# 读取训练和测试数据
X_train = pd.read_csv('./X_train.csv').values
y_train = pd.read_csv('./Y_train.csv').values.reshape(-1, 1)
X_test = pd.read_csv('./X_test.csv').values
y_test = pd.read_csv('./Y_test.csv').values.reshape(-1, 1)

# 对 X 进行最小-最大归一化到 [-1,1]
X_train_min = np.min(X_train, axis=0)
X_train_max = np.max(X_train, axis=0)
X_train = 2 * (X_train - X_train_min) / (X_train_max - X_train_min) - 1
X_test = 2 * (X_test - X_train_min) / (X_train_max - X_train_min) - 1

# 在 X_train 和 X_test 最后一列添加全 1 作为偏置项
X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])

# 超参数
delta = 0.0001  # 分段阈值
gamma = 0.001  # 学习率衰减率
max_iter = 1000  # 最大迭代次数
eps = 0  # 避免除零错误


# 计算分段梯度
def piecewise_gradient(epsilon_i, delta, a_k, breakpoints, smooth_results_df):
    """ 计算样本 i 的梯度 """
    grad = 0
    segment_index = np.digitize([epsilon_i], breakpoints)[0] - 1  # 计算该点在哪个区间
    segment_index = int(np.clip(segment_index, 0, len(a_k) - 1))  # 确保是整数索引

    # 检查是否在分段点附近
    if np.abs(epsilon_i - breakpoints[segment_index]) < delta:
        # 查询二次函数的平滑参数
        smooth_params = smooth_results_df[smooth_results_df['Breakpoint'] == breakpoints[segment_index]]
        if not smooth_params.empty:
            a_quad = smooth_params['A'].values[0]
            b_quad = smooth_params['B'].values[0]
            # 计算二次函数梯度
            grad = 2 * a_quad * epsilon_i + b_quad
    else:
        # 普通分段内，梯度取 a_k[segment_index]
        grad = a_k[segment_index]
    return grad


# 计算误差
def evaluate(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    return mse, rmse, mae


# 训练和预测函数
def train_and_predict():
    """ 对整体数据进行训练和预测 """
    a_k = fit_results_df['Slope'].values  # 斜率 a_k
    breakpoints = fit_results_df['Breakpoint_Start'].values  # 分段点

    # 初始化参数 w
    n, d = X_train.shape
    w = np.zeros((d, 1))
    eta = 6  # 初始学习率

    # 训练过程
    for t in range(1, max_iter + 1):
        grad_w = np.zeros_like(w)
        for i in range(n):  # 遍历所有样本
            epsilon_i = (X_train[i] @ w - y_train[i]) / y_train[i]
            grad_L_i = piecewise_gradient(epsilon_i, delta, a_k, breakpoints, smooth_results_df)
            grad_w += (grad_L_i / y_train[i]) * X_train[i].reshape(-1, 1)
        w -= eta * grad_w  # 梯度更新
        eta = eta / (1 + gamma * t)  # 学习率衰减

        # 每 100 轮输出一次进程信息
        if t % 100 == 0:
            y_pred_train = X_train @ w
            mse_train, _, _ = evaluate(y_train, y_pred_train)
            print(f"迭代 {t}: 训练集 MSE={mse_train:.4f}")
            np.save(os.path.join(model_params_dir, f'w_iter_{t}.npy'), w)

    # 训练集和测试集预测
    y_pred_train = X_train @ w
    y_pred_test = X_test @ w

    # 计算最终评价指标
    mse_train, rmse_train, mae_train = evaluate(y_train, y_pred_train)
    mse_test, rmse_test, mae_test = evaluate(y_test, y_pred_test)
    print(f"最终训练集: MSE={mse_train:.4f}, RMSE={rmse_train:.4f}, MAE={mae_train:.4f}")
    print(f"最终测试集: MSE={mse_test:.4f}, RMSE={rmse_test:.4f}, MAE={mae_test:.4f}")

    # 保存预测结果
    pd.DataFrame({"y_true": y_train.flatten(), "y_pred": y_pred_train.flatten()}).to_csv(
        os.path.join(output_dir, 'predictions_train.csv'), index=False)
    pd.DataFrame({"y_true": y_test.flatten(), "y_pred": y_pred_test.flatten()}).to_csv(
        os.path.join(output_dir, '5.5+.csv'), index=False)

    # 保存最终模型参数
    np.save(os.path.join(model_params_dir, 'w_final.npy'), w)


# 训练和预测
train_and_predict()