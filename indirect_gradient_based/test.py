import pandas as pd
import numpy as np
import os


# 定义函数进行单小时预测
def predict_hourly_y(hour, X_test, theta, beta):
    # 使用 theta 和 beta 进行线性预测
    y_pred = np.dot(X_test, theta) + beta
    return y_pred


# 定义函数将预测值按原始顺序重新组合
def combine_predictions(predictions, predictions_lengths):
    # 计算总行数
    total_rows = sum(predictions_lengths)
    # 初始化一个全零数组，用于存储组合后的预测值
    combined_y_pred = np.zeros(total_rows)

    # 初始化一个指针，用于定位插入位置
    pointer = 0

    # 遍历 24 小时
    for hour in range(24):
        # 获取当前小时的预测值
        y_pred = predictions[hour]
        # 获取当前小时的预测值长度
        length = predictions_lengths[hour]
        # 将预测值插入到正确的位置
        combined_y_pred[pointer:pointer + length] = y_pred
        # 更新指针
        pointer += length

    return combined_y_pred


# 主函数
def predict_and_combine():
    # 加载训练好的 theta 和 beta 值
    theta_results_df = pd.read_csv('hourly_theta_results.csv', index_col='Hour')
    beta_results_df = pd.read_csv('hourly_beta_results.csv', index_col='Hour')

    # 初始化一个列表，用于存储所有小时的预测值
    all_predictions = []
    # 初始化一个列表，用于存储每个小时的预测值长度
    predictions_lengths = []

    # 遍历 24 小时
    for hour in range(1, 25):
        # 加载对应小时的测试数据
        X_test_file = f'./test_split/X_test_hour_{hour}.csv'
        X_test = pd.read_csv(X_test_file)

        # 获取对应小时的 theta 和 beta 值
        theta = theta_results_df.loc[hour].values
        beta = beta_results_df.loc[hour, 'Beta']

        # 进行预测
        y_pred = predict_hourly_y(hour, X_test, theta, beta)

        # 将预测值添加到列表中
        all_predictions.append(y_pred)
        # 记录当前小时的预测值长度
        predictions_lengths.append(len(y_pred))

    # 将预测值按原始顺序重新组合
    combined_y_pred = combine_predictions(all_predictions, predictions_lengths)

    # 将组合后的预测值保存为 CSV 文件
    combined_y_pred_df = pd.DataFrame(combined_y_pred, columns=['Predicted_Y'])
    combined_y_pred_df.to_csv('combined_y_test_predictions.csv', index=False)
    print("组合后的 Y_test 预测值已保存到 combined_y_test_predictions.csv 文件中。")


# 执行主函数
predict_and_combine()