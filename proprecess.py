import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
import missingno as msno  # 用于可视化缺失值

import matplotlib as mpl
mpl.use('TkAgg')  # 使用非交互式后端避免PyCharm兼容性问题

# 2. 设置字体支持 - 解决中文字符显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows中文字体
plt.rcParams['axes.unicode_minus'] = False    # 正确显示负号
# 根据图片数据创建数据集 (实际应用中替换为CSV读取)
df = pd.read_csv('data/diabetes_binary_health_indicators_BRFSS2015.csv')

column_order = [
    'Diabetes_binary', 'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke',
    'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies', 'HvyAlcoholConsump', 'AnyHealthcare',
    'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age'
]
df = df[column_order]

# 显示原始数据信息
print("原始数据摘要:")
print(f"总样本数: {len(df)}")
print(f"存在缺失值的特征数量: {df.isnull().any().sum()}")
print(f"总缺失值数量: {df.isnull().sum().sum()}")

# 缺失值处理（糖尿病分组KNN填补）

def group_knn_impute(df, group_col, features_to_impute, n_neighbors=3):
    """
    按照指定组别进行KNN缺失值填补

    参数:
        df (DataFrame): 原始数据
        group_col (str): 分组列名
        features_to_impute (list): 需要填补的特征列表
        n_neighbors (int): KNN邻居数

    返回:
        DataFrame: 填补完成的数据
    """
    # 检查分组列是否存在缺失值
    if df[group_col].isnull().any():
        raise ValueError(f"分组列 '{group_col}' 存在缺失值，无法分组")

    # 创建数据副本
    filled_df = df.copy()

    # 分离数值和非数值特征
    numeric_features = [f for f in features_to_impute if df[f].dtype in [np.int64, np.float64]]
    non_numeric_features = [f for f in features_to_impute if f not in numeric_features]

    scaler = StandardScaler()

    for group in df[group_col].unique():
        # 获取当前组的索引
        group_idx = df[df[group_col] == group].index

        if len(group_idx) == 0:
            continue

        # 只处理该组内的数据
        group_data = df.loc[group_idx, features_to_impute].copy()

        # 单独处理非数值特征
        for feat in non_numeric_features:
            if group_data[feat].isnull().any():
                # 非数值特征使用众数填补
                mode_val = group_data[feat].mode()
                if not mode_val.empty:
                    filled_df.loc[group_idx, feat] = group_data[feat].fillna(mode_val[0])

        # 处理数值特征 - 标准化
        if numeric_features:
            group_data_num = group_data[numeric_features].copy()
            group_data_num.loc[:, :] = scaler.fit_transform(group_data_num)

            # 应用KNN填补
            imputer = KNNImputer(n_neighbors=min(n_neighbors, max(1, len(group_idx) - 1)))
            imputed_data = imputer.fit_transform(group_data_num)

            # 将填补结果放回原始数据框
            filled_df.loc[group_idx, numeric_features] = scaler.inverse_transform(imputed_data)

    return filled_df


# 指定分组列和需要填补的特征
group_col = 'Diabetes_binary'
features_to_impute = ['Diabetes_binary', 'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke',
    'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies', 'HvyAlcoholConsump', 'AnyHealthcare',
    'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age']

# 执行分组填补
filled_df = group_knn_impute(df, group_col, features_to_impute, n_neighbors=3)

print("\n填补后的数据摘要:")
print(f"总缺失值数量: {filled_df.isnull().sum().sum()}")

# ================================
# 数据保存
# ================================

# 确保数据目录存在
os.makedirs('data', exist_ok=True)

# 保存原始数据和填补后的数据
df.to_csv('data/original_diabetes_data.csv', index=False)
filled_df.to_csv('data/filled_diabetes_data.csv', index=False)

print("\n数据保存成功:")
print("原始数据保存至: data/original_diabetes_data.csv")
print("填补后数据保存至: data/filled_diabetes_data.csv")

# ================================
# 缺失值可视化
# ================================

# 创建可视化结果目录
os.makedirs('visualizations', exist_ok=True)

# 1. 原始数据缺失值矩阵图
plt.figure(figsize=(12, 8))
msno.matrix(df, fontsize=12)
plt.title('原始数据缺失情况', fontsize=16)
plt.savefig('visualizations/missing_values_before_imputation.png', dpi=300, bbox_inches='tight')

# 2. 填补后数据缺失值矩阵图
plt.figure(figsize=(12, 8))
msno.matrix(filled_df, fontsize=12)
plt.title('填补后数据缺失情况', fontsize=16)
plt.savefig('visualizations/missing_values_after_imputation.png', dpi=300, bbox_inches='tight')

# 3. 缺失值条形图对比
plt.figure(figsize=(14, 8))
plt.subplot(1, 2, 1)
msno.bar(df, fontsize=12)
plt.title('原始数据各特征缺失比例', fontsize=14)

plt.subplot(1, 2, 2)
msno.bar(filled_df, fontsize=12)
plt.title('填补后数据各特征缺失比例', fontsize=14)
plt.tight_layout()
plt.savefig('visualizations/missing_values_comparison.png', dpi=300)

print("\n缺失值可视化结果已保存至 visualizations 目录")

# 显示缺失值填补前后的数据对比表格
print("\n原始数据前5行（部分特征）：")
print(df[['Diabetes_binary', 'HighBP', 'HighChol', 'Smoker', 'MentHlth', 'PhysHlth', 'Sex', 'Age']].head())

print("\n填补后数据前5行（部分特征）：")
print(filled_df[['Diabetes_binary', 'HighBP', 'HighChol', 'Smoker', 'MentHlth', 'PhysHlth', 'Sex', 'Age']].head())

# ================================
# 数据分析报告
# ================================
print("\n=== 数据分析报告 ===")
for col in features_to_impute:
    before_missing = df[col].isnull().sum()
    after_missing = filled_df[col].isnull().sum()
    if before_missing > 0:
        print(f"特征 '{col}':")
        print(f"  原始缺失值数: {before_missing} ({before_missing / len(df) * 100:.1f}%)")
        print(f"  填补后缺失值数: {after_missing}")
        print(f"  填补比例: {(before_missing - after_missing) / before_missing * 100:.1f}%")

        # 显示缺失值变化示例
        missing_idx = df[df[col].isnull()].index
        if any(missing_idx):
            idx = missing_idx[0]
            print(f"  示例填补: 样本#{idx}: 原始={df.at[idx, col]} → 填补后={filled_df.at[idx, col]}")