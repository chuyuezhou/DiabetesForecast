import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


mpl.use('TkAgg')  # 使用非交互式后端避免PyCharm兼容性问题

# 2. 设置字体支持 - 解决中文字符显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows中文字体
plt.rcParams['axes.unicode_minus'] = False    # 正确显示负号

# 读取数据 (请替换为您的数据集路径)
df = pd.read_csv('data/diabetes_binary_health_indicators_BRFSS2015.csv')

# 创建与图片匹配的列名映射 (图片中的缩写格式)
column_mapping = {
    'Diabetes_binary': 'Diabetes',
    'HighBP': 'HighBP',
    'HighChol': 'HighChol',
    'CholCheck': 'CholCheck',
    'BMI': 'BMI',
    'Smoker': 'Smoker',
    'Stroke': 'Stroke',
    'HeartDiseaseorAttack': 'HeartDise',
    'PhysActivity': 'PhysActivi',
    'Fruits': 'Fruits',
    'Veggies': 'Veggies',
    'HvyAlcoholConsump': 'HvyAlcohol',
    'AnyHealthcare': 'AnyHealth',
    'NoDocbcCost': 'NoDocbc',
    'GenHlth': 'GenHlth',
    'MentHlth': 'MentHlth',
    'PhysHlth': 'PhysHlth',
    'DiffWalk': 'DiffWalk',
    'Sex': 'Sex',
    'Age': 'Age',
    'Education': 'Education',
    'Income': 'Income'
}

# 重命名列以匹配图片中的变量名称
df = df.rename(columns=column_mapping)

# 计算相关系数矩阵
corr = df.corr()

# 创建更小的相关性矩阵 (只显示相关性强的前10个特征)
diabetes_corr = corr['Diabetes'].sort_values(ascending=False)
significant_features = diabetes_corr[abs(diabetes_corr) > 0.05].index.tolist()
significant_corr = df[significant_features].corr()

# 创建紧凑的热力图
plt.figure(figsize=(12, 10))
sns.heatmap(
    significant_corr,
    annot=True,
    fmt=".2f",
    cmap='coolwarm',
    center=0,
    linewidths=0.5,
    annot_kws={'size': 9}
)

# 格式化标题和标签
plt.title('糖尿病特征相关性分析', fontsize=16, pad=20)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(fontsize=10)

# 保存图像而不是显示 (避免交互式问题)
plt.tight_layout()
plt.savefig('diabetes_correlation_heatmap.png', dpi=300)
print("成功保存相关性热力图: diabetes_correlation_heatmap.png")

# 额外输出重要相关性结果
print("\n与糖尿病相关性最强的特征:")
diabetes_corr_sorted = corr['Diabetes'].abs().sort_values(ascending=False).drop('Diabetes')
top_features = diabetes_corr_sorted.head(5).index.tolist()

# 创建特征重要性表格
importance_df = pd.DataFrame({
    '特征': top_features,
    '相关性': [corr.loc[feat, 'Diabetes'] for feat in top_features],
    '类型': ['负面' if corr.loc[feat, 'Diabetes'] < 0 else '正面' for feat in top_features]
})

print(importance_df)