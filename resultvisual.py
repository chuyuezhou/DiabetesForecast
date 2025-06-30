# ======================
# 实验3.1：模型训练与结果可视化
# ======================
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import (confusion_matrix, classification_report,
                             roc_curve, auc, precision_recall_curve, average_precision_score, accuracy_score,
                             precision_score, recall_score, f1_score, roc_auc_score)
import seaborn as sns
import joblib
import matplotlib.pyplot as plt
import matplotlib as mpl


mpl.use('TkAgg')  # 使用非交互式后端避免PyCharm兼容性问题

# 2. 设置字体支持 - 解决中文字符显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows中文字体
plt.rcParams['axes.unicode_minus'] = False    # 正确显示负号


# 设置图表样式
plt.style.use('seaborn-v0_8-bright')
sns.set_palette("Set2")

# 加载处理后的数据
df = pd.read_csv('data/filled_diabetes_data.csv')

# 定义特征和目标变量
features = ['HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke',
            'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
            'HvyAlcoholConsump', 'AnyHealthcare', 'MentHlth', 'PhysHlth',
            'DiffWalk', 'Sex', 'Age']
target = 'Diabetes_binary'

X = df[features]
y = df[target]

# 划分训练测试集 (70%训练，30%测试)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

print(f"训练集大小: {X_train.shape[0]} | 测试集大小: {X_test.shape[0]}")
print(f"糖尿病患者比例: 训练集 {y_train.mean():.3f} | 测试集 {y_test.mean():.3f}")

# 训练随机森林模型
print("\n训练随机森林模型中...")
rf_model = RandomForestClassifier(
    n_estimators=150,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)

# 保存模型
joblib.dump(rf_model, 'models/rf_diabetes_model.pkl')
print("模型已保存至 models/rf_diabetes_model.pkl")

# ======================
# 3.1.1 模型评估指标
# ======================
print("\n===== 模型评估指标 =====")
y_pred = rf_model.predict(X_test)
y_proba = rf_model.predict_proba(X_test)[:, 1]

# 计算各项指标
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

print(f"准确率: {accuracy:.4f}")
print(f"精确率: {precision:.4f}")
print(f"召回率: {recall:.4f}")
print(f"F1分数: {f1:.4f}")
print(f"AUC值: {roc_auc:.4f}")

# ======================
# 3.1.2 混淆矩阵可视化
# ======================
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=['预测健康', '预测糖尿病'],
            yticklabels=['实际健康', '实际糖尿病'])
plt.ylabel('真实类别', fontsize=12)
plt.xlabel('预测类别', fontsize=12)
plt.title('随机森林模型混淆矩阵', fontsize=14)
plt.savefig('results/confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n混淆矩阵已保存至 results/confusion_matrix.png")

# ======================
# 3.1.3 ROC曲线可视化
# ======================
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2,
         label=f'ROC曲线 (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='随机猜测')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('假阳性率 (FPR)', fontsize=12)
plt.ylabel('真阳性率 (TPR)', fontsize=12)
plt.title('随机森林模型ROC曲线', fontsize=14)
plt.legend(loc="lower right", fontsize=12)
plt.grid(alpha=0.3)
plt.savefig('results/roc_curve.png', dpi=300, bbox_inches='tight')
plt.close()

# 找到最佳阈值
# 选择距离(0,1)点最近的阈值
distances = np.sqrt((fpr - 0) ** 2 + (tpr - 1) ** 2)
optimal_idx = np.argmin(distances)
optimal_threshold = thresholds[optimal_idx]

print(f"最佳分类阈值: {optimal_threshold:.4f}")
print("ROC曲线已保存至 results/roc_curve.png")

# ======================
# 3.1.4 精确率-召回率曲线
# ======================
precision, recall, _ = precision_recall_curve(y_test, y_proba)
average_precision = average_precision_score(y_test, y_proba)

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, lw=2,
         label=f'PR曲线 (AP={average_precision:.4f})')
plt.xlabel('召回率', fontsize=12)
plt.ylabel('精确率', fontsize=12)
plt.title('精确率-召回率曲线', fontsize=14)
plt.legend(loc="upper right", fontsize=12)
plt.grid(alpha=0.3)
plt.savefig('results/precision_recall_curve.png', dpi=300, bbox_inches='tight')
plt.close()
print("PR曲线已保存至 results/precision_recall_curve.png")


# ======================
# 3.1.5 学习曲线分析
# ======================
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=5,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 10)):
    plt.figure(figsize=(10, 6))
    plt.title(title, fontsize=14)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("训练样本数", fontsize=12)
    plt.ylabel("得分 (AUC)", fontsize=12)

    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes,
        scoring='roc_auc', random_state=42
    )

    # 计算平均值和标准差
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.grid(alpha=0.3)

    # 填充标准差区域
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1,
                     color="g")

    # 绘制学习曲线
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="训练得分")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="交叉验证得分")

    plt.legend(loc="best")
    return plt


# 绘制学习曲线
plot_learning_curve(
    rf_model, "随机森林学习曲线 (AUC)",
    X_train, y_train, ylim=(0.6, 1.01),
    cv=3, n_jobs=-1
)
plt.savefig('results/learning_curve.png', dpi=300, bbox_inches='tight')
plt.close()
print("学习曲线已保存至 results/learning_curve.png")

# ======================
# 3.1.6 特征重要性分析
# ======================
feature_importance = pd.Series(rf_model.feature_importances_, index=features).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importance, y=feature_importance.index)
plt.title('特征重要性排序', fontsize=14)
plt.xlabel('特征重要性', fontsize=12)
plt.ylabel('特征', fontsize=12)
plt.tight_layout()
plt.savefig('results/feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()
print("特征重要性图已保存至 results/feature_importance.png")