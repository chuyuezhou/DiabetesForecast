import pandas as pd
import numpy as np
import seaborn as sns
import joblib
import os
import time
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner
from matplotlib import gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import griddata
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score

import matplotlib.pyplot as plt
import matplotlib as mpl


mpl.use('TkAgg')  # 使用非交互式后端避免PyCharm兼容性问题

# 2. 设置字体支持 - 解决中文字符显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows中文字体
plt.rcParams['axes.unicode_minus'] = False    # 正确显示负号


# 设置图表样式
plt.style.use('seaborn-v0_8-bright')
sns.set_palette("Set2")

# 设置结果目录
os.makedirs('models', exist_ok=True)
os.makedirs('results/optuna_plots', exist_ok=True)

# 设置随机种子
SEED = 42
np.random.seed(SEED)

# 2. 加载和准备数据
print("加载和处理数据...")
# 这里使用您的实际数据加载代码

# 加载处理后的数据
data = pd.read_csv('data/filled_diabetes_data.csv')

# 定义特征和目标变量
features = ['HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke',
            'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
            'HvyAlcoholConsump', 'AnyHealthcare', 'MentHlth', 'PhysHlth',
            'DiffWalk', 'Sex', 'Age']
target = 'Diabetes_binary'

# 分割特征和目标
X = data.drop('Diabetes_binary', axis=1)
y = data['Diabetes_binary']

# 分割训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=SEED
)

print(f"训练集: {X_train.shape[0]} 样本 (阳性率: {y_train.mean():.4f})")
print(f"测试集: {X_test.shape[0]} 样本 (阳性率: {y_test.mean():.4f})")


# 3. 贝叶斯优化函数
def objective(trial):
    """Optuna优化目标函数"""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 30),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_float('max_features', 0.1, 0.99),
        'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
        'class_weight': trial.suggest_categorical('class_weight', ['balanced', None]),
        'random_state': SEED
    }

    # 创建模型实例
    model = RandomForestClassifier(**params, n_jobs=-1)

    # 3折交叉验证的ROC AUC
    scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=3,
        scoring='roc_auc',
        n_jobs=1
    )

    return np.mean(scores)


# 4. 创建优化研究
def optimize_hyperparameters(n_trials=50):
    """运行贝叶斯优化研究"""
    print(f"\n开始贝叶斯优化，最大迭代次数: {n_trials}...")

    sampler = TPESampler(seed=SEED)
    pruner = HyperbandPruner()

    study = optuna.create_study(
        direction='maximize',
        sampler=sampler,
        pruner=pruner
    )

    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"优化完成! 最佳试验: {study.best_trial.number}")
    print(f"最佳AUC: {study.best_value:.4f}")
    print(f"最佳参数: {study.best_params}")

    return study


# 5. 执行优化
start_time = time.time()
study = optimize_hyperparameters(n_trials=50)
optimization_time = time.time() - start_time

print(f"\n⏱ 优化耗时: {optimization_time:.2f}秒")
print(f"平均每次试验耗时: {optimization_time / 50:.2f}秒")

# 6. 保存最佳模型
best_params = study.best_params
best_model = RandomForestClassifier(**best_params, random_state=SEED, n_jobs=-1)

# 在整个训练集上重新训练
print("\n使用最佳参数重新训练模型...")
best_model.fit(X_train, y_train)

# 保存模型
joblib.dump(best_model, 'models/rf_optuna_optimized.pkl')
print("优化模型已保存至 models/rf_optuna_optimized.pkl")

# 7. 评估最终模型
y_pred_prob = best_model.predict_proba(X_test)[:, 1]
test_auc = roc_auc_score(y_test, y_pred_prob)
print(f"\n🎯 测试集AUC: {test_auc:.4f}")

# 8. 使用Matplotlib创建所有可视化
print("\n使用Matplotlib生成优化过程可视化...")
trials_df = study.trials_dataframe()

# 8.1 优化历史图
plt.figure(figsize=(12, 6))
plt.plot(trials_df['number'], trials_df['value'], 'o-', markersize=4, linewidth=1.5, alpha=0.7)
plt.scatter(study.best_trial.number, study.best_value, s=150, c='red', zorder=5)
plt.annotate(f'Best trial\nAUC={study.best_value:.4f}',
             (study.best_trial.number, study.best_value),
             xytext=(10, -20),
             textcoords='offset points',
             arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.2"))
plt.xlabel('Trial Number')
plt.ylabel('Validation AUC')
plt.title('Optimization History')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('results/optuna_plots/optimization_history.png', dpi=300, bbox_inches='tight')
plt.close()
print("优化历史图已保存")

# 8.2 参数重要性图
# 计算参数重要性
param_importance = {}
for param in trials_df.columns:
    if param.startswith('params_') and trials_df[param].nunique() > 1:
        # 使用皮尔逊相关系数作为重要性的简化度量
        correlation = trials_df['value'].corr(trials_df[param], method='pearson')
        param_importance[param.replace('params_', '')] = abs(correlation)

# 转换为DataFrame并排序
importance_df = pd.DataFrame(list(param_importance.items()), columns=['Parameter', 'Importance'])
importance_df = importance_df.sort_values('Importance', ascending=False)

# 更新参数重要性图
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Parameter', data=importance_df,
            hue='Parameter', palette='viridis_r', legend=False, dodge=False)
plt.title('Hyperparameter Importance')
plt.xlabel('Correlation Magnitude (|ρ|)')
plt.ylabel('Hyperparameter')
plt.grid(axis='x', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('results/optuna_plots/param_importance.png', dpi=300, bbox_inches='tight')
plt.close()
print("参数重要性图已保存")


# 8.3 并行坐标图 (改进版散点图矩阵)
# 选择最重要的4个参数
if len(importance_df) > 4:
    top_params = importance_df.head(4)['Parameter'].tolist()
else:
    top_params = importance_df['Parameter'].tolist()

fig = plt.figure(figsize=(16, 12))

# 创建散点图矩阵
# 更新核密度图
for i, param1 in enumerate(top_params):
    for j, param2 in enumerate(top_params):
        ax = plt.subplot(len(top_params), len(top_params), i * len(top_params) + j + 1)

        if i == j:  # 对角线 - 核密度图
            sns.kdeplot(trials_df[f'params_{param1}'], fill=True, ax=ax, color='skyblue')  # 改为 fill=True
            ax.set_title(f'Distribution of {param1}')
        else:
            scatter = ax.scatter(trials_df[f'params_{param1}'], trials_df[f'params_{param2}'],
                                 c=trials_df['value'], cmap='viridis', alpha=0.7, s=20)
            # 高亮最佳试验
            ax.scatter(study.best_trial.params.get(param1), study.best_trial.params.get(param2),
                       s=150, c='red', zorder=5, edgecolors='white')

            ax.set_xlabel(param1)
            ax.set_ylabel(param2)

            # 只在右侧和最下边添加颜色条
            if j == len(top_params) - 1 and i < len(top_params) - 1:
                divider = make_axes_locatable(ax)
                cax = divider.append_axes('right', size='5%', pad=0.1)
                plt.colorbar(scatter, cax=cax).set_label('AUC Score')

        if j == 0:
            ax.set_ylabel(param1)
        if i == len(top_params) - 1:
            ax.set_xlabel(param2)

plt.suptitle('Hyperparameter Relationships', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig('results/optuna_plots/param_relationships.png', dpi=300, bbox_inches='tight')
plt.close()
print("参数关系图已保存")

# 8.4 热力图替代
if 'params_n_estimators' in trials_df and 'params_max_depth' in trials_df:
    # 准备数据
    n_est = trials_df['params_n_estimators']
    max_dep = trials_df['params_max_depth']
    auc_values = trials_df['value']

    # 创建网格
    xi = np.linspace(min(n_est), max(n_est), 100)
    yi = np.linspace(min(max_dep), max(max_dep), 100)
    zi = griddata((n_est, max_dep), auc_values, (xi[None, :], yi[:, None]), method='cubic')

    plt.figure(figsize=(10, 8))

    # 等高线图
    contour = plt.contourf(xi, yi, zi, levels=20, cmap='viridis', alpha=0.8)
    plt.colorbar(contour).set_label('Validation AUC')

    # 最佳点标记
    best_n = study.best_params.get('n_estimators')
    best_d = study.best_params.get('max_depth')
    if best_n and best_d:
        plt.scatter(best_n, best_d, s=200, c='red', edgecolors='white', linewidth=1.5, zorder=5)
        plt.annotate(f'Best point\nAUC={study.best_value:.4f}',
                     (best_n, best_d),
                     xytext=(15, -15),
                     textcoords='offset points',
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.7),
                     arrowprops=dict(arrowstyle="->", color="red"))

    plt.xlabel('n_estimators')
    plt.ylabel('max_depth')
    plt.title('Hyperparameter Performance Heatmap')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/optuna_plots/contour_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("热力图已保存")

# 9. 特征重要性可视化
plt.figure(figsize=(12, 8))
feature_importances = pd.Series(best_model.feature_importances_, index=X_train.columns)
feature_importances.nlargest(15).sort_values().plot(kind='barh', color=sns.color_palette('viridis_r', 15))
plt.title(f'Top 15 Feature Importances (Test AUC={test_auc:.4f})')
plt.xlabel('Importance Score')
plt.grid(axis='x', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('results/optuna_plots/feature_importances.png', dpi=300, bbox_inches='tight')
plt.close()
print("特征重要性图已保存")

print("\n✅ 所有可视化图表已成功生成并保存!")