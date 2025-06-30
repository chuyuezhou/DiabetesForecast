import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, recall_score, precision_score
from sklearn.pipeline import Pipeline

# 确保特征名称与要求一致
REQUIRED_FEATURES = [
    'Diabetes_binary', 'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke',
    'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies', 'HvyAlcoholConsump', 'AnyHealthcare',
    'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age'
]

# 加载处理好的数据集
data = pd.read_csv('data/filled_diabetes_data.csv')

# 验证特征名称匹配
missing_features = set(REQUIRED_FEATURES) - set(data.columns)
if missing_features:
    raise ValueError(f"数据集缺少以下必要特征: {', '.join(missing_features)}")

# 调整列顺序
data = data[REQUIRED_FEATURES]

# 数据集划分
X = data.drop('Diabetes_binary', axis=1)
y = data['Diabetes_binary']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

print("数据分割完成:")
print(f"训练集: {X_train.shape[0]} 样本 (阳性: {y_train.sum()}，比例: {y_train.mean():.2f})")
print(f"测试集: {X_test.shape[0]} 样本 (阳性: {y_test.sum()}，比例: {y_test.mean():.2f})")


# 构建模型框架
class DiabetesClassifier:
    """糖尿病预测模型框架"""

    def __init__(self, model_type='random_forest', save_model=True):
        """
        初始化模型
        model_type: 模型类型 ['logistic', 'random_forest', 'gbm']
        save_model: 是否保存训练好的模型
        """
        self.model_type = model_type
        self.save_model = save_model
        self.scaler = StandardScaler()
        self.models = {
            'logistic': self._build_logistic(),
            'random_forest': self._build_random_forest(),
            'gbm': self._build_gbm()
        }

        if model_type not in self.models:
            raise ValueError(f"不支持的模型类型: {model_type}")

        self.pipeline = Pipeline([
            ('scaler', self.scaler),
            ('classifier', self.models[model_type])
        ])

    def _build_logistic(self):
        """构建逻辑回归模型"""
        return LogisticRegression(
            penalty='l2',
            C=0.1,
            class_weight='balanced',
            max_iter=1000,
            solver='liblinear',
            random_state=42
        )

    def _build_random_forest(self):
        """构建随机森林模型"""
        return RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )

    def _build_gbm(self):
        """构建梯度提升树模型"""
        return GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=7,
            min_samples_split=20,
            subsample=0.8,
            random_state=42
        )

    def train(self, X_train, y_train):
        """训练模型"""
        print(f"开始训练 {self.model_type} 模型...")
        self.pipeline.fit(X_train, y_train)
        print(f"{self.model_type} 模型训练完成")

        if self.save_model:
            model_file = f"models/diabetes_{self.model_type}_model.pkl"
            joblib.dump(self.pipeline, model_file)
            print(f"模型已保存至: {model_file}")

    def predict(self, X):
        """预测糖尿病概率"""
        return self.pipeline.predict_proba(X)[:, 1]

    def evaluate(self, X_test, y_test):
        """评估模型性能"""
        y_pred_proba = self.predict(X_test)
        y_pred = (y_pred_proba > 0.5).astype(int)

        metrics = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1-Score': f1_score(y_test, y_pred),
            'AUC': roc_auc_score(y_test, y_pred_proba)
        }

        print(f"\n{self.model_type} 模型在测试集上的表现:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")

        return metrics


# 创建模型目录
import os

os.makedirs('models', exist_ok=True)

# 模型训练与评估
final_metrics = {}
for model_type in ['logistic', 'random_forest', 'gbm']:
    classifier = DiabetesClassifier(model_type=model_type)
    classifier.train(X_train, y_train)
    metrics = classifier.evaluate(X_test, y_test)
    final_metrics[model_type] = metrics
    print("-" * 60)

# 保存模型比较结果
model_comparison = pd.DataFrame(final_metrics).T
model_comparison.to_csv('models/model_comparison.csv')
print("模型比较结果已保存至 models/model_comparison.csv")

print("\n糖尿病预测模型构建完成")