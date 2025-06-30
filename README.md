# 糖尿病预测机器学习项目

本项目基于BRFSS2015数据集，使用多种机器学习算法进行糖尿病预测，包含完整的数据预处理、模型训练和结果可视化流程。

## 主要功能
- 数据探索与预处理
- 特征相关性分析
- 多种分类模型训练（逻辑回归、随机森林、梯度提升等）
- 超参数优化（使用Optuna）
- 模型评估与可视化

## 环境要求
```bash
Python 3.8+
pip install -r requirements.txt
```

## 数据准备
1. 原始数据存放于`data/`目录
2. 运行`preprocess.py`进行数据预处理
3. 生成清洗后的数据集`data/filled_diabetes_data.csv`

## 使用说明
```bash
# 训练基础模型
python modelbuild.py

# 生成模型的可视化结果
python resultvisual.py

# 执行超参数优化并生成可视化图像 
python parameter_tuning.py


```

## 项目结构
```
├── data/              # 数据集
├── models/            # 训练好的模型
├── results/           # 评估结果与图表
├── visualizations/    # 数据可视化
├── main.py            # 主程序
└── requirements.txt   # 依赖库
```

## 贡献指南
欢迎提交Pull Request，请确保：
1. 代码符合PEP8规范
2. 新增功能需包含单元测试
3. 更新相关文档

## 许可
[MIT License](LICENSE)