# train_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier  # 示例模型，可替换
import pickle
import os

# 定义训练函数
def train_ufo_model(data_path, model_save_path="./ufo-model.pkl"):
    """
    训练UFO目击国家预测模型
    :param data_path: 训练数据路径（csv文件）
    :param model_save_path: 模型保存路径
    :return: 训练完成的模型（同时保存到本地）
    """
    # 1. 加载数据（需确保数据格式：seconds, latitude, longitude, country）
    # country列是标签：0=Australia,1=Canada,2=Germany,3=UK,4=US（和app.py对应）
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"训练数据文件不存在：{data_path}")
    
    df = pd.read_csv(data_path)
    
    # 2. 数据预处理（根据你的数据调整，示例仅做基础处理）
    # 检查必要列是否存在
    required_cols = ["seconds", "latitude", "longitude", "country"]
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"数据缺少必要列，需包含：{required_cols}")
    
    # 处理缺失值
    df = df.dropna(subset=required_cols)
    
    # 特征和标签分离
    X = df[["seconds", "latitude", "longitude"]].astype(float)  # 特征
    y = df["country"].astype(int)  # 标签（0-4对应不同国家）
    
    # 3. 划分训练集/测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 4. 初始化模型（可替换为你需要的模型：如逻辑回归、SVM等）
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # 5. 训练模型
    model.fit(X_train, y_train)
    
    # 6. 保存模型到本地
    with open(model_save_path, "wb") as f:
        pickle.dump(model, f)
    
    # 7. 可选：输出训练精度（方便调试）
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    print(f"训练完成！训练集精度：{train_acc:.2f}，测试集精度：{test_acc:.2f}")
    
    return model

# 测试训练函数（可选，单独运行train_model.py时执行）
if __name__ == "__main__":
    train_ufo_model(data_path="./data/ufo_data.csv")