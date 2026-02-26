# app.py
import numpy as np
from flask import Flask, request, render_template, jsonify
import pickle
import os
from train_model import train_ufo_model  # 导入训练函数

app = Flask(__name__)

# 模型路径
MODEL_PATH = "./ufo-model.pkl"

# 加载模型（如果模型不存在，先提示训练）
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

# 首页（预测页面）
@app.route("/")
def home():
    return render_template("index.html")

# 预测路由
@app.route("/predict", methods=["POST"])
def predict():
    global model
    # 检查模型是否加载
    if model is None:
        return render_template("index.html", 
                               prediction_text="错误：模型未训练！请先访问 /train 训练模型")
    
    try:
        # 获取表单数据并转换（注意纬度/经度是浮点数，原代码int转换错误，需修正）
        features = [
            float(request.form["seconds"]),
            float(request.form["latitude"]),
            float(request.form["longitude"])
        ]
        final_features = [np.array(features)]
        prediction = model.predict(final_features)
        output = prediction[0]
        countries = ["Australia", "Canada", "Germany", "UK", "US"]
        return render_template("index.html", 
                               prediction_text=f"Likely country: {countries[output]}")
    except Exception as e:
        return render_template("index.html", 
                               prediction_text=f"预测失败：{str(e)}")

# 训练路由（支持GET/POST，GET直接触发训练，POST可传参数）
@app.route("/train", methods=["GET", "POST"])
def train():
    global model
    try:
        # 训练数据路径（根据你的实际路径调整）
        data_path = "./data/ufos.csv"
        # 触发训练
        model = train_ufo_model(data_path=data_path, model_save_path=MODEL_PATH)
        return jsonify({
            "status": "success",
            "message": "模型训练完成！",
            "model_path": MODEL_PATH
        })
    except Exception as e:
        return jsonify({
            "status": "failed",
            "message": f"训练失败：{str(e)}"
        })

if __name__ == "__main__":
    app.run(debug=True)