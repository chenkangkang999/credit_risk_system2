# 可公开的演示资产

sample.json：使用原项目生成器产生的 5,000 条合成记录，种子 42。
bundle.joblib：仅由上述合成数据训练，含 XGBoost、LightGBM、MLP 和原配置的 XGBoost+MLP 融合结果，以及训练时的转换状态。

生成方式：python scripts/prepare_demo.py

没有真实客户记录或 API 密钥。不要把其他训练结果手工覆盖到这里。joblib 是 Python 序列化格式，应用只加载项目自带文件和本地可信训练产物，不支持从页面上传模型文件。
