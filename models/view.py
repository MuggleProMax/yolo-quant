import numpy as np
import onnxruntime as ort

# 加载 ONNX 模型
onnx_model_path = "yolo11n-pose.onnx"  # 替换为你的模型路径
session = ort.InferenceSession(onnx_model_path)

# 获取模型的输入信息
print("模型的输入参数信息：")
for input_tensor in session.get_inputs():
    print(f"输入名称: {input_tensor.name}")
    print(f"输入形状: {input_tensor.shape}")
    print(f"输入数据类型: {input_tensor.type}")

# 获取模型的输出信息
print("\n模型的输出参数信息：")
for output_tensor in session.get_outputs():
    print(f"输出名称: {output_tensor.name}")
    print(f"输出形状: {output_tensor.shape}")
    print(f"输出数据类型: {output_tensor.type}")

# 创建一个虚拟输入，用于测试推理（假设输入为 [1, 3, 640, 640] 的张量）
input_name = session.get_inputs()[0].name
dummy_input = np.random.rand(1, 3, 640, 640).astype(np.float32)  # 随机生成输入数据

# 推理
outputs = session.run(None, {input_name: dummy_input})  # `None` 表示获取所有输出

# 打印输出信息
print("\n模型的推理结果：")
print(f"模型输出的形状：{outputs[0].shape}")  # 打印第一个输出的形状
print("模型输出的内容示例：", outputs[0][:, :5, :5])  # 打印前 5 个数据点的示例

