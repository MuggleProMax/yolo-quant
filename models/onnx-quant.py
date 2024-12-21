import numpy as np
import cv2
from onnxruntime.quantization import quantize_static, QuantType, CalibrationDataReader

def load_calibration_data(image_paths, input_size=(640, 640)):
    def preprocess(image_path):
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image, input_size)
        image_normalized = image_resized / 255.0
        input_tensor = image_normalized.transpose(2, 0, 1).astype(np.float32)
        return input_tensor

    data = []
    for path in image_paths:
        data.append(preprocess(path))
    return np.stack(data, axis=0)

class CalibrationReader(CalibrationDataReader):
    def __init__(self, image_paths_file, input_name, input_size=(640, 640)):
        with open(image_paths_file, "r") as f:
            self.image_paths = [line.strip() for line in f]
        self.data = load_calibration_data(self.image_paths, input_size)
        self.index = 0
        self.input_name = input_name

    def get_next(self):
        if self.index >= len(self.data):
            return None
        batch = self.data[self.index:self.index + 1]  # Batch size = 1
        self.index += 1
        return {self.input_name: batch}

def quantize_model():
    # 模型路径
    model_input_path = "yolo11n-pose.onnx"
    model_output_path = "yolo11n-pose-quantized.onnx"

    # 输入节点名称（请确认输入节点的实际名称）
    input_name = "images"

    # 校准数据路径
    image_paths_file = "image_paths.txt"

    # 创建校准数据读取器
    calibration_reader = CalibrationReader(image_paths_file, input_name)

    # 执行量化
    quantize_static(
        model_input=model_input_path,  # 原始模型路径
        model_output=model_output_path,  # 量化后模型路径
        calibration_data_reader=calibration_reader,
        quant_format=QuantType.QInt8  # 使用 INT8 量化
    )

    print(f"量化完成，量化模型保存至：{model_output_path}")

if __name__ == "__main__":
    quantize_model()

