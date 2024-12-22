import os
import cv2
import onnx
import numpy as np

# 如果你想用 onnxruntime-tools 进行量化 (某些旧版本api略有不同)
# from onnxruntime.quantization import quantize_static, QuantType, QuantFormat, CalibrationMethod

# 如果 onnxruntime-tools 版本太旧，你可以尝试用官方 onnxruntime 中自带的 quantization:
from onnxruntime.quantization import (
    quantize_static,
    QuantType,
    QuantFormat,
    CalibrationMethod
)
from onnxruntime.quantization.calibrate import CalibrationDataReader


###############################
# 第 1 步：你的预处理函数
###############################
def preprocess(image, input_size=(640, 640)):
    """
    预处理输入图像：调整尺寸、归一化和转换格式
    保持和推理脚本一致。
    """
    original_size = image.shape[:2]
    image_resized = cv2.resize(image, input_size)
    image_normalized = image_resized / 255.0
    input_tensor = image_normalized.transpose(2, 0, 1)[None, :, :, :].astype(np.float32)
    return input_tensor, original_size


###############################
# 第 2 步：校准数据读取器
###############################
class YOLOCalibDataReader(CalibrationDataReader):
    """
    用于静态量化校准的数据读取器：
      - 根据 image_paths.txt 获取图像路径
      - 使用和推理时相同的 preprocess 函数处理图像
      - 每次返回一个 batch 的输入（此处示例一次返回 1 张图）
    """
    def __init__(self, image_list_file, input_name, preprocess_func, input_size=(640, 640)):
        super().__init__()
        self.input_name = input_name
        self.preprocess_func = preprocess_func
        self.input_size = input_size

        # 读取 image_paths.txt 内所有图像路径
        with open(image_list_file, 'r') as f:
            lines = f.readlines()
        self.image_paths = [line.strip() for line in lines if line.strip()]

        self.index = 0
        self.total_count = len(self.image_paths)

    def get_next(self):
        """
        每次返回一个 batch 的数据（这里示例一次返回 1 张图片）。
        如果数据读完则返回 None。
        """
        if self.index >= self.total_count:
            return None

        image_path = self.image_paths[self.index]
        self.index += 1

        # 读取并预处理
        image = cv2.imread(image_path)
        # 如果需要 BGR->RGB，这里也可以加上：image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # 视你的推理 preprocess 需求而定
        input_tensor, _ = self.preprocess_func(image, self.input_size)

        # 返回的格式必须是 { "输入名" : numpy_array }
        return {self.input_name: input_tensor}

    def rewind(self):
        """
        当需要多轮校准时，量化器可能会调用该函数重复使用数据。
        """
        self.index = 0


###############################
# 第 3 步：量化主函数
###############################
def run_quantization():
    # 1. 指定原始模型路径 (fp32) 以及量化后输出的模型路径 (int8)
    fp32_model_path = "yolo11n-pose.onnx"
    int8_model_path = "yolo11n-pose-int8.onnx"

    # 2. 构建校准数据读取器
    input_name = "images"                # 请确保与实际模型输入名一致
    image_list_file = "image_paths.txt"  # 校准图片列表文件
    data_reader = YOLOCalibDataReader(
        image_list_file=image_list_file,
        input_name=input_name,
        preprocess_func=preprocess,      # 使用上面定义的预处理函数
        input_size=(640, 640)            # 模型要求的输入大小
    )

    # 3. 调用 quantize_static 开始静态量化
    quantize_static(
        model_input=fp32_model_path,
        model_output=int8_model_path,
        calibration_data_reader=data_reader,
        quant_format=QuantFormat.QDQ,           # 量化格式可选 QOperator / QDQ
        per_channel=False,                      # 是否使用 per-channel 量化
        activation_type=QuantType.QUInt8,       # 激活量化类型
        weight_type=QuantType.QInt8,            # 权重量化类型
        calibrate_method=CalibrationMethod.MinMax
    )

    print("[INFO] 量化完成，已生成模型:", int8_model_path)


###############################
# 第 4 步：脚本入口示例
###############################
if __name__ == "__main__":
    # 直接调用量化函数
    run_quantization()
