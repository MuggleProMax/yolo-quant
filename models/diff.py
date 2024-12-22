import onnxruntime as ort
import numpy as np

# Load original and quantized models
original_session = ort.InferenceSession("yolo11n-pose.onnx")
quantized_session = ort.InferenceSession("yolo11n-pose-int8.onnx")

# Prepare dummy input
dummy_input = np.random.rand(1, 3, 640, 640).astype(np.float32)

# Run inference
original_output = original_session.run(None, {"images": dummy_input})
quantized_output = quantized_session.run(None, {"images": dummy_input})

# Compare results
for orig, quant in zip(original_output, quantized_output):
    print("Max diff:", np.max(np.abs(orig - quant)))

