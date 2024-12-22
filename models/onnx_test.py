import cv2
import onnxruntime as ort
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def preprocess(image, input_size=(640, 640)):
    """
    预处理输入图像：调整尺寸、归一化和转换格式
    """
    original_size = image.shape[:2]
    image_resized = cv2.resize(image, input_size)
    image_normalized = image_resized / 255.0
    input_tensor = image_normalized.transpose(2, 0, 1)[None, :, :, :].astype(np.float32)
    return input_tensor, original_size

def nms(boxes, scores, iou_threshold=0.5):
    """
    非极大值抑制，过滤多余的边界框
    """
    # 确保 boxes 和 scores 为 NumPy 数组
    boxes_np = np.array(boxes)
    scores_np = np.array(scores)
    indices = cv2.dnn.NMSBoxes(boxes_np.tolist(), scores_np.tolist(), score_threshold=0.5, nms_threshold=iou_threshold)
    return indices.flatten() if len(indices) > 0 else []

def connect_keypoints(image, keypoints, skeleton, colors):
    """
    根据骨架连接规则连接关键点
    """
    for i, (start, end) in enumerate(skeleton):
        if keypoints[start][2] > 0.5 and keypoints[end][2] > 0.5:
            x1, y1 = int(keypoints[start][0]), int(keypoints[start][1])
            x2, y2 = int(keypoints[end][0]), int(keypoints[end][1])
            cv2.line(image, (x1, y1), (x2, y2), colors[i % len(colors)], 2)

def postprocess(output_boxs, keypoints, original_image, input_size, original_size, conf_thresh=0.5):
    """
    解析模型输出并绘制边界框和关键点
    """
    input_h, input_w = input_size
    orig_h, orig_w = original_size
    scale_h, scale_w = orig_h / input_h, orig_w / input_w

    boxes = []
    confidences = []
    kpts_list = []

    for i in range(output_boxs.shape[0]):
        conf = output_boxs[i][-1]
        if conf >= conf_thresh:
            x_center = output_boxs[i][0] * scale_w
            y_center = output_boxs[i][1] * scale_h
            width = output_boxs[i][2] * scale_w
            height = output_boxs[i][3] * scale_h
            x1, y1 = int(x_center - width / 2), int(y_center - height / 2)
            x2, y2 = int(x_center + width / 2), int(y_center + height / 2)

            boxes.append([x1, y1, x2 - x1, y2 - y1])
            confidences.append(conf)
            cur_kps = keypoints[i].reshape(-1, 3)
            cur_kps[:, 0] *= scale_w
            cur_kps[:, 1] *= scale_h
            kpts_list.append(cur_kps)

    indices = nms(boxes, confidences)

    skeleton = [
        (0, 1), (1, 3), (0, 2), (2, 4),  # 头部连接：鼻子到左眼、左眼到左耳，鼻子到右眼、右眼到右耳
        (0, 5), (5, 7), (7, 9),          # 左臂：鼻子到左肩、左肩到左肘、左肘到左腕
        (0, 6), (6, 8), (8, 10),         # 右臂：鼻子到右肩、右肩到右肘、右肘到右腕
        (5, 11), (11, 13), (13, 15),     # 左腿：左肩到左髋、左髋到左膝、左膝到左脚踝
        (6, 12), (12, 14), (14, 16),     # 右腿：右肩到右髋、右髋到右膝、右膝到右脚踝
        (11, 12) 
    ]

    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (0, 255, 255), (255, 0, 255), (128, 128, 0), (128, 0, 128),
        (0, 128, 128), (128, 128, 128)
    ]

    for i in indices:
        box = boxes[i]
        x1, y1, w, h = box
        conf = confidences[i]
        cv2.rectangle(original_image, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)
        cv2.putText(original_image, f"person {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        connect_keypoints(original_image, kpts_list[i], skeleton, colors)

    return original_image

def real_time_detection():
    model_path = "yolo11n-pose-int8.onnx"
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    global session
    session = ort.InferenceSession(
        model_path, sess_options=session_options, providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    )
    global input_name, output_name
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: 无法打开摄像头！")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: 无法读取帧！")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_tensor, original_size = preprocess(frame_rgb)

        outputs = session.run([output_name], {input_name: input_tensor})
        outputs_rs = outputs[0][0].T

        boxs = outputs_rs[:, 0:5]
        keypoints = outputs_rs[:, 5:]

        result_frame = postprocess(boxs, keypoints, frame, (640, 640), original_size, conf_thresh=0.5)

        cv2.imshow("YOLOv11n-Pose Real-Time Detection", result_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    real_time_detection()
