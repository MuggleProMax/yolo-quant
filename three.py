from ultralytics import YOLO
import cv2
import os
import numpy as np

def initialize_models():
    """初始化所有模型"""
    try:
        face_model = YOLO(os.path.join('models', 'yolov8n-face.pt'))  # 人脸检测
        person_model = YOLO(os.path.join('models', 'yolo11n.pt'))     # 人体检测
        pose_model = YOLO(os.path.join('models', 'yolo11n-pose.pt'))  # 姿态检测
        return face_model, person_model, pose_model
    except Exception as e:
        print(f"加载模型失败: {str(e)}")
        return None, None, None

def draw_poses(frame, results):
    """绘制姿态检测结果"""
    # 定义骨架连接
    skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13],
                [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],
                [2, 4], [3, 5], [4, 6], [5, 7]]
    
    # 定义关键点颜色
    pose_palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102], [230, 230, 0],
                            [255, 153, 255], [153, 204, 255], [255, 102, 255], [255, 51, 255],
                            [102, 178, 255], [51, 153, 255], [255, 153, 153], [255, 102, 102],
                            [255, 51, 51], [153, 255, 153], [102, 255, 102], [51, 255, 51],
                            [0, 255, 0]], dtype=np.uint8)

    for r in results:
        if r.keypoints is not None and len(r.keypoints.data) > 0:
            keypoints = r.keypoints.data[0]
            
            # 绘制骨架
            for i, connection in enumerate(skeleton):
                idx1, idx2 = connection[0] - 1, connection[1] - 1
                
                if (idx1 < len(keypoints) and idx2 < len(keypoints) and 
                    keypoints[idx1][2] > 0.5 and keypoints[idx2][2] > 0.5):
                    
                    x1, y1 = int(keypoints[idx1][0]), int(keypoints[idx1][1])
                    x2, y2 = int(keypoints[idx2][0]), int(keypoints[idx2][1])
                    color = pose_palette[i % len(pose_palette)]
                    cv2.line(frame, (x1, y1), (x2, y2), color.tolist(), 2)
            
            # 绘制关键点
            for i, kpt in enumerate(keypoints):
                x, y, conf = kpt
                if conf > 0.5:
                    color = pose_palette[i % len(pose_palette)]
                    cv2.circle(frame, (int(x), int(y)), 4, color.tolist(), -1)

def detect_all():
    """同时执行人脸、人体和姿态检测"""
    face_model, person_model, pose_model = initialize_models()
    if None in (face_model, person_model, pose_model):
        return
    
    print("正在打开摄像头...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("无法打开摄像头")
        return

    print("按 'q' 键退出程序")
    
    while cap.isOpened():
        success, frame = cap.read()
        
        if success:
            # 人脸检测 (绿色框)
            face_results = face_model(frame)
            for r in face_results:
                boxes = r.boxes
                for box in boxes:
                    conf = float(box.conf)
                    if conf > 0.5:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = f'Face {conf:.2f}'
                        cv2.putText(frame, label, (x1, y1 - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # 人体检测 (蓝色框)
            person_results = person_model(frame, classes=[0])  # 只检测人类
            for r in person_results:
                boxes = r.boxes
                for box in boxes:
                    conf = float(box.conf)
                    if conf > 0.5:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        label = f'Person {conf:.2f}'
                        cv2.putText(frame, label, (x1, y1 - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            # 姿态检测
            pose_results = pose_model(frame)
            draw_poses(frame, pose_results)
            
            # 显示FPS
            fps = cap.get(cv2.CAP_PROP_FPS)
            cv2.putText(frame, f'FPS: {int(fps)}', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # 显示结果
            cv2.imshow('Multiple Detection', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print("无法读取摄像头画面")
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # 检查必要的包是否安装
    try:
        import ultralytics
        import cv2
        import numpy as np
    except ImportError:
        print("首次运行需要安装必要的包...")
        os.system('pip install ultralytics opencv-python numpy')
    
    detect_all()
