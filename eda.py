import os
import json
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import numpy as np

def iou(box1, box2):
    """
    计算两个边界框的IoU
    box格式：[x1, y1, x2, y2]
    """
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    iou = inter_area / float(box1_area + box2_area - inter_area)
    return iou

def calculate_map(pred_bboxes, gt_bbox, iou_threshold=0.5):
    """
    计算mAP
    pred_bboxes: 预测的边界框列表，每个元素是[x1, y1, x2, y2]
    gt_bbox: 真实的边界框，[x1, y1, x2, y2]
    iou_threshold: 判断为True Positive的IoU阈值
    """
    tp = 0  # True Positives
    fp = 0  # False Positives
    fn = 1  # False Negatives (只有一个gt，所以初始为1)
    
    for pred_bbox in pred_bboxes:
        if iou(pred_bbox, gt_bbox) >= iou_threshold:
            tp += 1
            fn -= 1
        else:
            fp += 1
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    ap = precision * recall  # 简单计算AP（实际中可能会使用更复杂的计算方法）

    return ap

# 示例数据
pred_bboxes = [
    [50, 50, 150, 150],
    [30, 30, 100, 100],
    [60, 60, 200, 200],
    [70, 70, 120, 120]
]

gt_bbox = [55, 55, 160, 160]

map_value = calculate_map(pred_bboxes, gt_bbox)
print(f'mAP: {map_value}')
def generate_gt_json(txt_dir):
    txt_files = os.listdir(txt_dir)
    sorted_txt_files = sorted(txt_files, key=lambda x: int(x[:-4]))
    for txt_file in sorted_txt_files:
        id = txt_file[:-4]
        if id == "105":
            break
        file = open(f"{txt_dir}/{txt_file}")

        img_path = f"/home/ubuntu/106-90t/personal_data/jt/dataset/dataset_release/train/{id}.jpg"

        # 读取图像并获取宽高
        with Image.open(img_path) as img:
            img_size = list(img.size)

        # 构建JSON数据结构
        didi_data = []
        for text in file:
            cls_index = text.split(' ')[0]
            bbox = [float(x) for x in text.split(' ')[1:]]
            didi_data.append({
                "class": cls_index,
                "xywh": bbox,
                "attrs": {}
            })

        json_data = {
            "imgsize_wh": img_size,
            "task_configs": "didi",
            "didi": didi_data,

        }

        # 写入到JSON文件中
        json_file_path = f"/home/ubuntu/jt/dinovdemo/didi/targets/{id}.json"
        with open(json_file_path, 'w') as json_file:
            json.dump(json_data, json_file, indent=4)

def visualize_pred_bbox():
    json_files = os.listdir('./preds')
    for json_file in json_files:
        with open(f"./preds/{json_file}", 'r') as file:
            pred_data = json.load(file)

        vp_cls = pred_data['didi'][0]['class']

        img_path = pred_data['img_path']
        img_id = img_path.split('/')[-1][:-4]

        with open(f"./targets/{img_id}.json", 'r') as file:
            gt_data = json.load(file)

        # Load image using OpenCV
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert color from BGR to RGB for matplotlib

        # Plot predicted bounding boxes in red
        for bbox in pred_data['didi']:
            x, y, w, h = bbox['xywh']
            xmin, ymin = int(x), int(y)
            xmax, ymax = int(x + w), int(y + h)
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)

        # Plot ground truth bounding boxes in green
        for bbox in gt_data['didi']:
            if bbox['class'] != vp_cls:
                continue
            x, y, w, h = bbox['xywh']
            xmin, ymin = int(x), int(y)
            xmax, ymax = int(x + w), int(y + h)
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

        cv2.imwrite(f"./visual/{img_id}.jpg", image)


if __name__ == "__main__":
    generate_gt_json("/home/ubuntu/jt/dinovdemo/new_dataset_2/train/bbox_infos")
    # visualize_pred_bbox()