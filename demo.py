import os
import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def load_annotations(annotation_dir):
    annotations = []
    images = []
    ann_id = 1
    for file_name in os.listdir(annotation_dir):
        file_path = os.path.join(annotation_dir, file_name)
        with open(file_path, 'r') as file:
            data = json.load(file)
            img_id = file_name.split('.')[0]  # 提取图像名, 需要纯数字!!!!
            imgsize_wh = data['imgsize_wh']  # 图像尺寸
            images.append({'id': img_id, 'width': imgsize_wh[0], 'height': imgsize_wh[1]})

            # 把所有gt bbox写入json文件, id从1开始递增，对应测试集的每个bbox
            for obj in data.get('movable_barrier', []):
                annotations.append({
                    'id': ann_id,
                    'image_id': img_id,
                    'category_id': CLASS[obj['class']],
                    'bbox': obj['xywh'],  # xy是中心点坐标
                    'area': obj['xywh'][2] * obj['xywh'][3],
                    'iscrowd': 0
                })
                ann_id += 1
    return annotations, images


def load_predictions(prediction_dir):
    predictions = []
    for file_name in os.listdir(prediction_dir):
        file_path = os.path.join(prediction_dir, file_name)
        with open(file_path, 'r') as file:
            data = json.load(file)
            img_id = file_name.split('.')[0]

            # 把所有pred bbox写入json文件, 只有image_id, 没有bbox的id
            for obj in data.get('movable_barrier', []):
                predictions.append({
                    'image_id': img_id,
                    'category_id': CLASS[obj['class']],
                    'bbox': obj['xywh'],  # xy是中心点坐标
                    'score': obj['score']
                })
    return predictions


def calculate_precision_recall(ground_truth_file, predictions_file, category_id=None):
    coco_gt = COCO(ground_truth_file)
    coco_dt = coco_gt.loadRes(predictions_file)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')

    # 要评估的类别
    if category_id is not None:
        coco_eval.params.catIds = [category_id]

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # 获取精度和召回率
    precision = coco_eval.stats[0] if coco_eval.stats[0] >= 0 else 0  # precision
    recall = coco_eval.stats[8] if coco_eval.stats[8] >= 0 else 0  # recall

    return precision, recall


def main(gt_dir, pred_dir, save_path, class_dict):
    gt_annotations, images = load_annotations(gt_dir)
    predictions = load_predictions(pred_dir)
    categories = [{"id": v, "name": k} for k, v in class_dict.items()]
    gt_coco = {
        'images': images,
        'annotations': gt_annotations,
        'categories': categories
    }

    gt_save_path = f'{save_path}/coco_gt.json'
    with open(gt_save_path, 'w') as f:
        json.dump(gt_coco, f, indent=4)
    
    dt_save_path = f'{save_path}/coco_dt.json'
    with open(dt_save_path, 'w') as f:
        json.dump(predictions, f, indent=4)

    # 按类别计算AP
    for categrory_name, category_id in class_dict.items():
        precision, recall = calculate_precision_recall(gt_save_path, dt_save_path, category_id)
        print(f"Class {categrory_name}: Precision = {precision}, Recall = {recall}")
    
    # 整体计算mAP
    # coco_gt = COCO(gt_save_path)
    # coco_dt = coco_gt.loadRes(dt_save_path)

    # coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    # coco_eval.evaluate()
    # coco_eval.accumulate()
    # coco_eval.summarize()
    

if __name__ == "__main__":
    # 类别从1开始
    CLASS = {'cone': 1, 'barrier': 2, 'barrel': 3, 'tripod': 4, 'ignore': 5}

    # 两个路径下的文件命名规则需要完全相同, 且文件名为数字
    # pred: 0.json, 1.json, ...
    # gt: 0.json, 1.json, ...
    gt_dir = './MovableBarrierInfer/targets'
    pred_dir = './MovableBarrierInfer/results'

    # 生成的coco格式json文件存储路径
    json_save_path = './MovableBarrierInfer'

    main(gt_dir, pred_dir, json_save_path, CLASS)
