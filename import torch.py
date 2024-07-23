import torch
import torch.nn as nn

class CIoULoss(nn.Module):
    def __init__(self):
        super(CIoULoss, self).__init__()

    def forward(self, pred_boxes, target_boxes):
        """
        计算 CIoU 损失

        :param pred_boxes: 预测边界框，形状为 [N, 4]，其中 N 是样本数量
        :param target_boxes: 真实边界框，形状为 [N, 4]
        :return: CIoU 损失值
        """
        # 计算每个边界框的 (x1, y1, x2, y2)
        pred_x1, pred_y1, pred_x2, pred_y2 = pred_boxes.split(1, dim=1)
        target_x1, target_y1, target_x2, target_y2 = target_boxes.split(1, dim=1)

        # 计算交集区域的 (x1, y1, x2, y2)
        inter_x1 = torch.max(pred_x1, target_x1)
        inter_y1 = torch.max(pred_y1, target_y1)
        inter_x2 = torch.min(pred_x2, target_x2)
        inter_y2 = torch.min(pred_y2, target_y2)

        # 计算交集区域的面积
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)

        # 计算预测框和目标框的面积
        pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
        target_area = (target_x2 - target_x1) * (target_y2 - target_y1)

        # 计算并集区域的面积
        union_area = pred_area + target_area - inter_area

        # 计算 IoU
        iou = inter_area / torch.clamp(union_area, min=1e-6)

        # 计算中心点距离
        pred_center_x = (pred_x1 + pred_x2) / 2
        pred_center_y = (pred_y1 + pred_y2) / 2
        target_center_x = (target_x1 + target_x2) / 2
        target_center_y = (target_y1 + target_y2) / 2

        center_distance = torch.sqrt((pred_center_x - target_center_x)**2 + (pred_center_y - target_center_y)**2)

        # 计算长宽比
        pred_aspect_ratio = (pred_x2 - pred_x1) / (pred_y2 - pred_y1)
        target_aspect_ratio = (target_x2 - target_x1) / (target_y2 - target_y1)
        aspect_ratio_diff = torch.abs(pred_aspect_ratio - target_aspect_ratio)

        # 计算 CIoU 损失
        ciou_loss = 1 - iou + (center_distance / torch.clamp(torch.sqrt(pred_area + target_area), min=1e-6)) + aspect_ratio_diff

        return ciou_loss.mean()
    
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 示例数据
num_samples = 100
pred_boxes = torch.rand(num_samples, 4)  # 随机生成预测框
target_boxes = torch.rand(num_samples, 4)  # 随机生成真实框

# 定义数据集和数据加载器
dataset = TensorDataset(pred_boxes, target_boxes)
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

# 定义模型（这里我们用一个简单的线性层作为示例）
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(4, 4)  # 输入和输出都是边界框

    def forward(self, x):
        return self.fc(x)

# 初始化模型、损失函数和优化器
model = SimpleModel()
criterion = CIoULoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 训练循环
for epoch in range(10):  # 训练10个epoch
    for batch_pred_boxes, batch_target_boxes in dataloader:
        optimizer.zero_grad()

        # 前向传播
        outputs = model(batch_pred_boxes)

        # 计算损失
        loss = criterion(outputs, batch_target_boxes)

        # 反向传播和优化
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch + 1}, Loss: {loss.item()}')