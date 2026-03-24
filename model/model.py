import torch.optim as optim
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

import numpy as np
import pandas as pd


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = Swish()
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.se = SEBlock(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += self.shortcut(x)
        out = self.act1(out) 
        return out


class MultiTaskLoss(nn.Module):
    def __init__(self):
        super(MultiTaskLoss, self).__init__()

    def forward(self, pred_reg, true_reg, pred_cls, true_cls):
        # 回归任务：Huber Loss (SmoothL1Loss) for robustness
        # 使用 SmoothL1Loss作为Huber Loss的替代 (beta=1.0)
        reg_loss = F.smooth_l1_loss(pred_reg.squeeze(), true_reg, reduction='mean')
        
        # 记录MSE用于评估指标
        mse_for_log = F.mse_loss(pred_reg.squeeze(), true_reg, reduction='mean')
        
        # 分类任务：二元交叉熵损失（使用logits）
        # true_cls需要转换为float类型用于BCEWithLogitsLoss
        bce = F.binary_cross_entropy_with_logits(pred_cls.squeeze(), true_cls.float(), reduction='mean')

        # Using fixed weights. 
        loss = reg_loss + 0.1 * bce

        return loss, {'mse': mse_for_log.item(), 'bce': bce.item(), 'log_var_reg': 0.0, 'log_var_cls': 0.0}


class CNN_HAR_KS(nn.Module):
    def __init__(self, dropout=0.1, fc1_dim=128):
        """
        Advanced CNN-HAR-KS with Residual connections, SE-Block, and Swish activation.
        Default values are set to the best parameters found via Bayesian Optimization:
        dropout=0.1, fc1_dim=128.
        """
        super(CNN_HAR_KS, self).__init__()
        
        # --- Layer 1: Initial Convolution ---
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.act1 = Swish()
        
        # --- Layer 2-4: Residual Blocks ---
        # 16x16 -> 8x8 (Downsampling, 32->64)
        self.layer1 = ResidualBlock(32, 64, stride=2)
        
        # 8x8 -> 8x8 (Keep size, 64->64)
        self.layer2 = ResidualBlock(64, 64, stride=1)
        
        # 8x8 -> 4x4 (Downsampling, 64->128)
        self.layer3 = ResidualBlock(64, 128, stride=2)
        
        self.dropout = nn.Dropout(p=dropout)
        
        # --- Layer 5: Fully Connected (MLP) ---
        # Input: 128 * 4 * 4 = 2048 flattened features
        self.flat_dim = 128 * 4 * 4
        self.fc1 = nn.Linear(self.flat_dim, fc1_dim)
        self.act2 = Swish()
        
        # --- Layer 6: Multi-task Output Heads ---
        self.fc_reg = nn.Linear(fc1_dim, 1)
        self.fc_cls = nn.Linear(fc1_dim, 1)

    def forward(self, x):
        # Initial Conv
        x = self.act1(self.bn1(self.conv1(x)))
        
        # ResBlocks
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        # Flattening
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        
        # Dense Layers
        x = self.act2(self.fc1(x))

        # Multi-task Outputs
        reg_output = self.fc_reg(x)      # 回归输出
        cls_output = self.fc_cls(x)      # 分类logits

        return reg_output, cls_output


def train_model(model, train_loader, test_loader, num_epochs=50, device='cpu', 
                lr=0.0012836791765097324, weight_decay=1e-06):
    """
    训练多任务CNN_HAR_KS模型的完整循环。
    Default hyperparameters are set to the best parameters found via Bayesian Optimization.
    """
    model = model.to(device)

    # --- Hyperparameters ---
    # Optimizer: Adam (standard choice)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # 多任务损失函数
    criterion = MultiTaskLoss().to(device)

    # Learning Rate Scheduler: Reduce LR on Plateau
    # 论文提到: "Reduce LR On Plateau, min lr = 0.0001"
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=5, min_lr=0.0001, #verbose=True
    )

    print(f"Starting multi-task training on {device}...")

    history = {
        'train_loss': [], 'test_loss': [],
        'train_reg_mse': [], 'test_reg_mse': [],
        'train_cls_acc': [], 'test_cls_acc': [],
        'train_cls_bce': [], 'test_cls_bce': []
    }

    for epoch in range(num_epochs):
        model.train()  # Set to training mode (enables Dropout)
        running_loss = 0.0
        running_reg_mse = 0.0
        running_cls_bce = 0.0
        correct_train = 0
        total_train = 0

        for images, y_reg, y_cls in train_loader:
            images = images.to(device)
            y_reg = y_reg.to(device)
            y_cls = y_cls.to(device)

            # 1. Zero gradients
            optimizer.zero_grad()

            # 2. Forward pass (多任务输出)
            reg_output, cls_output = model(images)

            # 3. Calculate Multi-task Loss
            loss, loss_components = criterion(reg_output, y_reg, cls_output, y_cls)

            # 4. Backward pass
            loss.backward()

            # 5. Optimization step
            optimizer.step()

            # Statistics
            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            running_reg_mse += loss_components['mse'] * batch_size
            running_cls_bce += loss_components['bce'] * batch_size

            # 分类准确率计算
            cls_pred = (torch.sigmoid(cls_output).view(-1) > 0.5).long()
            correct_train += (cls_pred == y_cls.view(-1)).sum().item()
            total_train += batch_size

        # 计算平均指标
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_reg_mse = running_reg_mse / len(train_loader.dataset)
        epoch_cls_bce = running_cls_bce / len(train_loader.dataset)
        train_acc = correct_train / total_train if total_train > 0 else 0

        # --- Validation Phase ---
        test_metrics = evaluate_model(model, test_loader, criterion, device)
        test_loss = test_metrics['loss']
        test_reg_mse = test_metrics['reg_mse']
        test_cls_bce = test_metrics['cls_bce']
        test_acc = test_metrics['cls_acc']

        # Update Scheduler based on Test Loss
        scheduler.step(test_loss)

        # 保存历史记录
        history['train_loss'].append(epoch_loss)
        history['train_reg_mse'].append(epoch_reg_mse)
        history['train_cls_bce'].append(epoch_cls_bce)
        history['train_cls_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_reg_mse'].append(test_reg_mse)
        history['test_cls_bce'].append(test_cls_bce)
        history['test_cls_acc'].append(test_acc)

        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Loss: {epoch_loss:.4f} | "
              f"Train Reg MSE: {epoch_reg_mse:.4f} | Train Cls Acc: {train_acc:.4f} | "
              f"Test Loss: {test_loss:.4f} | Test Reg MSE: {test_reg_mse:.4f} | Test Cls Acc: {test_acc:.4f}")

    return history

def evaluate_model(model, data_loader, criterion, device):
    """
    评估多任务模型在测试集上的表现
    """
    model.eval()  # Set to evaluation mode (disables Dropout)
    correct = 0
    total = 0
    running_loss = 0.0
    running_reg_mse = 0.0
    running_cls_bce = 0.0

    with torch.no_grad():  # No need to track gradients
        for images, y_reg, y_cls in data_loader:
            images = images.to(device)
            y_reg = y_reg.to(device)
            y_cls = y_cls.to(device)

            # 多任务输出
            reg_output, cls_output = model(images)

            # 计算多任务损失
            loss, loss_components = criterion(reg_output, y_reg, cls_output, y_cls)

            # 统计
            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            running_reg_mse += loss_components['mse'] * batch_size
            running_cls_bce += loss_components['bce'] * batch_size

            # 分类准确率
            cls_pred = (torch.sigmoid(cls_output).view(-1) > 0.5).long()
            correct += (cls_pred == y_cls.view(-1)).sum().item()
            total += batch_size

    avg_loss = running_loss / len(data_loader.dataset)
    avg_reg_mse = running_reg_mse / len(data_loader.dataset)
    avg_cls_bce = running_cls_bce / len(data_loader.dataset)
    accuracy = correct / total if total > 0 else 0

    return {
        'loss': avg_loss,
        'reg_mse': avg_reg_mse,
        'cls_bce': avg_cls_bce,
        'cls_acc': accuracy
    }


def export_test_predictions(
    model,
    test_loader,
    metadata,
    device='cpu',
    benchmark_path=None,
    output_path=None,
):
    """
    导出测试集上的回归预测和分类结果，并按基准文件日期对齐。
    """
    if metadata is None:
        raise ValueError("metadata 不能为空，需包含测试集日期和回归标签的 scaler。")

    required_keys = {'test_dates', 'y_reg_test_raw', 'y_cls_test', 'reg_scaler'}
    missing_keys = required_keys.difference(metadata)
    if missing_keys:
        raise ValueError(f"metadata 缺少必要字段: {sorted(missing_keys)}")

    model = model.to(device)
    model.eval()

    reg_predictions = []
    cls_logits = []
    cls_probabilities = []
    cls_predictions = []

    with torch.no_grad():
        for images, _, _ in test_loader:
            images = images.to(device)
            reg_output, cls_output = model(images)

            reg_predictions.append(reg_output.view(-1).cpu().numpy())

            batch_logits = cls_output.view(-1).cpu().numpy()
            batch_probabilities = torch.sigmoid(cls_output).view(-1).cpu().numpy()

            cls_logits.append(batch_logits)
            cls_probabilities.append(batch_probabilities)
            cls_predictions.append((batch_probabilities > 0.5).astype(int))

    reg_predictions = np.concatenate(reg_predictions)
    cls_logits = np.concatenate(cls_logits)
    cls_probabilities = np.concatenate(cls_probabilities)
    cls_predictions = np.concatenate(cls_predictions)

    reg_scaler = metadata['reg_scaler']
    reg_predictions_raw = reg_scaler.inverse_transform(reg_predictions.reshape(-1, 1)).ravel()

    test_dates = pd.to_datetime(metadata['test_dates'])
    true_rv = np.asarray(metadata['y_reg_test_raw'])
    true_cls = np.asarray(metadata['y_cls_test'])

    if len(test_dates) != len(reg_predictions_raw):
        raise ValueError(
            f"测试集日期数量 ({len(test_dates)}) 与预测数量 ({len(reg_predictions_raw)}) 不一致。"
        )

    prediction_frame = pd.DataFrame(
        {
            'date': test_dates,
            'true_rv': true_rv,
            'cnn_pred': reg_predictions_raw,
            'true_cls': true_cls,
            'cnn_cls_logit': cls_logits,
            'cnn_cls_prob': cls_probabilities,
            'cnn_cls_pred': cls_predictions,
        }
    )
    prediction_frame['date'] = prediction_frame['date'].dt.normalize()

    if benchmark_path is not None:
        benchmark_frame = pd.read_csv(benchmark_path, parse_dates=['date'])
        benchmark_frame['date'] = pd.to_datetime(benchmark_frame['date']).dt.normalize()

        aligned_frame = benchmark_frame[['date', 'true_rv']].merge(
            prediction_frame.drop(columns=['true_rv']),
            on='date',
            how='inner',
            sort=False,
        )

        if aligned_frame.empty:
            raise ValueError("CNN 测试集日期与 benchmark_predictions.csv 没有重叠区间。")
    else:
        aligned_frame = prediction_frame.copy()

    if output_path is not None:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        aligned_frame.to_csv(output_file, index=False)

    return aligned_frame
# 实例化并打印模型结构以验证
if __name__ == "__main__":
    model = CNN_HAR_KS()
    # 假设输入一个 Batch size 为 32 的数据
    dummy_input = torch.randn(32, 1, 16, 16)
    reg_output, cls_output = model(dummy_input)
    print(f"回归输出形状: {reg_output.shape}")  # 应该是 [32, 1]
    print(f"分类输出形状: {cls_output.shape}")  # 应该是 [32, 1]
    print("\n模型结构:")
    print(model)