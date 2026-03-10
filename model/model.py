import torch.optim as optim
import time
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiTaskLoss(nn.Module):
    def __init__(self):
        super(MultiTaskLoss, self).__init__()
        # 定义两个可学习的参数 (log variance)
        self.log_vars = nn.Parameter(torch.zeros(2))

    def forward(self, pred_reg, true_reg, pred_cls, true_cls):
        # 回归任务：均方误差损失
        mse = nn.MSELoss()(pred_reg.squeeze(), true_reg)
        # 分类任务：二元交叉熵损失（使用logits）
        # true_cls需要转换为float类型用于BCEWithLogitsLoss
        bce = nn.BCEWithLogitsLoss()(pred_cls.squeeze(), true_cls.float())

        # 这里的公式推导来自于贝叶斯深度学习
        # 简单理解：如果某个任务很难（Loss大），模型会自动增大分母(var)来降低该任务对总Loss的贡献
        loss = (0.5 * torch.exp(-self.log_vars[0]) * mse + 0.5 * self.log_vars[0]) + \
               (torch.exp(-self.log_vars[1]) * bce + 0.5 * self.log_vars[1])

        return loss, {'mse': mse.item(), 'bce': bce.item(), 'log_var_reg': self.log_vars[0].item(), 'log_var_cls': self.log_vars[1].item()}


class CNN_HAR_KS(nn.Module):
    def __init__(self):
        super(CNN_HAR_KS, self).__init__()
        
        # --- Layer 1: Convolution ---
        # Input: (Batch, 1, 16, 16)
        # Output: (Batch, 32, 16, 16)
        # Padding=1 ensures the size remains 16x16 with a 3x3 kernel
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        
        # --- Layer 2: Convolution ---
        # Input: (Batch, 32, 16, 16)
        # Output: (Batch, 64, 16, 16)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        
        # --- Layer 3: Max Pooling ---
        # Input: (Batch, 64, 16, 16)
        # Output: (Batch, 64, 8, 8) -> Flat features: 64 * 8 * 8 = 4096
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # --- Layer 4: Dropout ---
        # Paper Appendix C: Dropout = 0.3
        self.dropout = nn.Dropout(p=0.3)
        
        # --- Layer 5: Fully Connected (MLP) ---
        # Input: 4096 flattened features
        # Output: 64
        self.fc1 = nn.Linear(64 * 8 * 8, 64)
        self.relu3 = nn.ReLU()
        
        # --- Layer 6: Multi-task Output Heads ---
        # 回归头：预测标准化后的RV值（输出1个标量）
        self.fc_reg = nn.Linear(64, 1)
        # 分类头：预测波动率是否下降（输出1个logit，用于二元分类）
        self.fc_cls = nn.Linear(64, 1)

    def forward(self, x):
        # Convolution Block
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.pool(x)
        
        # Regularization
        x = self.dropout(x)
        
        # Flattening: (Batch, 64, 8, 8) -> (Batch, 4096)
        x = x.view(x.size(0), -1)
        
        # Dense Layers
        x = self.relu3(self.fc1(x))

        # Multi-task Outputs
        reg_output = self.fc_reg(x)      # 回归输出: (batch_size, 1)
        cls_output = self.fc_cls(x)      # 分类logits: (batch_size, 1)

        return reg_output, cls_output


def train_model(model, train_loader, test_loader, num_epochs=50, device='cpu'):
    """
    训练多任务CNN_HAR_KS模型的完整循环
    """
    model = model.to(device)

    # --- Hyperparameters from Paper Appendix C ---
    # Optimizer: Adam (standard choice)
    # Learning Rate: 0.001 (search space start)
    # L2 Regularization (Weight Decay): 0.001
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)

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
            cls_pred = (torch.sigmoid(cls_output) > 0.5).long().squeeze()
            correct_train += (cls_pred == y_cls).sum().item()
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
            cls_pred = (torch.sigmoid(cls_output) > 0.5).long().squeeze()
            correct += (cls_pred == y_cls).sum().item()
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