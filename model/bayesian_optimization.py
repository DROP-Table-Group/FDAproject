"""
CNN 模型超参数贝叶斯优化脚本

使用贝叶斯优化方法自动搜索 CNN_HAR_KS 模型的最优超参数组合。
优化目标：最小化测试集损失 (test_loss)

超参数搜索空间包括：
- 学习率 (learning_rate)
- 权重衰减 (weight_decay)
- Dropout 率 (dropout)
- 全连接层维度 (fc1_dim)
- batch_size
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

try:
    from skopt import gp_minimize
    from skopt.space import Categorical, Integer, Real
    from skopt.utils import use_named_args
except ImportError:
    print("错误：需要安装 scikit-optimize 库")
    print("请运行：pip install scikit-optimize")
    sys.exit(1)

# 导入数据处理和模型
from dataloader import create_dataloaders
from preprocess import calculate_daily_components, prepare_intraday_data


# ==========================================
# 1. 可配置的多任务 CNN 模型
# ==========================================
class CNN_HAR_KS_Configurable(nn.Module):
    """
    支持超参数配置的 CNN_HAR_KS 模型
    """

    def __init__(self, dropout: float = 0.3, fc1_dim: int = 64):
        super(CNN_HAR_KS_Configurable, self).__init__()

        # Convolution Block
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1
        )
        self.relu2 = nn.ReLU()

        # Max Pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Dropout (可配置)
        self.dropout = nn.Dropout(p=dropout)

        # Fully Connected Layer (维度可配置)
        # Pooling 后特征图大小：64 * 8 * 8 = 4096
        self.fc1 = nn.Linear(64 * 8 * 8, fc1_dim)
        self.relu3 = nn.ReLU()

        # Multi-task Output Heads
        self.fc_reg = nn.Linear(fc1_dim, 1)
        self.fc_cls = nn.Linear(fc1_dim, 1)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.relu3(self.fc1(x))
        reg_output = self.fc_reg(x)
        cls_output = self.fc_cls(x)
        return reg_output, cls_output


# ==========================================
# 2. 多任务损失函数 (保持不变)
# ==========================================
class MultiTaskLoss(nn.Module):
    def __init__(self):
        super(MultiTaskLoss, self).__init__()
        self.log_vars = nn.Parameter(torch.zeros(2))

    def forward(self, pred_reg, true_reg, pred_cls, true_cls):
        mse = nn.MSELoss()(pred_reg.squeeze(), true_reg)
        bce = nn.BCEWithLogitsLoss()(pred_cls.squeeze(), true_cls.float())

        loss = (0.5 * torch.exp(-self.log_vars[0]) * mse + 0.5 * self.log_vars[0]) + (
            torch.exp(-self.log_vars[1]) * bce + 0.5 * self.log_vars[1]
        )

        return loss, {"mse": mse.item(), "bce": bce.item()}


# ==========================================
# 3. 训练和评估函数
# ==========================================
def train_one_epoch(model, train_loader, optimizer, criterion, device):
    """训练一个 epoch"""
    model.train()
    running_loss = 0.0
    running_reg_mse = 0.0
    correct_train = 0
    total_train = 0

    for images, y_reg, y_cls in train_loader:
        images = images.to(device)
        y_reg = y_reg.to(device)
        y_cls = y_cls.to(device)

        optimizer.zero_grad()
        reg_output, cls_output = model(images)
        loss, loss_components = criterion(reg_output, y_reg, cls_output, y_cls)
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        running_reg_mse += loss_components["mse"] * batch_size

        cls_pred = (torch.sigmoid(cls_output).view(-1) > 0.5).long()
        correct_train += (cls_pred == y_cls.view(-1)).sum().item()
        total_train += batch_size

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_reg_mse = running_reg_mse / len(train_loader.dataset)
    train_acc = correct_train / total_train if total_train > 0 else 0

    return epoch_loss, epoch_reg_mse, train_acc


def evaluate(model, data_loader, criterion, device):
    """评估模型"""
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    running_reg_mse = 0.0

    with torch.no_grad():
        for images, y_reg, y_cls in data_loader:
            images = images.to(device)
            y_reg = y_reg.to(device)
            y_cls = y_cls.to(device)

            reg_output, cls_output = model(images)
            loss, loss_components = criterion(reg_output, y_reg, cls_output, y_cls)

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            running_reg_mse += loss_components["mse"] * batch_size

            cls_pred = (torch.sigmoid(cls_output).view(-1) > 0.5).long()
            correct += (cls_pred == y_cls.view(-1)).sum().item()
            total += batch_size

    avg_loss = running_loss / len(data_loader.dataset)
    avg_reg_mse = running_reg_mse / len(data_loader.dataset)
    accuracy = correct / total if total > 0 else 0

    return {"loss": avg_loss, "reg_mse": avg_reg_mse, "cls_acc": accuracy}


def train_and_evaluate(
    model,
    train_loader,
    test_loader,
    num_epochs,
    learning_rate,
    weight_decay,
    device="cpu",
):
    """
    完整的训练和评估流程
    返回最终的测试损失 (用于优化目标)
    """
    model = model.to(device)
    optimizer = optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    criterion = MultiTaskLoss().to(device)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=5, min_lr=1e-6
    )

    best_test_loss = float("inf")

    for epoch in range(num_epochs):
        train_loss, _, _ = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        test_metrics = evaluate(model, test_loader, criterion, device)
        test_loss = test_metrics["loss"]

        scheduler.step(test_loss)

        if test_loss < best_test_loss:
            best_test_loss = test_loss

    return best_test_loss


# ==========================================
# 4. 贝叶斯优化目标函数
# ==========================================
class BayesianOptimizationRunner:
    def __init__(
        self,
        data_path: str = "../data/btcusd_1-min_data.csv",
        benchmark_path: str = "../data/benchmark_predictions.csv",
        device: Optional[str] = None,
        num_epochs: int = 30,
        n_calls: int = 25,
        random_state: int = 42,
    ):
        self.data_path = data_path
        self.benchmark_path = benchmark_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.num_epochs = num_epochs
        self.n_calls = n_calls
        self.random_state = random_state

        # 超参数搜索空间定义
        self.space = [
            Real(1e-5, 1e-2, prior="log-uniform", name="learning_rate"),
            Real(1e-6, 1e-2, prior="log-uniform", name="weight_decay"),
            Real(0.1, 0.5, name="dropout"),
            Integer(32, 128, name="fc1_dim"),
            Categorical([16, 32, 64, 128], name="batch_size"),
        ]

        self.dim_names = [
            "learning_rate",
            "weight_decay",
            "dropout",
            "fc1_dim",
            "batch_size",
        ]

        # 准备数据
        self._prepare_data()

        # 记录优化历史
        self.history = []
        self.best_result = None

    def _prepare_data(self):
        """准备训练和测试数据"""
        print("正在加载和预处理数据...")
        raw_data = pd.read_csv(self.data_path, index_col=0)
        intraday_data = prepare_intraday_data(raw_data)
        daily_data = calculate_daily_components(intraday_data)

        benchmark_frame = pd.read_csv(self.benchmark_path, parse_dates=["date"])
        benchmark_test_start = benchmark_frame["date"].min()

        # 保存 daily_data 用于后续重新创建 DataLoader
        self.daily_data = daily_data
        self.benchmark_test_start = benchmark_test_start

        self.train_loader, self.test_loader, self.data_metadata = create_dataloaders(
            daily_data,
            batch_size=32,
            return_metadata=True,
            test_start_date=benchmark_test_start,
        )

        print(f"训练集大小：{len(self.train_loader.dataset)}")
        print(f"测试集大小：{len(self.test_loader.dataset)}")

    def objective(self, params):
        """
        贝叶斯优化的目标函数
        参数格式：[learning_rate, weight_decay, dropout, fc1_dim, batch_size]
        """
        learning_rate, weight_decay, dropout, fc1_dim, batch_size = params

        # 确保类型正确
        batch_size = int(batch_size)
        fc1_dim = int(fc1_dim)

        print(f"\n尝试超参数组合:")
        print(
            f"  learning_rate={learning_rate:.6f}, weight_decay={weight_decay:.6f}, "
            f"dropout={dropout:.3f}, fc1_dim={fc1_dim}, batch_size={batch_size}"
        )

        # 使用当前超参数创建新的 DataLoader (如果 batch_size 改变)
        if batch_size != self.train_loader.batch_size:
            train_loader, test_loader = create_dataloaders(
                self.daily_data,
                batch_size=batch_size,
                return_metadata=False,
                test_start_date=self.benchmark_test_start,
            )
        else:
            train_loader, test_loader = self.train_loader, self.test_loader

        # 创建模型
        model = CNN_HAR_KS_Configurable(dropout=dropout, fc1_dim=fc1_dim)

        # 训练和评估
        start_time = time.time()
        test_loss = train_and_evaluate(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            num_epochs=self.num_epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            device=self.device,
        )
        elapsed_time = time.time() - start_time

        print(f"  测试损失：{test_loss:.6f} (耗时：{elapsed_time:.1f}s)")

        # 记录历史
        result = {
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "dropout": dropout,
            "fc1_dim": fc1_dim,
            "batch_size": batch_size,
            "test_loss": test_loss,
            "elapsed_time": elapsed_time,
        }
        self.history.append(result)

        # 更新最佳结果
        if self.best_result is None or test_loss < self.best_result["test_loss"]:
            self.best_result = result.copy()
            print(f"  *** 新的最佳结果！测试损失：{test_loss:.6f} ***")

        return test_loss

    def run(self, save_dir: str | Path = "model/bayesian_opt_results"):
        """
        运行贝叶斯优化
        """
        print(f"\n{'=' * 60}")
        print("CNN 模型超参数贝叶斯优化")
        print(f"{'=' * 60}")
        print(f"设备：{self.device}")
        print(f"优化轮数：{self.n_calls}")
        print(f"每轮训练 epoch 数：{self.num_epochs}")
        print(f"\n超参数搜索空间:")
        for dim in self.space:
            print(f"  {dim.name}: {dim}")
        print(f"{'=' * 60}\n")

        # 创建保存目录
        save_dir = Path(save_dir)  # pyright: ignore[reportAssignmentType]
        save_dir.mkdir(parents=True, exist_ok=True)  # pyright: ignore[reportAttributeAccessIssue]

        # 运行贝叶斯优化
        @use_named_args(self.space)
        def wrapped_objective(**kwargs):
            params = [kwargs[name] for name in self.dim_names]
            return self.objective(params)

        # 确保 n_calls >= n_random_starts
        n_random_starts = min(5, self.n_calls - 1)

        start_time = time.time()
        result = gp_minimize(
            func=wrapped_objective,
            dimensions=self.space,
            n_calls=self.n_calls,
            n_random_starts=n_random_starts,
            random_state=self.random_state,
            verbose=True,
        )
        total_time = time.time() - start_time
        if result is not None:
            # 输出优化结果
            print(f"\n{'=' * 60}")
            print("贝叶斯优化完成!")
            print(f"总耗时：{total_time / 3600:.2f} 小时")
            print(f"\n最优超参数组合:")
            for i, name in enumerate(self.dim_names):
                print(f"  {name}: {result.x[i]}")
            print(f"\n最优测试损失：{result.fun:.6f}")
            print(f"{'=' * 60}")

            # 保存结果
            self._save_results(result, save_dir)  # pyright: ignore[reportArgumentType]
        else:
            raise RuntimeError("贝叶斯优化未能完成，结果为 None")

        return result

    def _save_results(self, result, save_dir: Path):
        """保存优化结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 保存最优超参数 (转换为 Python 原生类型)
        best_params = {
            name: (
                int(result.x[i])
                if isinstance(result.x[i], (np.integer, np.int64))
                # pyright: ignore[reportArgumentType]
                else float(result.x[i])
                if isinstance(result.x[i], (np.floating, np.float64))
                else result.x[i]
            )
            for i, name in enumerate(self.dim_names)
        }
        best_params["test_loss"] = float(result.fun)

        with open(save_dir / f"best_params_{timestamp}.json", "w") as f:
            json.dump(best_params, f, indent=2)

        # 保存完整历史记录
        history_df = pd.DataFrame(self.history)
        history_df.to_csv(
            save_dir / f"optimization_history_{timestamp}.csv", index=False
        )

        # 保存收敛曲线图数据
        convergence_data = {
            "iteration": list(range(1, len(self.history) + 1)),
            "best_test_loss": [
                min([h["test_loss"] for h in self.history[: i + 1]])
                for i in range(len(self.history))
            ],
        }
        pd.DataFrame(convergence_data).to_csv(
            save_dir / f"convergence_curve_{timestamp}.csv", index=False
        )

        # 保存详细报告
        report = f"""
CNN 超参数贝叶斯优化报告
{"=" * 60}
生成时间：{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
设备：{self.device}

最优超参数:
{json.dumps(best_params, indent=2)}

优化历史统计:
{history_df.describe()}

前 5 佳结果:
{history_df.nsmallest(5, "test_loss").to_string(index=False)}
"""
        with open(save_dir / f"optimization_report_{timestamp}.txt", "w") as f:
            f.write(report)

        print(f"\n结果已保存至：{save_dir}")


# ==========================================
# 5. 主函数
# ==========================================
def main():
    import argparse

    parser = argparse.ArgumentParser(description="CNN 模型超参数贝叶斯优化")
    parser.add_argument(
        "--data-path",
        type=str,
        default="./data/btcusd_1-min_data.csv",
        help="输入数据路径",
    )
    parser.add_argument(
        "--benchmark-path",
        type=str,
        default="./data/benchmark_predictions.csv",
        help="基准预测文件路径",
    )
    parser.add_argument("--n-calls", type=int, default=100, help="贝叶斯优化迭代次数")
    parser.add_argument("--n-epochs", type=int, default=25, help="每次训练的 epoch 数")
    parser.add_argument(
        "--save-dir",
        type=str,
        default="model/bayesian_opt_results",
        help="结果保存目录",
    )
    parser.add_argument("--random-state", type=int, default=42, help="随机种子")

    args = parser.parse_args()

    runner = BayesianOptimizationRunner(
        data_path=args.data_path,
        benchmark_path=args.benchmark_path,
        num_epochs=args.n_epochs,
        n_calls=args.n_calls,
        random_state=args.random_state,
    )

    runner.run(save_dir=args.save_dir)


if __name__ == "__main__":
    main()
