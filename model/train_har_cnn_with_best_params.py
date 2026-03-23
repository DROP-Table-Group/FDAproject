"""
使用贝叶斯优化得到的最优超参数，复现 har_cnn.ipynb 的训练与评估流程。

默认读取:
- model/bayesian_opt_results/best_params_20260321_190834.json
- data/btcusd_1-min_data.csv
- data/benchmark_predictions.csv

运行示例:
python model/train_har_cnn_with_best_params.py --num-epochs 50
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.stats.diagnostic import acorr_ljungbox

from dataloader import create_dataloaders
from model import MultiTaskLoss, evaluate_model, export_test_predictions # type: ignore
from preprocess import calculate_daily_components, prepare_intraday_data


class CNN_HAR_KS_Configurable(nn.Module):
    """与贝叶斯优化脚本一致的可配置 CNN_HAR_KS 结构。"""

    def __init__(self, dropout: float = 0.3, fc1_dim: int = 64):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=dropout)

        self.fc1 = nn.Linear(64 * 8 * 8, fc1_dim)
        self.relu3 = nn.ReLU()

        self.fc_reg = nn.Linear(fc1_dim, 1)
        self.fc_cls = nn.Linear(fc1_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.pool(x)
        x = self.dropout(x)

        x = x.view(x.size(0), -1)
        x = self.relu3(self.fc1(x))

        reg_output = self.fc_reg(x)
        cls_output = self.fc_cls(x)
        return reg_output, cls_output


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    y_pred_safe = np.maximum(y_pred, 1e-8)
    ratio = y_true / y_pred_safe
    qlike = np.mean(ratio - np.log(ratio) - 1)

    return mse, mae, qlike # type: ignore


def train_model_with_best_params(
    model: nn.Module,
    train_loader,
    test_loader,
    learning_rate: float,
    weight_decay: float,
    num_epochs: int,
    device: torch.device,
) -> Dict[str, list[float]]:
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = MultiTaskLoss().to(device)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.1,
        patience=5,
        min_lr=1e-6,
    )

    history = {
        "train_loss": [],
        "test_loss": [],
        "train_reg_mse": [],
        "test_reg_mse": [],
        "train_cls_acc": [],
        "test_cls_acc": [],
        "train_cls_bce": [],
        "test_cls_bce": [],
    }

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_reg_mse = 0.0
        running_cls_bce = 0.0
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
            running_cls_bce += loss_components["bce"] * batch_size

            cls_pred = (torch.sigmoid(cls_output).view(-1) > 0.5).long()
            correct_train += (cls_pred == y_cls.view(-1)).sum().item()
            total_train += batch_size

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_reg_mse = running_reg_mse / len(train_loader.dataset)
        epoch_cls_bce = running_cls_bce / len(train_loader.dataset)
        train_acc = correct_train / total_train if total_train > 0 else 0.0

        test_metrics = evaluate_model(model, test_loader, criterion, device)
        test_loss = test_metrics["loss"]
        test_reg_mse = test_metrics["reg_mse"]
        test_cls_bce = test_metrics["cls_bce"]
        test_acc = test_metrics["cls_acc"]

        scheduler.step(test_loss)

        history["train_loss"].append(epoch_loss)
        history["test_loss"].append(test_loss)
        history["train_reg_mse"].append(epoch_reg_mse)
        history["test_reg_mse"].append(test_reg_mse)
        history["train_cls_acc"].append(train_acc)
        history["test_cls_acc"].append(test_acc)
        history["train_cls_bce"].append(epoch_cls_bce)
        history["test_cls_bce"].append(test_cls_bce)

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch [{epoch + 1}/{num_epochs}] "
            f"LR: {current_lr:.6g} | "
            f"Train Loss: {epoch_loss:.4f} | "
            f"Train Reg MSE: {epoch_reg_mse:.4f} | Train Cls Acc: {train_acc:.4f} | "
            f"Test Loss: {test_loss:.4f} | Test Reg MSE: {test_reg_mse:.4f} | Test Cls Acc: {test_acc:.4f}"
        )

    return history


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    parser = argparse.ArgumentParser(description="用 BO 最优参数训练并评估 CNN-HAR-KS")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=project_root / "data" / "btcusd_1-min_data.csv",
        help="1分钟级数据 CSV 路径",
    )
    parser.add_argument(
        "--benchmark-path",
        type=Path,
        default=project_root / "data" / "benchmark_predictions.csv",
        help="基准预测 CSV 路径",
    )
    parser.add_argument(
        "--best-params-path",
        type=Path,
        default=script_dir / "bayesian_opt_results" / "best_params_20260321_190834.json",
        help="最优超参数 JSON 路径",
    )
    parser.add_argument("--num-epochs", type=int, default=50, help="训练轮数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument(
        "--model-output",
        type=Path,
        default=script_dir / "cnn_har_ks_multitask_model_best_params.pth",
        help="模型权重输出路径",
    )
    parser.add_argument(
        "--predictions-output",
        type=Path,
        default=project_root / "data" / "cnn_har_ks_multitask_predictions_best_params.csv",
        help="预测结果输出路径",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if not args.best_params_path.exists():
        raise FileNotFoundError(f"未找到最优参数文件: {args.best_params_path}")

    with args.best_params_path.open("r", encoding="utf-8") as f:
        best_params: Dict[str, Any] = json.load(f)

    learning_rate = float(best_params["learning_rate"])
    weight_decay = float(best_params["weight_decay"])
    dropout = float(best_params["dropout"])
    fc1_dim = int(best_params["fc1_dim"])
    batch_size = int(best_params["batch_size"])

    print("加载到的最优超参数:")
    print(json.dumps(best_params, ensure_ascii=False, indent=2))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    raw_data = pd.read_csv(args.data_path, index_col=0)
    intraday_data = prepare_intraday_data(raw_data)
    daily_data = calculate_daily_components(intraday_data)

    benchmark_frame = pd.read_csv(args.benchmark_path, parse_dates=["date"])
    benchmark_test_start = benchmark_frame["date"].min()

    train_loader, test_loader, metadata = create_dataloaders(
        daily_data,
        batch_size=batch_size,
        return_metadata=True,
        test_start_date=benchmark_test_start,
    )

    print(f"CNN测试集起始日期: {metadata['test_dates'][0].date()}")
    print(f"基准测试集起始日期: {benchmark_test_start.date()}")
    print(f"训练集batch数量: {len(train_loader)}")
    print(f"测试集batch数量: {len(test_loader)}")

    model = CNN_HAR_KS_Configurable(dropout=dropout, fc1_dim=fc1_dim).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数数量: {total_params:,}")
    print(f"可训练参数数量: {trainable_params:,}")

    print("开始训练多任务模型...")
    history = train_model_with_best_params(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        num_epochs=args.num_epochs,
        device=device,
    )

    print("\n多任务训练完成!")
    print(f"最终测试损失: {history['test_loss'][-1]:.4f}")
    print(f"最终测试回归MSE: {history['test_reg_mse'][-1]:.4f}")
    print(f"最终测试分类准确率: {history['test_cls_acc'][-1]:.4f}")

    best_epoch_cls = int(np.argmax(history["test_cls_acc"]))
    best_epoch_reg = int(np.argmin(history["test_reg_mse"]))
    print(f"最佳测试分类准确率: {history['test_cls_acc'][best_epoch_cls]:.4f} (Epoch {best_epoch_cls + 1})")
    print(f"最佳测试回归MSE: {history['test_reg_mse'][best_epoch_reg]:.4f} (Epoch {best_epoch_reg + 1})")

    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.model_output)
    print(f"模型已保存到: {args.model_output.resolve()}")

    aligned_predictions = export_test_predictions(
        model=model,
        test_loader=test_loader,
        metadata=metadata,
        device=device,
        benchmark_path=args.benchmark_path,
        output_path=args.predictions_output,
    )

    print(f"预测结果已保存到: {args.predictions_output.resolve()}")
    print(f"输出行数: {len(aligned_predictions)}")
    print(f"起始日期: {aligned_predictions['date'].min().date()}")
    print(f"结束日期: {aligned_predictions['date'].max().date()}")

    benchmark_eval = pd.read_csv(args.benchmark_path, parse_dates=["date"])
    benchmark_eval["date"] = pd.to_datetime(benchmark_eval["date"]).dt.normalize()

    evaluation_frame = aligned_predictions.merge(
        benchmark_eval[["date", "garch_pred", "ewma_pred"]],
        on="date",
        how="left",
    )

    model_predictions = {
        "CNN-HAR-KS": evaluation_frame["cnn_pred"],
        "GARCH(1,1)": evaluation_frame["garch_pred"],
        "EWMA": evaluation_frame["ewma_pred"],
    }

    metrics_rows = []
    for model_name, predictions in model_predictions.items():
        valid_frame = evaluation_frame[["true_rv"]].copy()
        valid_frame["pred"] = predictions
        valid_frame = valid_frame.dropna()

        mse, mae, qlike = compute_metrics(valid_frame["true_rv"], valid_frame["pred"])
        metrics_rows.append(
            {
                "Model": model_name,
                "MSE": mse,
                "MAE": mae,
                "QLIKE": qlike,
                "SampleSize": len(valid_frame),
            }
        )

    metrics_summary = pd.DataFrame(metrics_rows)
    print("\n样本外误差指标:")
    print(metrics_summary.to_string(index=False))

    cnn_mse = metrics_summary.loc[metrics_summary["Model"] == "CNN-HAR-KS", "MSE"].iloc[0]
    garch_mse = metrics_summary.loc[metrics_summary["Model"] == "GARCH(1,1)", "MSE"].iloc[0]
    ewma_mse = metrics_summary.loc[metrics_summary["Model"] == "EWMA", "MSE"].iloc[0]

    print(f"CNN-HAR-KS 相比 GARCH 的 MSE 改进: {(garch_mse - cnn_mse) / garch_mse * 100:.2f}%")
    print(f"CNN-HAR-KS 相比 EWMA 的 MSE 改进: {(ewma_mse - cnn_mse) / ewma_mse * 100:.2f}%")

    residual_frame = evaluation_frame[["date", "true_rv", "cnn_pred", "garch_pred", "ewma_pred"]].copy()
    residual_frame["cnn_residual"] = residual_frame["true_rv"] - residual_frame["cnn_pred"]
    residual_frame["garch_residual"] = residual_frame["true_rv"] - residual_frame["garch_pred"]
    residual_frame["ewma_residual"] = residual_frame["true_rv"] - residual_frame["ewma_pred"]

    print("\n残差 Ljung-Box 检验:")
    for label, column in [
        ("CNN-HAR-KS", "cnn_residual"),
        ("GARCH(1,1)", "garch_residual"),
        ("EWMA", "ewma_residual"),
    ]:
        residuals = residual_frame[column].dropna()
        lb_result = acorr_ljungbox(residuals, lags=[10, 20], return_df=True)
        print(f"\n{label}:")
        print(lb_result.to_string())


if __name__ == "__main__":
    main()
