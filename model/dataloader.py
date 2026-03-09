import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. 核心数据处理逻辑 (Pandas -> Numpy)
# ==========================================
class HARDataProcessor:
    def __init__(self, daily_df, target_col='RV', test_ratio=0.2):
        """
        daily_df: 包含16个HAR组件的日度DataFrame
        target_col: 用于计算标签的基础列 (通常是 RV)
        test_ratio: 测试集占比 (时间序列切分)
        """
        self.raw_df = daily_df.copy()
        self.target_col = target_col
        self.test_ratio = test_ratio
        
        # 确保列的顺序固定 (16个特征)
        # 这一步很重要，这决定了图像的"高度" (Height) 对应的特征含义
        self.feature_cols = daily_df.columns.tolist() 
        assert len(self.feature_cols) == 16, f"输入特征必须刚好是16个，当前为 {len(self.feature_cols)}"

    def process(self):
        """
        执行完整的预处理流程：
        1. 构造 16x16 图像矩阵
        2. 构造 标签 (0/1)
        3. 划分 训练/测试集
        4. 标准化 (只在训练集上Fit，防止数据泄露)
        """
        
        # --- A. 标签生成 (Target Generation) ---
        # 论文公式 (2): RVD = 1 if RV_t < RV_{t-1} else 0
        # 也就是预测 "今天" 相对于 "昨天" 是涨还是跌
        # 注意：这里的 Label 是对应 index t 的
        rv = self.raw_df[self.target_col]
        # diff = RV_t - RV_{t-1}
        # 如果 diff < 0 (波动率下降), label = 1
        labels = (rv.diff() < 0).astype(int)
        
        # --- B. 特征图像构造 (Image Construction) ---
        # 图像结构: (N, 16 features, 16 time_windows)
        # 关键点: 我们用 t-1 的信息来预测 t 的标签
        # 所以特征数据必须全部 shift(1)
        df_shifted = self.raw_df.shift(1)
        
        tensor_slices = []
        
        # 第1列: Lag-1 (即 shift(1) 后的原始值)
        tensor_slices.append(df_shifted.values)
        
        # 第2-16列: MA(6) 到 MA(20)
        # 注意：是对 shift(1) 后的数据求滚动平均
        for w in range(6, 21): # 6, 7, ..., 20
            # min_periods=w 保证数据不足时产生 NaN，方便后续清洗
            ma = df_shifted.rolling(window=w).mean().values
            tensor_slices.append(ma)
            
        # 堆叠数据
        # 现在的 list 结构是 16 个 (N, 16_features) 的数组
        # 我们希望最终形状是 (N, 16_features, 16_windows)
        # stack axis=2 会把列表的维度放在最后
        X_all = np.stack(tensor_slices, axis=2) 
        
        # --- C. 数据清洗 (Drop NaN) ---
        # 因为做了 shift(1) 和 rolling(20)，前21行会有 NaN
        valid_mask = ~np.isnan(X_all).any(axis=(1, 2))
        
        X_valid = X_all[valid_mask]
        y_valid = labels.iloc[valid_mask].values # 对齐索引
        
        # --- D. 划分训练/测试集 (Time Series Split) ---
        # 金融数据不能随机打乱 (Shuffle)，必须按时间切分
        split_idx = int(len(X_valid) * (1 - self.test_ratio))
        
        X_train, X_test = X_valid[:split_idx], X_valid[split_idx:]
        y_train, y_test = y_valid[:split_idx], y_valid[split_idx:]
        
        # --- E. 归一化 (Normalization) ---
        # CNN 对数值幅度敏感。我们需要对图像进行标准化。
        # 策略：对每一个特征通道 (16个 Feature) 独立进行标准化。
        # 比如 "RV" 行的数值很小，"Daily Return" 可能是负数，需要各自归一化。
        
        # 形状变换方便 scaler: (N, 16, 16) -> (N*16, 16 features) ? 不对
        # 应该是对每个 Feature (行) 在时间维度和样本维度上做归一化
        # 简单做法：对 (N, 16, 16) 展平为 (N, 256) 做 scaler，或者按通道。
        # 论文推荐做法通常是按 Feature 类型标准化。
        
        X_train_norm, X_test_norm = self._normalize_by_feature(X_train, X_test)
        
        # 增加 Channel 维度适配 PyTorch Conv2d: (N, 1, 16, 16)
        X_train_norm = X_train_norm[:, np.newaxis, :, :]
        X_test_norm = X_test_norm[:, np.newaxis, :, :]
        
        return {
            'X_train': X_train_norm, 'y_train': y_train,
            'X_test': X_test_norm,   'y_test': y_test
        }

    def _normalize_by_feature(self, train, test):
        # 输入形状: (Samples, Features=16, Windows=16)
        # 我们希望对每一个 Feature (第1维) 单独计算 Mean/Std
        
        n_features = train.shape[1]
        train_norm = np.zeros_like(train)
        test_norm = np.zeros_like(test)
        
        # 对 16 个 HAR 组件分别遍历
        for i in range(n_features):
            scaler = StandardScaler()
            
            # 取出所有样本的所有时间窗口，展平进行拟合
            # 这一行的数据: (Samples, Windows)
            train_slice = train[:, i, :]
            test_slice = test[:, i, :]
            
            # Fit on Train
            scaler.fit(train_slice)
            
            # Transform both
            train_norm[:, i, :] = scaler.transform(train_slice)
            test_norm[:, i, :] = scaler.transform(test_slice)
            
        return train_norm, test_norm

# ==========================================
# 2. PyTorch Dataset 定义
# ==========================================
class FinancialImageDataset(Dataset):
    def __init__(self, X, y):
        """
        X: numpy array (N, 1, 16, 16)
        y: numpy array (N,)
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        # Classification 需要 Long 类型标签
        self.y = torch.tensor(y, dtype=torch.long)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ==========================================
# 3. 辅助函数: 创建 DataLoaders
# ==========================================
def create_dataloaders(daily_df, batch_size=32, test_ratio=0.2):
    # 1. 处理数据
    processor = HARDataProcessor(daily_df, test_ratio=test_ratio)
    data_dict = processor.process()
    
    print(f"Train shape: {data_dict['X_train'].shape}")
    print(f"Test shape:  {data_dict['X_test'].shape}")
    
    # 2. 创建 Dataset 实例
    train_dataset = FinancialImageDataset(data_dict['X_train'], data_dict['y_train'])
    test_dataset = FinancialImageDataset(data_dict['X_test'], data_dict['y_test'])
    
    # 3. 创建 DataLoader
    # 训练集可以 shuffle，测试集通常不 shuffle (方便画时序图，虽然分类问题无所谓)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

# ==========================================
# 4. 运行示例 (生成假数据测试)
# ==========================================
if __name__ == "__main__":
    # 模拟一个符合论文结构的 DataFrame (1000天, 16个特征)
    dates = pd.date_range(start='2020-01-01', periods=1000, freq='D')
    feature_names = [
        'RV', 'BPV', 'ABD_jump', 'ABD_CSP', 'BNS_jump', 'BNS_CSP', 
        'Jo_jump', 'Jo_CSP', 'RS_plus', 'RS_minus', 'Daily_ret', 
        'Neg_RV', 'SJ', 'SJ_plus', 'SJ_minus', 'TQ'
    ]
    
    # 生成随机数据
    dummy_data = np.random.randn(1000, 16)
    # 保证 RV 是正数
    dummy_data[:, 0] = np.abs(dummy_data[:, 0]) 
    
    df_daily_har = pd.DataFrame(dummy_data, columns=feature_names, index=dates)
    
    print("原始数据预览:")
    print(df_daily_har.head(3))
    
    # --- 生成 DataLoader ---
    train_dl, test_dl = create_dataloaders(df_daily_har, batch_size=64)
    
    # --- 检查一个 Batch ---
    images, labels = next(iter(train_dl))
    
    print("\nBatch 信息:")
    print(f"Image Batch Shape: {images.shape}") # 预期: [64, 1, 16, 16]
    print(f"Label Batch Shape: {labels.shape}") # 预期: [64]
    print(f"Label Example: {labels[:5]}")
    
    # 这就是可以直接喂给 CNN 模型 (forward) 的数据了
    # model(images) -> output