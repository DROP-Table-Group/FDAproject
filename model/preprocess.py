import pandas as pd
import numpy as np
import math

def prepare_intraday_data(df_1min):
    # 1. 确保索引是DatetimeIndex（如果索引是数值型时间戳，则转换为日期时间）
    if not isinstance(df_1min.index, pd.DatetimeIndex):
        # 假设索引是以秒为单位的Unix时间戳
        df_1min = df_1min.copy()
        df_1min.index = pd.to_datetime(df_1min.index, unit='s')
    
    # 2. 重采样为5分钟级别 (取每5分钟的最后一个收盘价)
    df_5min = df_1min['Close'].resample('5min').last().dropna()
    
    # 3. 计算对数收益率
    df_5min = pd.DataFrame(df_5min)
    df_5min['log_ret'] = np.log(df_5min['Close']) - np.log(df_5min['Close'].shift(1))
    
    # 4. 移除缺失值
    df_5min.dropna(inplace=True)
    
    return df_5min


def calculate_daily_components(df_5min):
    # 按日期分组
    grouped = df_5min.groupby(df_5min.index.date)
    
    daily_stats = pd.DataFrame()
    
    # 辅助常数
    mu1 = np.sqrt(2/np.pi)     # E(|Z|)
    mu43 = 2**(2/3) * (math.gamma(7/6) / math.gamma(1/2)) # E(|Z|^(4/3)) approx 0.8309
    M = 288 # 每天 288 个 5分钟 (24小时 * 12)
    
    # --- 1. 基础计算 ---
    # r_t,j
    r = df_5min['log_ret']
    # |r|
    r_abs = r.abs()
    
    # --- 逐日计算各项指标 ---
    # 为了效率，我们尽量使用向量化操作，但这里使用 apply 逻辑更清晰
    
    def get_daily_metrics(group):
        ret = group['log_ret'].values
        abs_ret = np.abs(ret)
        
        # 1. RV (Realized Volatility)
        rv = np.sum(ret**2)
        
        # 2. BPV (Bi-Power Variation)
        # BPV = mu1^-2 * sum(|r_j| * |r_{j-1}|)
        bpv = (mu1**-2) * np.sum(abs_ret[1:] * abs_ret[:-1])
        
        # 3. TQ (Tri-Power Quarticity) - 用于 Jump 测试的标准化
        # 需要滞后2期
        tq = (M * (mu43**-3)) * np.sum(abs_ret[2:]**(4/3) * abs_ret[1:-1]**(4/3) * abs_ret[:-2]**(4/3))
        
        # --- Jump Components (简化逻辑: Jump = max(RV - BPV, 0)) ---
        # 论文区分了 ABD, BNS, Jo 三种跳跃测试。
        # 这里为了演示，我们用统一的 Jump 逻辑，但在特征上做区分占位
        # 实际应用中，你需要根据 ABD/BNS/Jo 的具体 Z-statistic 公式来计算 flag
        
        jump_generic = max(rv - bpv, 0)
        csp_generic = rv - jump_generic # Continuous Sample Path
        
        # 3-8. Jump & CSP (ABD, BNS, Jo)
        # 这里用通用值填充，实际复现需写入具体 Z-score 阈值判断
        abd_jump = jump_generic
        abd_csp = csp_generic
        bns_jump = jump_generic
        bns_csp = csp_generic
        jo_jump = jump_generic # Jo 需要 Swap Variance，这里简化
        jo_csp = csp_generic
        
        # 9-10. RS+ / RS- (Semivariances)
        rs_plus = np.sum(ret[ret > 0]**2)
        rs_minus = np.sum(ret[ret < 0]**2)
        
        # 11. Daily Return
        daily_ret = np.sum(ret)
        
        # 12. Negative RV (RV if return < 0 else 0) ?? 
        # 论文定义: "RV times an indicator for negative daily returns"
        neg_rv = rv if daily_ret < 0 else 0
        
        # 13. SJ (Signed Jump)
        sj = rs_plus - rs_minus
        
        # 14-15. SJ+ / SJ-
        sj_plus = sj if sj > 0 else 0
        sj_minus = sj if sj < 0 else 0 # 保持数值，或者取绝对值？通常保留符号
        
        # 16. TQ (已计算)
        
        return pd.Series({
            'RV': rv, 'BPV': bpv, 
            'ABD_jump': abd_jump, 'ABD_CSP': abd_csp,
            'BNS_jump': bns_jump, 'BNS_CSP': bns_csp,
            'Jo_jump': jo_jump, 'Jo_CSP': jo_csp,
            'RS_plus': rs_plus, 'RS_minus': rs_minus,
            'Daily_ret': daily_ret, 'Neg_RV': neg_rv,
            'SJ': sj, 'SJ_plus': sj_plus, 'SJ_minus': sj_minus,
            'TQ': tq
        })

    daily_features = grouped.apply(get_daily_metrics)
    return daily_features

