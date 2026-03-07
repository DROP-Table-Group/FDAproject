
"""
高频金融数据分析报告生成器 (HTML版)
基于 webapp/app.py 改编，用于生成静态 HTML 报告
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import warnings
import os

warnings.filterwarnings('ignore')

# ----------------- 核心功能函数 (改编自 app.py) -----------------

def load_sample_data():
    """加载示例数据"""
    try:
        # 尝试读取示例数据文件 (相对路径调整)
        data_path = os.path.join(os.path.dirname(__file__), '../data/btcusd_1-min_data.csv')
        sample_data = pd.read_csv(data_path)
        if len(sample_data) == 0:
            raise ValueError("示例数据文件为空")
        elif len(sample_data) > 100000:
            print(f"示例数据过大 ({len(sample_data)} 行)，仅加载后100000行")
            sample_data = sample_data.tail(100000)
        print(f"成功加载数据: {data_path}")
        return sample_data
    except Exception as e:
        print(f"无法加载文件: {e}")
        print("正在创建随机示例数据...")
        # 创建示例数据
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='1min')
        n_samples = len(dates)
        np.random.seed(42)

        # 生成随机价格数据
        prices = 100 + np.cumsum(np.random.randn(n_samples) * 0.01)
        prices = np.maximum(prices, 0.1)  # 确保价格为正

        sample_data = pd.DataFrame({
            'Timestamp': dates,
            'Open': prices,
            'High': prices + np.random.rand(n_samples) * 0.5,
            'Low': prices - np.random.rand(n_samples) * 0.5,
            'Close': prices,
            'Volume': np.random.rand(n_samples) * 1000
        })
        return sample_data

def calculate_returns(data, price_col='Close', log_returns=True):
    """计算收益率"""
    if price_col not in data.columns:
        print(f"数据中未找到价格列: {price_col}")
        return None

    prices = data[price_col].dropna()
    if log_returns:
        returns = np.log(prices / prices.shift(1))
    else:
        returns = prices.pct_change()

    returns = returns.dropna()
    return returns

def calculate_realized_volatility(returns, window='1D', annualize=True):
    """计算已实现波动率"""
    if returns is None or len(returns) == 0:
        return None

    # 计算滚动波动率
    if window == '1min':
        window_size = 1
    elif window == '5min':
        window_size = 5
    elif window == '15min':
        window_size = 15
    elif window == '1H':
        window_size = 60
    elif window == '1D':
        window_size = 1440  # 假设1天有1440分钟
    else:
        window_size = 1440

    # 计算滚动标准差
    volatility = returns.rolling(window=window_size).std()

    # 年化处理
    if annualize:
        if window == '1min':
            scaling_factor = np.sqrt(1440 * 365)  # 分钟数据年化
        elif window == '5min':
            scaling_factor = np.sqrt(288 * 365)   # 5分钟数据年化
        elif window == '15min':
            scaling_factor = np.sqrt(96 * 365)    # 15分钟数据年化
        elif window == '1H':
            scaling_factor = np.sqrt(24 * 365)    # 小时数据年化
        else:
            scaling_factor = np.sqrt(365)         # 日数据年化

        volatility = volatility * scaling_factor

    volatility = volatility.dropna()
    return volatility

def get_descriptive_stats(data, name="数据"):
    """获取描述性统计"""
    if data is None or len(data) == 0:
        return None

    stats_dict = {
        '统计量': ['数量', '均值', '标准差', '最小值', '25%分位数',
                 '中位数', '75%分位数', '最大值', '偏度', '峰度', 'Jarque-Bera检验p值'],
        '值': []
    }

    stats_dict['值'].append(len(data))
    stats_dict['值'].append(f"{data.mean():.6f}")
    stats_dict['值'].append(f"{data.std():.6f}")
    stats_dict['值'].append(f"{data.min():.6f}")
    stats_dict['值'].append(f"{data.quantile(0.25):.6f}")
    stats_dict['值'].append(f"{data.quantile(0.5):.6f}")
    stats_dict['值'].append(f"{data.quantile(0.75):.6f}")
    stats_dict['值'].append(f"{data.max():.6f}")
    stats_dict['值'].append(f"{data.skew():.6f}")
    stats_dict['值'].append(f"{data.kurtosis():.6f}")

    # Jarque-Bera正态性检验
    from scipy import stats as scipy_stats
    jb_stat, jb_pvalue = scipy_stats.jarque_bera(data)
    stats_dict['值'].append(f"{jb_pvalue:.6f}")

    return pd.DataFrame(stats_dict)

def create_price_chart(data, price_col='Close'):
    """创建价格图表"""
    if data is None or price_col not in data.columns:
        return None

    fig = go.Figure()

    # 添加价格线
    x_data = data.index if 'Timestamp' not in data.columns else data['Timestamp']
    fig.add_trace(go.Scatter(
        x=x_data,
        y=data[price_col],
        mode='lines',
        name='价格',
        line=dict(color='blue', width=1)
    ))

    # 添加移动平均线
    if len(data) > 50:
        ma_20 = data[price_col].rolling(window=20).mean()
        ma_50 = data[price_col].rolling(window=50).mean()

        fig.add_trace(go.Scatter(
            x=x_data,
            y=ma_20,
            mode='lines',
            name='20期移动平均',
            line=dict(color='orange', width=1, dash='dash')
        ))

        fig.add_trace(go.Scatter(
            x=x_data,
            y=ma_50,
            mode='lines',
            name='50期移动平均',
            line=dict(color='red', width=1, dash='dash')
        ))

    fig.update_layout(
        title='价格走势图',
        xaxis_title='时间',
        yaxis_title='价格',
        hovermode='x unified',
        height=400
    )

    return fig

def create_returns_chart(returns):
    """创建收益率图表"""
    if returns is None or len(returns) == 0:
        return None

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('收益率时间序列', '收益率分布直方图',
                       '收益率自相关函数', '收益率QQ图'),
        vertical_spacing=0.15,
        horizontal_spacing=0.15
    )

    # 收益率时间序列
    fig.add_trace(
        go.Scatter(
            x=returns.index,
            y=returns.values,
            mode='lines',
            name='收益率',
            line=dict(color='green', width=1)
        ),
        row=1, col=1
    )

    # 收益率分布直方图
    fig.add_trace(
        go.Histogram(
            x=returns.values,
            nbinsx=50,
            name='分布',
            marker_color='lightblue',
            opacity=0.7
        ),
        row=1, col=2
    )

    # 添加正态分布曲线
    from scipy import stats as scipy_stats
    x_norm = np.linspace(returns.min(), returns.max(), 100)
    y_norm = scipy_stats.norm.pdf(x_norm, returns.mean(), returns.std())
    y_norm = y_norm * len(returns) * (returns.max() - returns.min()) / 50

    fig.add_trace(
        go.Scatter(
            x=x_norm,
            y=y_norm,
            mode='lines',
            name='正态分布',
            line=dict(color='red', width=2)
        ),
        row=1, col=2
    )

    # 自相关函数
    from statsmodels.tsa.stattools import acf
    acf_values = acf(returns, nlags=40)
    lags = np.arange(len(acf_values))

    fig.add_trace(
        go.Bar(
            x=lags,
            y=acf_values,
            name='ACF',
            marker_color='purple'
        ),
        row=2, col=1
    )

    # 添加置信区间线
    conf_int = 1.96 / np.sqrt(len(returns))
    fig.add_trace(
        go.Scatter(
            x=[lags[0], lags[-1]],
            y=[conf_int, conf_int],
            mode='lines',
            name='95%置信区间',
            line=dict(color='red', width=1, dash='dash')
        ),
        row=2, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=[lags[0], lags[-1]],
            y=[-conf_int, -conf_int],
            mode='lines',
            showlegend=False,
            line=dict(color='red', width=1, dash='dash')
        ),
        row=2, col=1
    )

    # QQ图
    qq_data = scipy_stats.probplot(returns, dist="norm")
    x = qq_data[0][0]
    y = qq_data[0][1]

    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode='markers',
            name='QQ图',
            marker=dict(color='blue', size=5)
        ),
        row=2, col=2
    )

    # 添加参考线
    fig.add_trace(
        go.Scatter(
            x=[x.min(), x.max()],
            y=[x.min(), x.max()],
            mode='lines',
            name='参考线',
            line=dict(color='red', width=1, dash='dash')
        ),
        row=2, col=2
    )

    fig.update_layout(
        height=600,
        showlegend=True,
        title_text="收益率分析"
    )

    fig.update_xaxes(title_text="时间", row=1, col=1)
    fig.update_yaxes(title_text="收益率", row=1, col=1)
    fig.update_xaxes(title_text="收益率", row=1, col=2)
    fig.update_yaxes(title_text="频数", row=1, col=2)
    fig.update_xaxes(title_text="滞后阶数", row=2, col=1)
    fig.update_yaxes(title_text="自相关系数", row=2, col=1)
    fig.update_xaxes(title_text="理论分位数", row=2, col=2)
    fig.update_yaxes(title_text="样本分位数", row=2, col=2)

    return fig

def create_volatility_chart(volatility, returns_data):
    """创建波动率图表 (修改：显式传入 returns 数据)"""
    if volatility is None or len(volatility) == 0:
        return None

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('波动率时间序列', '波动率分布直方图',
                       '波动率自相关函数', '波动率聚类现象'),
        vertical_spacing=0.15,
        horizontal_spacing=0.15
    )

    # 波动率时间序列
    fig.add_trace(
        go.Scatter(
            x=volatility.index,
            y=volatility.values,
            mode='lines',
            name='波动率',
            line=dict(color='red', width=1)
        ),
        row=1, col=1
    )

    # 波动率分布直方图
    fig.add_trace(
        go.Histogram(
            x=volatility.values,
            nbinsx=50,
            name='分布',
            marker_color='lightcoral',
            opacity=0.7
        ),
        row=1, col=2
    )

    # 自相关函数
    from statsmodels.tsa.stattools import acf
    acf_values = acf(volatility, nlags=40)
    lags = np.arange(len(acf_values))

    fig.add_trace(
        go.Bar(
            x=lags,
            y=acf_values,
            name='ACF',
            marker_color='darkorange'
        ),
        row=2, col=1
    )

    # 添加置信区间线
    conf_int = 1.96 / np.sqrt(len(volatility))
    fig.add_trace(
        go.Scatter(
            x=[lags[0], lags[-1]],
            y=[conf_int, conf_int],
            mode='lines',
            name='95%置信区间',
            line=dict(color='red', width=1, dash='dash')
        ),
        row=2, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=[lags[0], lags[-1]],
            y=[-conf_int, -conf_int],
            mode='lines',
            showlegend=False,
            line=dict(color='red', width=1, dash='dash')
        ),
        row=2, col=1
    )

    # 波动率聚类现象
    if len(volatility) > 1:
        abs_returns = np.abs(returns_data) if returns_data is not None else None
        if abs_returns is not None and len(abs_returns) > 100:
            # 确保对齐
            lagged_abs = abs_returns.shift(1).dropna()
            current_abs = abs_returns.loc[lagged_abs.index]

            fig.add_trace(
                go.Scatter(
                    x=lagged_abs,
                    y=current_abs,
                    mode='markers',
                    name='聚类现象',
                    marker=dict(
                        color=list(range(len(current_abs))), # 简单的数字作为颜色
                        colorscale='Viridis',
                        size=5,
                        showscale=True,
                        colorbar=dict(title="时间顺序")
                    )
                ),
                row=2, col=2
            )

            # 添加趋势线
            from scipy import stats as scipy_stats
            slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(
                lagged_abs, current_abs
            )

            x_line = np.array([lagged_abs.min(), lagged_abs.max()])
            y_line = intercept + slope * x_line

            fig.add_trace(
                go.Scatter(
                    x=x_line,
                    y=y_line,
                    mode='lines',
                    name=f'趋势线 (R²={r_value**2:.3f})',
                    line=dict(color='red', width=2)
                ),
                row=2, col=2
            )

    fig.update_layout(
        height=600,
        showlegend=True,
        title_text="波动率分析"
    )

    fig.update_xaxes(title_text="时间", row=1, col=1)
    fig.update_yaxes(title_text="波动率", row=1, col=1)
    fig.update_xaxes(title_text="波动率", row=1, col=2)
    fig.update_yaxes(title_text="频数", row=1, col=2)
    fig.update_xaxes(title_text="滞后阶数", row=2, col=1)
    fig.update_yaxes(title_text="自相关系数", row=2, col=1)
    fig.update_xaxes(title_text="滞后收益率绝对值", row=2, col=2)
    fig.update_yaxes(title_text="当前收益率绝对值", row=2, col=2)

    return fig

# ----------------- HTML 生成逻辑 -----------------

def generate_html_report():
    print("开始生成 HTML 报告...")
    
    # 1. 加载数据
    data = load_sample_data()
    
    # 确保时间索引
    if 'Timestamp' in data.columns and not isinstance(data.index, pd.DatetimeIndex):
        data['Timestamp'] = pd.to_datetime(data['Timestamp'])
        data.set_index('Timestamp', inplace=True)
    
    price_col = 'Close'

    # 2. 计算指标
    returns = calculate_returns(data, price_col=price_col)
    volatility = calculate_realized_volatility(returns)

    # 3. 统计信息
    price_stats = get_descriptive_stats(data[price_col], "价格")
    returns_stats = get_descriptive_stats(returns, "收益率")
    vol_stats = get_descriptive_stats(volatility, "波动率")

    # 4. 生成图表
    fig_price = create_price_chart(data, price_col)
    fig_returns = create_returns_chart(returns)
    fig_volatility = create_volatility_chart(volatility, returns)

    # 5. 构建 HTML 内容
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>高频金融数据分析报告</title>
        <meta charset="utf-8">
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
            .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
            h1, h2 {{ color: #333; }}
            .section {{ margin-bottom: 40px; }}
            table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .stats-container {{ display: flex; gap: 20px; flex-wrap: wrap; }}
            .stats-box {{ flex: 1; min-width: 300px; }}
        </style>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    </head>
    <body>
        <div class="container">
            <h1>📈 高频金融数据分析报告</h1>
            <p>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <div class="section">
                <h2>1. 数据概览</h2>
                <p>数据源: 示例数据 (Data)</p>
                <div class="stats-container">
                    <div class="stats-box">
                        <h3>价格统计</h3>
                        {price_stats.to_html(index=False, classes='table')}
                    </div>
                </div>
            </div>

            <div class="section">
                <h2>2. 价格走势</h2>
                {fig_price.to_html(full_html=False, include_plotlyjs='cdn')}
            </div>
            
            <div class="section">
                <h2>3. 收益率分析</h2>
                <div class="stats-box">
                    <h3>收益率统计</h3>
                    {returns_stats.to_html(index=False, classes='table')}
                </div>
                {fig_returns.to_html(full_html=False, include_plotlyjs=False)}
            </div>

            <div class="section">
                <h2>4. 波动率分析</h2>
                <div class="stats-box">
                    <h3>波动率统计</h3>
                    {vol_stats.to_html(index=False, classes='table')}
                </div>
                {fig_volatility.to_html(full_html=False, include_plotlyjs=False)}
            </div>
        </div>
    </body>
    </html>
    """

    output_file = os.path.join(os.path.dirname(__file__), 'finance_report.html')
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"报告已生成: {output_file}")

if __name__ == "__main__":
    generate_html_report()
