"""
高频金融数据分析应用
用于上传高频行情数据，生成收益率和波动率的描述性统计报告
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 页面配置
st.set_page_config(
    page_title="高频金融数据分析",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 应用标题
st.title("📈 高频金融数据分析平台")
st.markdown("""
    上传高频行情数据，自动计算收益率和波动率，生成描述性统计报告和可视化图表。
    支持CSV格式数据，需要包含时间戳和价格列。
""")

# 初始化session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'returns' not in st.session_state:
    st.session_state.returns = None
if 'volatility' not in st.session_state:
    st.session_state.volatility = None

def load_sample_data():
    """加载示例数据"""
    try:
        # 尝试读取示例数据文件
        sample_data = pd.read_csv('../data/btcusd_1-min_data.csv')
        return sample_data
    except:
        # 如果文件不存在，创建一个示例数据
        st.warning("示例数据文件不存在，正在创建示例数据...")
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
        st.error(f"数据中未找到价格列: {price_col}")
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

    stats = {
        '统计量': ['数量', '均值', '标准差', '最小值', '25%分位数',
                 '中位数', '75%分位数', '最大值', '偏度', '峰度', 'Jarque-Bera检验p值'],
        '值': []
    }

    stats['值'].append(len(data))
    stats['值'].append(f"{data.mean():.6f}")
    stats['值'].append(f"{data.std():.6f}")
    stats['值'].append(f"{data.min():.6f}")
    stats['值'].append(f"{data.quantile(0.25):.6f}")
    stats['值'].append(f"{data.quantile(0.5):.6f}")
    stats['值'].append(f"{data.quantile(0.75):.6f}")
    stats['值'].append(f"{data.max():.6f}")
    stats['值'].append(f"{data.skew():.6f}")
    stats['值'].append(f"{data.kurtosis():.6f}")

    # Jarque-Bera正态性检验
    from scipy import stats as scipy_stats
    jb_stat, jb_pvalue = scipy_stats.jarque_bera(data)
    stats['值'].append(f"{jb_pvalue:.6f}")

    return pd.DataFrame(stats)

def create_price_chart(data, price_col='Close'):
    """创建价格图表"""
    if data is None or price_col not in data.columns:
        return None

    fig = go.Figure()

    # 添加价格线
    fig.add_trace(go.Scatter(
        x=data.index if 'Timestamp' not in data.columns else data['Timestamp'],
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
            x=data.index if 'Timestamp' not in data.columns else data['Timestamp'],
            y=ma_20,
            mode='lines',
            name='20期移动平均',
            line=dict(color='orange', width=1, dash='dash')
        ))

        fig.add_trace(go.Scatter(
            x=data.index if 'Timestamp' not in data.columns else data['Timestamp'],
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
    from statsmodels.graphics.tsaplots import plot_acf
    import matplotlib.pyplot as plt
    from io import BytesIO

    # 创建ACF图
    fig_acf = plt.figure()
    plot_acf(returns, lags=40, ax=plt.gca())
    plt.close()

    # 由于Plotly不支持直接从matplotlib转换，我们手动计算ACF
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
    from scipy import stats as scipy_stats
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

def create_volatility_chart(volatility):
    """创建波动率图表"""
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

    # 波动率聚类现象（收益率绝对值与滞后收益率绝对值的关系）
    if len(volatility) > 1:
        abs_returns = np.abs(st.session_state.returns) if st.session_state.returns is not None else None
        if abs_returns is not None and len(abs_returns) > 100:
            lagged_abs = abs_returns.shift(1).dropna()
            current_abs = abs_returns.iloc[1:]

            fig.add_trace(
                go.Scatter(
                    x=lagged_abs,
                    y=current_abs,
                    mode='markers',
                    name='聚类现象',
                    marker=dict(
                        color=current_abs.index.astype(int),
                        colorscale='Viridis',
                        size=5,
                        showscale=True,
                        colorbar=dict(title="时间索引")
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

def create_summary_report(data, returns, volatility):
    """创建总结报告"""
    if data is None:
        return "请先上传数据"

    report = []
    report.append("# 金融数据分析报告")
    report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")

    # 数据概览
    report.append("## 1. 数据概览")
    report.append(f"- 数据总行数: {len(data):,}")
    report.append(f"- 时间范围: {data.index[0]} 到 {data.index[-1]}" if hasattr(data.index, '__len__') else "- 时间范围: 未指定")
    report.append(f"- 数据列: {', '.join(data.columns.tolist())}")
    report.append("")

    # 收益率分析
    if returns is not None:
        report.append("## 2. 收益率分析")
        report.append(f"- 收益率数量: {len(returns):,}")
        report.append(f"- 平均收益率: {returns.mean():.6f}")
        report.append(f"- 收益率标准差: {returns.std():.6f}")
        report.append(f"- 收益率偏度: {returns.skew():.6f}")
        report.append(f"- 收益率峰度: {returns.kurtosis():.6f}")
        report.append(f"- 最小收益率: {returns.min():.6f}")
        report.append(f"- 最大收益率: {returns.max():.6f}")
        report.append("")

        # 正态性检验
        from scipy import stats as scipy_stats
        jb_stat, jb_pvalue = scipy_stats.jarque_bera(returns)
        report.append(f"- Jarque-Bera检验p值: {jb_pvalue:.6f}")
        if jb_pvalue < 0.05:
            report.append("  → 收益率分布显著偏离正态分布（p < 0.05）")
        else:
            report.append("  → 收益率分布与正态分布无显著差异（p ≥ 0.05）")
        report.append("")

    # 波动率分析
    if volatility is not None:
        report.append("## 3. 波动率分析")
        report.append(f"- 波动率数量: {len(volatility):,}")
        report.append(f"- 平均波动率: {volatility.mean():.6f}")
        report.append(f"- 波动率标准差: {volatility.std():.6f}")
        report.append(f"- 最小波动率: {volatility.min():.6f}")
        report.append(f"- 最大波动率: {volatility.max():.6f}")
        report.append("")

    # 风险指标
    if returns is not None:
        report.append("## 4. 风险指标")
        # VaR (Value at Risk)
        var_95 = np.percentile(returns, 5)
        var_99 = np.percentile(returns, 1)
        report.append(f"- 95% VaR: {var_95:.6f}")
        report.append(f"- 99% VaR: {var_99:.6f}")

        # Expected Shortfall (CVaR)
        cvar_95 = returns[returns <= var_95].mean()
        cvar_99 = returns[returns <= var_99].mean()
        report.append(f"- 95% Expected Shortfall: {cvar_95:.6f}")
        report.append(f"- 99% Expected Shortfall: {cvar_99:.6f}")

        # Sharpe Ratio (假设无风险利率为0)
        if returns.std() > 0:
            sharpe_ratio = returns.mean() / returns.std()
            report.append(f"- Sharpe Ratio (无风险利率=0): {sharpe_ratio:.6f}")
        report.append("")

    # 结论
    report.append("## 5. 主要发现")
    if returns is not None:
        if returns.mean() > 0:
            report.append("- 平均收益率为正，表明资产有正向回报")
        else:
            report.append("- 平均收益率为负，表明资产有负向回报")

        if returns.skew() < 0:
            report.append("- 收益率分布左偏，极端负收益出现的概率较高")
        elif returns.skew() > 0:
            report.append("- 收益率分布右偏，极端正收益出现的概率较高")
        else:
            report.append("- 收益率分布基本对称")

        if returns.kurtosis() > 3:
            report.append("- 收益率分布具有尖峰厚尾特征，极端事件发生概率高于正态分布")
        elif returns.kurtosis() < 3:
            report.append("- 收益率分布具有低峰薄尾特征，极端事件发生概率低于正态分布")
        else:
            report.append("- 收益率分布的峰度与正态分布相似")

    return "\n".join(report)

# 侧边栏
with st.sidebar:
    st.header("📊 数据上传")

    data_source = st.radio(
        "选择数据来源",
        ["上传文件", "使用示例数据"],
        help="上传CSV文件或使用内置示例数据"
    )

    if data_source == "上传文件":
        uploaded_file = st.file_uploader(
            "选择CSV文件",
            type=['csv'],
            help="上传包含时间戳和价格列的CSV文件"
        )

        if uploaded_file is not None:
            try:
                data = pd.read_csv(uploaded_file)
                st.session_state.data = data
                st.success(f"成功加载数据，共 {len(data):,} 行")
            except Exception as e:
                st.error(f"文件读取失败: {e}")
    else:
        if st.button("加载示例数据"):
            data = load_sample_data()
            st.session_state.data = data
            st.success(f"成功加载示例数据，共 {len(data):,} 行")

    st.divider()

    if st.session_state.data is not None:
        st.header("⚙️ 分析设置")

        # 选择价格列
        price_cols = [col for col in st.session_state.data.columns
                     if col.lower() in ['close', 'price', 'last', 'settle']]
        if not price_cols:
            price_cols = st.session_state.data.columns.tolist()

        price_col = st.selectbox(
            "选择价格列",
            price_cols,
            index=0 if price_cols else 0
        )

        # 选择收益率类型
        return_type = st.radio(
            "收益率计算方式",
            ["对数收益率", "简单收益率"],
            help="对数收益率具有更好的统计性质，简单收益率更直观"
        )

        # 选择波动率窗口
        vol_window = st.selectbox(
            "波动率计算窗口",
            ["1min", "5min", "15min", "1H", "1D"],
            index=4,
            help="选择计算已实现波动率的时间窗口"
        )

        annualize_vol = st.checkbox(
            "年化波动率",
            value=True,
            help="将波动率转换为年化值以便比较"
        )

        if st.button("开始分析", type="primary"):
            with st.spinner("正在计算收益率和波动率..."):
                # 计算收益率
                log_returns = (return_type == "对数收益率")
                returns = calculate_returns(st.session_state.data, price_col, log_returns)
                st.session_state.returns = returns

                # 计算波动率
                volatility = calculate_realized_volatility(
                    returns, vol_window, annualize_vol
                )
                st.session_state.volatility = volatility

                st.success("分析完成！")

# 主内容区域
if st.session_state.data is not None:
    # 显示数据预览
    st.header("📋 数据预览")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("数据行数", f"{len(st.session_state.data):,}")
    with col2:
        st.metric("数据列数", len(st.session_state.data.columns))
    with col3:
        numeric_cols = st.session_state.data.select_dtypes(include=[np.number]).columns
        st.metric("数值列数", len(numeric_cols))

    # 显示数据前几行
    with st.expander("查看数据前10行"):
        st.dataframe(st.session_state.data.head(10), use_container_width=True)

    # 显示数据统计信息
    with st.expander("查看数据描述性统计"):
        st.dataframe(st.session_state.data.describe(), use_container_width=True)

    # 价格图表
    st.header("📊 价格走势")
    price_chart = create_price_chart(st.session_state.data, price_col)
    if price_chart:
        st.plotly_chart(price_chart, use_container_width=True)

    # 收益率分析
    if st.session_state.returns is not None:
        st.header("📈 收益率分析")

        # 收益率统计
        returns_stats = get_descriptive_stats(st.session_state.returns, "收益率")
        if returns_stats is not None:
            st.dataframe(returns_stats, use_container_width=True, hide_index=True)

        # 收益率图表
        returns_chart = create_returns_chart(st.session_state.returns)
        if returns_chart:
            st.plotly_chart(returns_chart, use_container_width=True)

    # 波动率分析
    if st.session_state.volatility is not None:
        st.header("🌊 波动率分析")

        # 波动率统计
        volatility_stats = get_descriptive_stats(st.session_state.volatility, "波动率")
        if volatility_stats is not None:
            st.dataframe(volatility_stats, use_container_width=True, hide_index=True)

        # 波动率图表
        volatility_chart = create_volatility_chart(st.session_state.volatility)
        if volatility_chart:
            st.plotly_chart(volatility_chart, use_container_width=True)

    # 总结报告
    if st.session_state.returns is not None or st.session_state.volatility is not None:
        st.header("📄 分析报告")

        report = create_summary_report(
            st.session_state.data,
            st.session_state.returns,
            st.session_state.volatility
        )

        st.markdown(report)

        # 下载报告
        report_text = report
        st.download_button(
            label="下载报告 (TXT)",
            data=report_text,
            file_name=f"financial_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )
else:
    # 显示欢迎信息
    st.info("👈 请从左侧边栏上传数据或加载示例数据开始分析")

    # 显示功能介绍
    with st.expander("查看应用功能说明"):
        st.markdown("""
        ## 功能特性

        ### 1. 数据上传与处理
        - 支持上传CSV格式的高频金融数据
        - 自动识别价格列和时间戳
        - 提供示例数据供测试使用

        ### 2. 收益率计算
        - 支持对数收益率和简单收益率计算
        - 自动处理缺失值和异常值
        - 提供完整的描述性统计

        ### 3. 波动率分析
        - 计算已实现波动率（Realized Volatility）
        - 支持不同时间窗口（1分钟到1天）
        - 可选年化处理

        ### 4. 可视化图表
        - 价格走势图（含移动平均线）
        - 收益率时间序列和分布图
        - 波动率时间序列和分布图
        - 自相关函数（ACF）分析
        - QQ图正态性检验

        ### 5. 风险指标
        - Value at Risk (VaR)
        - Expected Shortfall (CVaR)
        - Sharpe Ratio

        ### 6. 报告生成
        - 自动生成详细的分析报告
        - 包含主要发现和结论
        - 支持报告下载

        ## 数据格式要求

        上传的CSV文件应包含以下列（列名不区分大小写）：
        - `Timestamp` 或 `Date`: 时间戳列
        - `Close` 或 `Price`: 收盘价或价格列
        - `Open`, `High`, `Low`, `Volume` (可选): 开盘价、最高价、最低价、成交量

        如果文件不包含标准列名，可以在分析设置中选择对应的列。
        """)

# 页脚
st.divider()
st.caption("高频金融数据分析应用 | 使用Streamlit, Pandas和Plotly构建")
