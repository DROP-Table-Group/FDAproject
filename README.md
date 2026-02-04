# FDAproject

#### 时间

+ 期中（3.25）之前做完：✔
  + 1月：数据收集
  + 2月：基准模型 重点模型
  + 3月：Webapp

#### 选题

##### 高频已实现波动率（Realized Volatility）的预测与比较：✔

波动率是衍生品定价的核心。加密货币的高频波动率具有长记忆性和聚集性，非常适合做时间序列分析。

* **问题描述 (Problem):** 利用高频数据（如*1分钟*或*Tick*数据）计算“已实现波动率(RV)”，并预测未来的波动率。这对期权定价和风险管理至关重要。
* **数据来源:** *BTC* 的高频交易价格数据。
* **模型设定 (Models):**
  * **基准模型 (Simple Model):** **GARCH(1,1)** 或 **EWMA**。
    * *符合要求:* 这是金融时间序列的标配，必须做平稳性检查。
  * **提议模型 (Proposed Model):** **HAR-RV (Heterogeneous Autoregressive model of Realized Volatility)**。
    * *理由:* HAR模型在微观结构研究中非常经典，它认为不同类型的交易者（日内、周线、月线）对波动率有不同的影响。相比GARCH，HAR在高频数据下表现通常更好。
* **Webapp交互点 (关键要求):**
  * **参数调整:** HAR模型中的日、周、月系数权重；采样频率（如用1分钟还是5分钟数据计算RV）。
  * **展示:** 对比GARCH和HAR-RV的预测曲线与真实波动率的拟合程度。
* **微观结构解释:** 讨论为何短期投机者（高频）和长期持有者对波动率贡献不同。

这是一个**回归问题**（预测下一个时间段的波动率数值）。

1. 判断依据 (Evidence)

* **样本外误差指标 (Out-of-Sample MSE/MAE/QLIKE):**
  * *核心:* 必须把数据分为训练集和测试集。展示模型在测试集上的 **MSE (均方误差)** 或 **MAE (平均绝对误差)**。
  * *QLIKE:* 这是一个专门针对波动率预测的损失函数，比MSE更具说服力（如果你能提到这一点，教授会觉得你很专业）。
* **Diebold-Mariano Test (DM Test):**
  * *杀手锏:* 这是统计学上用来比较两个预测模型是否有**显著差异**的标准方法。
  * *证据:* 如果 DM Test 的 p-value < 0.05，你就可以在报告里大胆地说：“我有95%的把握认为我的模型在统计上优于基准模型”，而不仅仅是“我的误差线看起来比它低”。
* **残差分析 (Residual Analysis) - 对应 Requirement 4:**
  * 检查模型预测后的残差（真实值 - 预测值）是否是**白噪声 (White Noise)**。如果是，说明模型提取了所有有效信息；如果残差还有规律（自相关性），说明模型没做好。

2. 理想结论 (Ideal Conclusion)

> “HAR-RV 模型在样本外预测中，MSE比简单的 GARCH(1,1) 低了 12%。DM Test 证实了这种差异的显著性。这解释了加密货币市场的**‘长记忆性 (Long Memory)’**特征——即过去一周甚至一月的波动率对今日波动率仍有显著影响，而 GARCH 模型过度关注短期冲击，无法捕捉这一特征。”

---

##### 针对Webapp开发的具体建议 (Python栈)

既然要求是 Webapp，且是金融分析，推荐使用 **Streamlit** 或 **Dash (Plotly)**。

1. **Streamlit (首选):** 极度适合数据科学项目。几行代码就能把 Python 的图表变成网页。完全不需要懂 HTML/CSS。
   * *优势:* 开发速度最快，完美支持 Pandas 和 Matplotlib/Plotly。
2. **Webapp 结构设计 (对应 Requirements):**
   * **Page 1: Introduction:** 项目介绍，数据概览（展示前几行数据，数据清洗过程）。
   * **Page 2: EDA & Stationarity Check:**
     * 放置通过/不通过平稳性检验（ADF test）的结果。
     * 展示自相关图（ACF/PACF）。
   * **Page 3: Model Configuration (核心):**
     * 左侧侧边栏放 Sliders（滑动条）控制参数（如学习率、窗口大小、ARIMA的p,d,q）。
     * 中间显示模型训练结果。
   * **Page 4: Comparison & Conclusion:**
     * 将 Simple Model 和 Proposed Model 的预测结果画在同一张图上对比。
     * 列出误差指标（MSE, RMSE, MAE）。

#### 分工

1. 数据收集 & 数据清洗 & 数据检验: MQ
2. 基准模型: LQR
3. 重点模型 & 调参: MYJ
4. 可视化 & Webapp: ZCK
