# DPFL Based On HAKP: 基于分层自适应知识池的动态参与联邦学习

![License](https://img.shields.io/badge/license-MIT-blue.svg) ![Language](https://img.shields.io/badge/language-Python-3776AB.svg) ![Framework](https://img.shields.io/badge/framework-PyTorch-EE4C2C.svg) ![Platform](https://img.shields.io/badge/platform-Windows-0078D6.svg)

## 📖 项目简介

**DPFL Based On HAKP** (Dynamic Participation Federated Learning Based On Hierarchical Adaptive Knowledge Pool) 是一个面向动态参与环境的联邦学习框架。针对现有联邦学习中客户端动态参与导致的知识遗忘和模型性能波动问题，本项目在 **DPFL** 的基础上，创新性地提出了 **HAKP (分层自适应知识池)** 机制。

通过引入 **客户端价值评分 (Client Value Score, CVS)**，系统能够动态识别客户端的数据价值与参与稳定性，将其划分为 **核心层 (Core)**、**活跃层 (Active)** 和 **边缘层 (Edge)**，并对不同层级的客户端采用差异化的知识保留、衰减与聚合策略，从而在保证模型性能的同时显著降低存储开销并提升系统鲁棒性。

## 💡 核心创新点：分层自适应知识池 (HAKP)

<img width="1408" height="768" alt="fig3" src="https://github.com/user-attachments/assets/41a7a942-a0f0-48d0-a421-bfc6719e0fe6" />

### 1. 🏗️ 层级定义 (Layer Definitions)

基于 $CVS$，HAKP 动态将客户端分为三个功能层。

*   **核心层 (Core Layer):** 由 $CVS$ 最高（例如前 20%）的客户端组成。这些客户端持有稀缺数据并表现出稳定的参与度。
*   **活跃层 (Active Layer):** 标准贡献者的默认层级，具有平均的数据价值和参与度。
*   **边缘层 (Edge Layer):** 由 $CVS$ 最低（例如后 20%）的客户端组成。这些客户端通常数据冗余或可用性极不稳定。

### 2. 📊 客户端价值评分 (CVS)

<img width="1408" height="768" alt="fig4" src="https://github.com/user-attachments/assets/4408536d-1077-419d-b6ef-46de94f2f01e" />

为了量化客户端价值，我们定义了 CVS 指标，它由两部分组成：

**数据稀缺性评分 ($DSS$):**

客户端数据的价值与其所持有类别的全局频率成反比。拥有“稀有”类别（例如医学影像中的罕见病）的客户端将被赋予更高的分数。客户端 $i$ 的 $DSS$ 定义为：
$DSS_i = \sum_{j=1}^{K} \left( 1 - \frac{N_j}{N_{total}} \right) \times \frac{n_{i,j}}{N_i}$
其中 $K$ 是类别总数，$N_j$ 是全局类别 $j$ 的样本总数，$N_{total}$ 是所有客户端的样本总数，$n_{i,j}$ 是客户端 $i$ 上类别 $j$ 的样本数量，$N_i$ 是客户端 $i$ 上的样本总数。该公式奖励那些本地数据分布集中在全局稀缺类别上的客户端。

**参与稳定性评分 ($PSS$):**

为了确保知识的可靠性，我们偏好那些频繁且规律参与的客户端。$PSS$ 是基于过去 $T$ 轮的滑动窗口计算的：
$PSS_i = \frac{\mathcal{F}_i}{T} \times (1 - CV_{interval})$
其中 $\mathcal{F}_i$ 是客户端 $i$ 在过去 $T$ 轮中的参与频率（次数）。$CV_{interval}$ 是参与时间间隔的变异系数（标准差 / 均值）。较低的 $CV$ 表示参与模式更具可预测性和规律性。

**综合评分 $CVS$:**

最终评分是由超参数 $\alpha$ 和 $\beta$ 控制的加权和：
$CVS_i = \alpha \cdot DSS_i + \beta \cdot PSS_i$
通常，我们将 $\alpha = 0.7$ 和 $\beta = 0.3$，以便在考虑系统稳定性的同时优先考虑数据价值。

### 3. 🔄 带滞后的动态调整 (Dynamic Adjustment with Hysteresis)

设 $\theta_{high}$ 和 $\theta_{low}$ 分别对应当前 $CVS$ 分布的前 20% 和后 20% 分位数的动态阈值。客户端 $c_i$ 基于以下条件在层级间移动：

*   **升级至核心层:** ${CVS}_i > \theta_{\text{high}} + \delta$
*   **降级至边缘层:** ${CVS}_i < \theta_{\text{low}} - \delta$

其中 $\delta$ 是缓冲边际。此外，只有当条件在连续的观察轮次中满足时，才会执行迁移，以防止因频繁切换层级导致的系统不稳定。

### 4. ⚙️ 差异化知识管理 (Differential Knowledge Management)

<img width="1408" height="768" alt="fig5" src="https://github.com/user-attachments/assets/da36c7f4-daff-4b39-9cc1-6a15bc2bb5e8" />

我们对每一层采用不同的策略，以平衡性能和效率：

| 策略维度 | 核心层 (Core) | 活跃层 (Active) | 边缘层 (Edge) |
| :--- | :--- | :--- | :--- |
| **知识保留** | 完整模型 $\theta_i$ | 完整模型 $\theta_i$ | 原型 $p_i$ |
| **衰减因子 ($\lambda$)** | -0.001 (永久) | -0.1 (标准) | -0.5 (快速) |
| **层级权重 ($\gamma$)** | 1.5 (增强) | 1.0 (基准) | 0.5 (降低) |

客户端的修正聚合权重 $W_i^{HAKP}$ 变为：
$W_i^{HAKP} = \gamma_{\text{layer}} \times (aw_{i} \cdot \epsilon_i + dw_{i})$

这确保了“核心”知识几乎不衰减，并且对全局模型的贡献显著增加，即使这些客户端暂时不活跃，也能保留稀缺模式。

### 5. 🧩 知识原型 (Knowledge Prototypes)

<img width="1408" height="768" alt="fig6" src="https://github.com/user-attachments/assets/2aeec7c2-dcf8-4a1f-9d59-34d58de3bc75" />

**边缘层 (Edge Layer)** 的一个关键创新是引入知识原型以降低存储开销。对于低价值客户端，我们不再存储其完整的高维参数向量 $\theta_i$，而是将其知识压缩为一个轻量级的原型向量 $p_i$。

原型 $p_i$ 代表了客户端特征空间的质心。它是通过对客户端本地数据集 $D_i$ 在其训练好的本地模型 $f(\cdot; \theta_i)$ 上提取的特征表示求平均来计算的：
$p_i = \frac{1}{N_i} \sum_{x \in D_i} f(x; \theta_i)$
这个原型向量捕获了客户端数据分布的基本语义信息，而无需承担完整模型的存储成本。在生成式知识蒸馏过程中，这些原型作为生成器的正则化约束或调节输入，确保全局模型保留边缘客户端的基本信息，而无需为其分配大量存储资源。

### 6. 🚀 算法流程 (Algorithm Workflow)

<img width="1408" height="768" alt="fig7" src="https://github.com/user-attachments/assets/bbac9a93-928e-4a3d-93df-765984e900e3" />

HAKP 的整体训练过程如下：

1.  **初始化 (Initialization):** 在训练开始时，所有客户端暂时被分配到 **活跃层 (Active Layer)**。
2.  **指标更新 (Metric Update):** 每隔 $R$ 轮，服务器根据所有注册客户端最新的累计 $DSS$ 和参与历史重新计算 $CVS_i$。
3.  **层级重分配 (Layer Re-assignment):** 基于更新后的 $CVS$ 和滞后阈值 $\theta_{high} \pm \delta$ 与 $\theta_{low} \pm \delta$，对客户端进行升级或降级。
4.  **本地训练与上传 (Local Training & Upload):** 活跃客户端 $\mathcal{C}_t$ 执行本地训练。核心和活跃客户端上传完整的模型更新 $\theta_i$；边缘客户端计算并上传知识原型 $p_i$。
5.  **知识池维护 (Knowledge Pool Maintenance):** 服务器更新知识池。
    *   对于 **核心 (Core)** 条目：应用缓慢衰减 ($\lambda_{ia} \approx 0$)。
    *   对于 **边缘 (Edge)** 条目：更新原型并应用快速衰减。
6.  **自适应聚合 (Adaptive Aggregation):** 使用生成式蒸馏过程（如 KPFL 中定义）更新全局模型，但使用包含层级重要性因子 $\gamma_{layer}$ 的新权重 $W_i^{HAKP}$ 进行加权。

该流程确保计算资源智能地向高价值客户端倾斜，显著提高了资源效率，并增强了模型对抗稀缺模式灾难性遗忘的鲁棒性。

## 🛠️ 环境要求

*   Python 3.8+
*   PyTorch 1.8+
*   Torchvision
*   Numpy
*   WandB (可选，用于日志记录)

安装依赖：
```bash
pip install -r requirements.txt
```

## ⚡ 快速开始

### 运行示例

使用 `FedAvg` 算法并启用 `KPFL` (即 HAKP 增强版) 在 `Office-Caltech` 数据集上运行：

```bash
python main.py --dataset Office-Caltech --algorithm fedavg --kpfl --num_rounds 100 --num_clients 10 --gpu 0
```

### 关键参数说明

*   `--kpfl`: **启用 HAKP 知识池模块（核心开关）**。
*   `--algorithm`: 基础联邦学习算法 (如 `fedavg`, `fedprox`, `scaffold` 等)。
*   `--dataset`: 数据集名称 (`MNIST`, `Cifar10`, `Office-Caltech`)。
*   `--dynamic_type`: 客户端动态参与模式 (`static`, `incremental-arrival`, `incremental-departure`, `random`, `markov`)。
*   `--skew_type`: 数据异构类型 (`label`, `feature`, `quantity`)。

## 📂 项目结构

```
DPFL Based On HAKP/
├── configs/                # 超参数配置文件
├── dpmodels/               # 动态参与模式实现 (Arrival, Departure, Markov等)
├── models/                 # 神经网络模型定义 (ResNet, CNN, Generator)
├── strategies/             # 联邦学习策略实现
│   ├── feddpfl.py          # [核心] HAKP 逻辑实现
│   ├── fedavg.py           # FedAvg 基线
│   └── ...
├── utils/                  # 工具函数 (数据加载, 绘图等)
├── main.py                 # 程序入口
├── server.py               # 服务端逻辑 (包含 HAKP 管理)
├── client.py               # 客户端逻辑
└── requirements.txt        # 项目依赖
```

## 📝 引用与致谢

- 本仓库在 NYCU-PAIR-Labs 开源项目 DPFL 的基础上进行了扩展与重构，特此致谢：
  - DPFL (NYCU-PAIR-Labs): https://github.com/NYCU-PAIR-Labs/DPFL
- 本项目新增与改进内容包括但不限于：分层自适应知识池（HAKP）、客户端价值评分（CVS）、带滞后阈值的层级迁移、分层差异化知识管理、边缘层知识原型与无数据蒸馏等模块。

数据与资源致谢：
- 实验使用的 Office-Caltech10 等数据集仅用于学术研究，版权归原作者所有。

开源生态感谢（按字母序）：
- Colorama、Matplotlib、NNI、NumPy、Pandas、PyTorch、Scikit-learn、Seaborn、Torchvision、TQDM、WandB 等社区提供的工具与支持。

许可与声明：
- 本项目以 MIT License 开源发布。若存在引用遗漏或侵权问题，请通过 Issue 与我们联系更正。
