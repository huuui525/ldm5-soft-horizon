# 五通道地震数据 Autoencoder 软层位约束改进方案 — 原理与代码详解

---

## 一、改进方案整体概述

本方案是在原有 LDM5 第一阶段 `AutoencoderKL` 训练流程上增加**软层位约束 (Soft Horizon Constraint)**，目标是提升五通道地震属性图的重构结构质量，尤其是增强反射界面、层位边界和跨通道同步突变位置的保真度。

原始 AE 训练主要依赖：

1. **像素重建损失**：保证重构值接近输入。
2. **梯度损失**：鼓励边缘和突变位置被保留。
3. **可选通道相关损失**：匹配五通道之间的整体相关矩阵。
4. **可选频率损失**：匹配频谱幅值。

改进后的 AE 训练在此基础上新增：

1. **软层位图损失 (`horizon_loss`)**：从五通道数据中自动提取层位响应图，约束重构结果保留层位结构。
2. **层位同步损失 (`layer_sync_loss`)**：约束每个通道的垂向突变位置与真实样本一致，提升五通道层位同步性。

整体训练流程保持不变：

```text
真实五通道数据 [B, 5, 224, 224]
        ↓
AutoencoderKL 编码
        ↓
潜在变量 [B, 8, 56, 56]
        ↓
AutoencoderKL 解码
        ↓
重构数据 [B, 5, 224, 224]
        ↓
L1 + Gradient + ChannelCorr + SoftHorizon + LayerSync + KL
```

该方案不改变 AE 网络结构，只改变训练目标函数。因此它不会增加推理阶段的额外输入，也不会破坏后续 LDM 使用 AE latent 的方式。

---

## 二、层位约束的地质含义

### 2.1 层位在五通道数据中的表现

在地震/地质属性剖面中，层位通常表现为沿横向较连续、沿深度方向发生突变的界面。对于当前五通道数据：

```text
dn, gas, gr, vp, vs
```

同一地质界面往往会在多个属性通道中同时出现响应，例如：

- `vp` 在某深度位置发生速度跳变；
- `vs` 在相近位置出现同步变化；
- `dn`、`gr` 或 `gas` 在同一界面附近出现属性变化。

因此，层位不是某一个通道独有的边缘，而是五个通道共同描述同一地质体时产生的**跨通道同步结构**。

### 2.2 为什么普通重构损失不够

普通 L1 重构损失更关注逐像素数值误差。如果重构图整体数值接近真实图，即使层位边界略微模糊、错位或弱化，L1 仍可能不高。

对于下游任务来说，这种问题会导致：

- 反射界面不清晰；
- 薄层结构被抹平；
- 五通道之间的突变位置不同步；
- 可提取的有效地质信息减少。

因此，需要显式把“层位边界是否保留”和“通道之间的层位是否同步”纳入损失函数。

---

## 三、软层位图提取模块 (`horizon.py`)

### 3.1 归一化函数

```python
# horizon.py:11-14
def normalize_map(x, eps=1e-6):
    mean = x.mean(dim=(-2, -1), keepdim=True)
    std = x.std(dim=(-2, -1), keepdim=True, unbiased=False)
    return (x - mean) / (std + eps)
```

该函数对每个样本的二维图进行标准化，使不同样本、不同强度范围的层位响应可以在统一尺度下比较。

### 3.2 软层位图 `soft_horizon_map`

```python
# horizon.py:17-29
def soft_horizon_map(x):
    vertical = (x[..., 1:, :] - x[..., :-1, :]).abs()
    vertical = F.pad(vertical, (0, 0, 1, 0))
    horizon = vertical.mean(dim=1, keepdim=True)
    return normalize_map(horizon)
```

输入输出形状：

| 张量 | 形状 | 含义 |
|------|------|------|
| `x` | `[B, 5, H, W]` | 五通道地震属性图 |
| `vertical` | `[B, 5, H, W]` | 每个通道的深度方向突变强度 |
| `horizon` | `[B, 1, H, W]` | 跨通道平均后的软层位响应图 |

核心思想：

```text
层位响应 = 五通道深度方向梯度绝对值的平均
```

数学形式可写为：

```text
E_c(i, j) = |X_c(i, j) - X_c(i-1, j)|

H(i, j) = mean_c E_c(i, j)
```

其中：

- `c` 表示五个物理属性通道；
- `i` 表示深度方向；
- `j` 表示横向位置；
- `H` 是软层位图。

### 3.3 软层位图与硬标签的区别

本方案用于 AE 训练时使用的是**软层位图**，不是二值分割标签。

软层位图的优点：

- 不需要人工标注；
- 保留层位强弱信息；
- 对不确定边界更平滑；
- 可微，可直接参与损失计算；
- 不会因为阈值选择过强而丢失弱层位。

`horizon.py` 中也保留了 `horizon_label_map` 和 `horizon_condition`，它们主要用于条件扩散实验；AE 软层位约束使用的是 `soft_horizon_map`。

---

## 四、损失函数改进 (`losses.py`)

### 4.1 原始损失组成

原始 AE 损失在 `losses.py` 中定义，包括：

```python
reconstruction_loss(pred, target)
gradient_loss(pred, target)
channel_correlation_loss(pred, target)
frequency_loss(pred, target)
```

训练脚本中还额外加入 KL 正则：

```python
total = loss + 1e-6 * kl
```

KL 权重很小，因此当前 AE 更接近一个带轻量 VAE 正则的重构自编码器。

### 4.2 深度方向边缘强度

```python
# losses.py
def vertical_edge_magnitude(x):
    edge = (x[..., 1:, :] - x[..., :-1, :]).abs()
    return F.pad(edge, (0, 0, 1, 0))
```

该函数保留每个通道各自的垂向突变强度：

```text
[B, 5, H, W] → [B, 5, H, W]
```

它和 `soft_horizon_map` 的区别是：

- `vertical_edge_magnitude`：保留五个通道各自的边界；
- `soft_horizon_map`：对五个通道求平均，得到一个共享层位图。

### 4.3 软层位图损失 `horizon_loss`

```python
def horizon_loss(pred, target):
    pred_map = soft_horizon_map(pred)
    target_map = soft_horizon_map(target)
    loss = F.l1_loss(pred_map, target_map)
    for scale in (2, 4):
        loss = loss + F.l1_loss(
            F.avg_pool2d(pred_map, kernel_size=scale, stride=scale),
            F.avg_pool2d(target_map, kernel_size=scale, stride=scale),
        )
    return loss / 3.0
```

该损失包含三个尺度：

| 尺度 | 操作 | 关注内容 |
|------|------|----------|
| 原始尺度 | `H x W` | 细层位、薄层、局部边界 |
| 1/2 尺度 | `avg_pool2d(scale=2)` | 中尺度连续层位 |
| 1/4 尺度 | `avg_pool2d(scale=4)` | 大尺度构造趋势 |

数学形式：

```text
L_horizon =
1/3 * (
  |H(pred) - H(target)|_1
  + |P2(H(pred)) - P2(H(target))|_1
  + |P4(H(pred)) - P4(H(target))|_1
)
```

其中 `P2` 和 `P4` 表示平均池化下采样。

物理意义：

- 让重构图的层位响应位置接近真实图；
- 减少界面模糊；
- 保留薄层和大构造两类信息；
- 避免只优化像素值而忽略地质结构。

### 4.4 层位同步损失 `layer_sync_loss`

```python
def layer_sync_loss(pred, target):
    pred_edges = normalize_map(vertical_edge_magnitude(pred))
    target_edges = normalize_map(vertical_edge_magnitude(target))
    return F.l1_loss(pred_edges, target_edges)
```

该损失直接比较五个通道各自的垂向边缘图：

```text
[B, 5, H, W] vs [B, 5, H, W]
```

与 `horizon_loss` 的区别：

| 损失 | 比较对象 | 作用 |
|------|----------|------|
| `horizon_loss` | 跨通道平均后的共享层位图 | 保证整体层位结构 |
| `layer_sync_loss` | 每个通道自己的边缘图 | 保证各通道层位同步与边缘一致 |

物理意义：

五个通道不是独立图像，而是同一地质体的不同属性投影。`layer_sync_loss` 强制每个通道在真实存在突变的位置也尽量保留突变，从而减少“某个通道边界清楚、另一个通道边界漂移”的问题。

### 4.5 总损失函数

改进后的 `autoencoder_loss` 接收两个新增权重：

```python
lambda_horizon: float = 0.0
lambda_layer_sync: float = 0.0
```

最终损失为：

```text
L_AE =
L_recon
+ λ_grad * L_grad
+ λ_corr * L_channel_corr
+ λ_freq * L_freq
+ λ_horizon * L_horizon
+ λ_sync * L_layer_sync
+ 1e-6 * KL
```

当前默认配置为：

```python
lambda_gradient = 0.1
lambda_channel_corr = 0.05
lambda_frequency = 0.0
lambda_horizon = 0.2
lambda_layer_sync = 0.05
```

---

## 五、配置修改 (`config.py`)

### 5.1 新增训练权重

```python
@dataclass
class TrainConfig:
    lambda_gradient: float = 0.1
    lambda_channel_corr: float = 0.05
    lambda_frequency: float = 0.0
    lambda_horizon: float = 0.2
    lambda_layer_sync: float = 0.05
```

各权重含义：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `lambda_gradient` | `0.1` | 保留普通水平/垂向边缘 |
| `lambda_channel_corr` | `0.05` | 匹配五通道整体相关性 |
| `lambda_frequency` | `0.0` | 当前未启用频率约束 |
| `lambda_horizon` | `0.2` | 强化共享软层位图 |
| `lambda_layer_sync` | `0.05` | 强化跨通道层位同步 |

### 5.2 权重选择原则

`lambda_horizon` 设置为较高值，是因为本次主要问题是重构图中可提取结构信息不足，需要直接强化层位边界。

`lambda_layer_sync` 设置较小，是为了避免过度强调边缘，使重构图出现不自然的锐化或噪声增强。

`lambda_channel_corr` 从 `0.0` 调整为 `0.05`，用于补充五通道联合分布约束。

---

## 六、训练脚本修改 (`train_autoencoder.py`)

### 6.1 命令行参数

训练脚本新增以下参数：

```python
--lambda_gradient
--lambda_channel_corr
--lambda_frequency
--lambda_horizon
--lambda_layer_sync
```

因此可以在不改代码的情况下调整损失权重：

```powershell
python -m ldm5.train_autoencoder `
  --batch_size 4 `
  --num_epochs 50 `
  --lambda_horizon 0.2 `
  --lambda_layer_sync 0.05
```

### 6.2 训练配置写入

脚本会把最终解析后的配置写入：

```text
<output_dir>/autoencoder/config.json
```

这样可以追踪每次训练使用的软层位损失权重。

### 6.3 日志字段

`TrainLogger` 的字段扩展为：

```python
["total", "recon", "grad", "chcorr", "freq", "horizon", "sync", "kl", "psnr"]
```

训练日志中可以直接观察：

- `horizon` 是否下降；
- `sync` 是否下降；
- `recon` 是否被结构损失牺牲过多；
- `psnr` 是否稳定提升。

### 6.4 训练循环中的损失调用

```python
loss, parts = autoencoder_loss(
    recon,
    batch,
    lambda_gradient=cfg.train.lambda_gradient,
    lambda_channel_corr=cfg.train.lambda_channel_corr,
    lambda_frequency=cfg.train.lambda_frequency,
    lambda_horizon=cfg.train.lambda_horizon,
    lambda_layer_sync=cfg.train.lambda_layer_sync,
)
```

该调用仍然发生在 AE 的重构训练阶段，不影响 LDM 第二阶段训练脚本。

---

## 七、评估指标修改 (`evaluate.py`)

### 7.1 新增结构指标

评估脚本新增：

```python
horizon_l1_mean
layer_sync_l1_mean
```

含义：

| 指标 | 越小越好 | 说明 |
|------|----------|------|
| `horizon_l1_mean` | 是 | 重构图与真实图的共享软层位图差异 |
| `layer_sync_l1_mean` | 是 | 五通道各自边缘同步结构差异 |

### 7.2 与传统指标的关系

传统指标：

- `L1`
- `MSE`
- `PSNR`
- `SSIM`

更偏向图像重构质量。

新增层位指标更偏向地质结构质量。

因此，评估时应同时观察：

```text
L1 / PSNR / SSIM + horizon L1 / sync L1
```

如果 PSNR 小幅变化，但 `horizon L1` 明显降低，说明重构图的层位结构更稳定，可能更适合后续地质解释或生成模型训练。

---

## 八、可视化修改 (`visualize.py`)

### 8.1 新增层位图对比

可视化脚本新增输出：

```text
ae_horizon_reconstruction.png
```

该图包含三类行：

```text
真实样本 soft horizon
重构样本 soft horizon
二者绝对误差 |err|
```

用于直观看出：

- 层位边界是否被重构；
- 重构层位是否发生上下偏移；
- 哪些区域层位误差最大；
- 五通道平均后的共享边界是否清晰。

### 8.2 原有可视化仍保留

原有输出仍然存在：

```text
real_samples.png
ae_reconstruction.png
ldm_samples.png
visualize_arrays.npz
```

如果只想看 AE，不运行 LDM 采样，可以使用：

```powershell
python -m ldm5.visualize `
  --ae_dir outputs_horizon/autoencoder/final `
  --output_dir outputs_horizon `
  --save_dir outputs_horizon/evaluation `
  --num_samples 6 `
  --skip_ldm
```

---

## 九、推荐训练命令

### 9.1 训练软层位约束 AE

```powershell
python -m ldm5.train_autoencoder `
  --data_root "标签" `
  --output_dir outputs_horizon `
  --batch_size 4 `
  --num_epochs 50 `
  --device cuda `
  --save_every 10
```

默认会使用：

```text
lambda_gradient = 0.1
lambda_channel_corr = 0.05
lambda_horizon = 0.2
lambda_layer_sync = 0.05
```

### 9.2 评估 AE

```powershell
python -m ldm5.evaluate `
  --ae_dir outputs_horizon/autoencoder/final `
  --output_dir outputs_horizon `
  --batch_size 4 `
  --num_eval_batches 24 `
  --save_dir outputs_horizon/evaluation `
  --skip_ldm
```

输出报告：

```text
outputs_horizon/evaluation/report.json
```

### 9.3 可视化 AE 与层位图

```powershell
python -m ldm5.visualize `
  --ae_dir outputs_horizon/autoencoder/final `
  --output_dir outputs_horizon `
  --save_dir outputs_horizon/evaluation `
  --num_samples 6 `
  --device cuda `
  --skip_ldm
```

输出图片：

```text
outputs_horizon/evaluation/real_samples.png
outputs_horizon/evaluation/ae_reconstruction.png
outputs_horizon/evaluation/ae_horizon_reconstruction.png
```

---

## 十、实验结果说明

在已有一次训练中，软层位约束 AE 相比原始 AE 的指标变化如下：

| 指标 | 原始 AE | 软层位约束 AE | 趋势 |
|------|---------|---------------|------|
| mean L1 | 0.1577 | 0.1565 | 略降 |
| mean PSNR | 29.92 dB | 30.07 dB | 略升 |
| mean SSIM | 0.9371 | 0.9540 | 明显提升 |
| horizon L1 | 0.2712 | 0.2166 | 明显下降 |
| sync L1 | 0.4922 | 0.4606 | 下降 |

结论：

1. 软层位约束没有破坏像素重构质量。
2. SSIM 提升说明结构相似性增强。
3. `horizon L1` 明显下降，说明重构图保留了更多可解释层位信息。
4. `sync L1` 下降，说明五通道边界同步性改善。

---

## 十一、与条件 U-Net 方案的区别

软层位约束方案和条件 U-Net 方案都使用层位先验，但作用位置不同。

| 方案 | 作用阶段 | 是否改变网络结构 | 是否需要推理条件输入 | 主要目标 |
|------|----------|------------------|----------------------|----------|
| 软层位约束 AE | AE 训练阶段 | 否 | 否 | 提升重构结构保真度 |
| 条件 U-Net | LDM 训练/采样阶段 | 是 | 是 | 按指定层位生成样本 |

当前实验对比表明，针对“重构图无法提取足够有效信息”的问题，软层位约束更直接有效，因为它作用在信息压缩与重构的瓶颈阶段。

---

## 十二、注意事项

### 12.1 不宜过度增大 `lambda_horizon`

如果 `lambda_horizon` 过大，模型可能过度关注边界位置，导致：

- 局部纹理被弱化；
- 属性值重构变差；
- 输出出现不自然锐化。

建议初始范围：

```text
0.1 <= lambda_horizon <= 0.3
```

### 12.2 `lambda_layer_sync` 应保持较小

`layer_sync_loss` 是逐通道边界约束，过大会放大各通道中的噪声边缘。推荐：

```text
0.02 <= lambda_layer_sync <= 0.1
```

### 12.3 软层位图来自归一化空间

当前 `soft_horizon_map` 在训练时基于归一化后的五通道张量计算。因此不同物理量纲已经被 z-score 标准化，避免 `vp/vs` 等数值范围较大的通道支配层位图。

### 12.4 该方案不替代真实人工层位解释

软层位图是从数据中自动提取的结构先验，不等价于人工解释层位。它适合作为训练正则项，用于增强结构保真；如果有高质量人工层位标签，可进一步替换或融合该自动层位图。

---

## 十三、文件变更摘要

| 文件 | 作用 |
|------|------|
| `horizon.py` | 提取软层位图、二值层位标签、条件层位图 |
| `losses.py` | 新增 `horizon_loss` 和 `layer_sync_loss` |
| `config.py` | 新增软层位约束默认权重 |
| `train_autoencoder.py` | 支持软层位约束训练和日志记录 |
| `evaluate.py` | 新增层位结构评估指标 |
| `visualize.py` | 新增软层位图重构对比图 |

---

## 十四、总结

软层位约束 AE 改进方案的核心思想是：

```text
不只要求重构图像数值接近，
还要求重构图在层位边界和跨通道同步结构上接近真实样本。
```

它通过自动提取五通道深度方向突变响应，构造软层位图，并将其作为训练正则项加入 AE 损失函数。该方法不需要人工标签，不改变模型结构，不增加推理成本，却能显著改善层位结构保真度，是当前 LDM5 流程中提升生成数据地质可解释性的直接切入点。
