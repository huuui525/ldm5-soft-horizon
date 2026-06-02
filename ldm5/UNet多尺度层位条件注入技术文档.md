# 五通道地震数据 U-Net 多尺度层位条件注入方案 — 原理与代码详解

---

## 一、改进方案整体概述

本方案是在 LDM5 第二阶段 `Latent DDPM` 的 U-Net 噪声预测器中加入**多尺度层位条件注入 (Multi-scale Horizon Conditioning)**。其目标不是改进 AE 重构，而是让扩散模型在生成 latent 时受到层位先验约束，从而生成符合指定层位结构的五通道地震属性数据。

原始 LDM 第二阶段为无条件生成：

```text
随机噪声 latent + timestep
        ↓
UNet2DModel
        ↓
预测噪声
```

多尺度层位条件方案改为：

```text
随机噪声 latent + timestep + 层位条件图
        ↓
MultiScaleHorizonUNet
        ↓
预测噪声
```

核心思想：

1. 从真实五通道样本中自动提取层位条件图；
2. 将层位图下采样到 latent 分辨率；
3. 在 U-Net 的不同下采样层级注入对应尺度的层位先验；
4. 浅层关注细节层位，深层关注大尺度构造边界；
5. 采样时给定层位条件图，生成符合该结构的五通道数据。

该方案与 AE 软层位约束的区别是：

| 方案 | 作用阶段 | 是否改变网络结构 | 是否需要采样条件 | 目标 |
|------|----------|------------------|------------------|------|
| AE 软层位约束 | 第一阶段 AE | 否 | 否 | 提升重构结构保真度 |
| U-Net 多尺度条件注入 | 第二阶段 LDM | 是 | 是 | 按指定层位结构生成 |

---

## 二、为什么在 U-Net 中注入层位条件

### 2.1 原始 LDM 的局限

原始 `train_latent_diffusion.py` 中，U-Net 前向过程为：

```python
pred = unet(noisy_latents, timesteps).sample
```

输入只有：

- 加噪后的 latent；
- 扩散时间步 `t`。

模型学习的是训练集整体分布，但无法显式指定：

- 生成样本有哪些层位；
- 层位在什么深度位置；
- 大构造边界如何变化；
- 五通道属性边界是否跟随某个指定层位结构。

因此，它适合无条件生成，但不适合“按层位结构控制生成”。

### 2.2 层位条件的作用

层位条件图告诉 U-Net：

```text
哪些空间位置应该存在强反射界面或属性突变。
```

在扩散去噪过程中，U-Net 每一步都根据该条件预测噪声，使最终 latent 在解码后更倾向于符合给定层位结构。

从地质角度看，层位条件图是一个结构先验：

- 约束层状几何结构；
- 保持近水平连续边界；
- 引导五通道属性在相近位置同步变化；
- 降低随机纹理式生成。

---

## 三、层位条件提取模块 (`horizon.py`)

### 3.1 软层位图 `soft_horizon_map`

```python
def soft_horizon_map(x):
    vertical = (x[..., 1:, :] - x[..., :-1, :]).abs()
    vertical = F.pad(vertical, (0, 0, 1, 0))
    horizon = vertical.mean(dim=1, keepdim=True)
    return normalize_map(horizon)
```

输入输出：

| 张量 | 形状 | 含义 |
|------|------|------|
| `x` | `[B, 5, H, W]` | 五通道归一化属性图 |
| `vertical` | `[B, 5, H, W]` | 各通道深度方向突变 |
| `horizon` | `[B, 1, H, W]` | 跨通道平均层位响应 |

数学形式：

```text
E_c(i, j) = |X_c(i, j) - X_c(i-1, j)|

H(i, j) = mean_c E_c(i, j)
```

其中：

- `c` 表示五个通道；
- `i` 表示深度方向；
- `j` 表示横向位置；
- `H` 是软层位条件图。

### 3.2 二值层位标签 `horizon_label_map`

```python
def horizon_label_map(x, quantile=0.75):
    soft = soft_horizon_map(x)
    flat = soft.flatten(start_dim=1)
    threshold = torch.quantile(flat, quantile, dim=1).view(-1, 1, 1, 1)
    return (soft >= threshold).to(dtype=x.dtype)
```

该函数将软层位图转换为二值标签：

```text
强层位位置 = soft_horizon >= 分位数阈值
```

默认 `quantile=0.75`，表示保留响应最强的 25% 位置作为层位标签。

### 3.3 条件图下采样 `horizon_condition`

```python
def horizon_condition(x, latent_size, mode="soft", quantile=0.75):
    if mode == "soft":
        cond = soft_horizon_map(x)
    elif mode == "label":
        cond = horizon_label_map(x, quantile=quantile)
    return F.interpolate(cond, size=latent_size, mode="bilinear")
```

原始图像尺寸为：

```text
[B, 5, 224, 224]
```

AE latent 尺寸为：

```text
[B, 8, 56, 56]
```

因此层位条件图会被缩放到：

```text
[B, 1, 56, 56]
```

此后 U-Net 内部会进一步按各下采样层级缩放到：

```text
56x56 → 28x28 → 14x14 → 7x7
```

### 3.4 标签落盘 `save_horizon_labels`

```python
save_horizon_labels(dataset, save_dir, mode="soft")
```

该函数可以把从现有数据中提取出的层位条件图保存为 `.npy` 文件。训练时可通过：

```powershell
--save_horizon_labels
```

生成：

```text
outputs_horizon_cond/horizon_latent_diffusion/horizon_soft_labels/
```

每个样本对应一个层位图文件。

---

## 四、条件 U-Net 模块 (`conditional_unet.py`)

### 4.1 总体结构

`conditional_unet.py` 中定义了两个核心类：

```python
class HorizonAdapter(nn.Module)
class MultiScaleHorizonUNet(nn.Module)
```

其中：

- `HorizonAdapter`：把一通道层位图投影为 U-Net 当前层级的特征通道数；
- `MultiScaleHorizonUNet`：包装 diffusers 的 `UNet2DModel`，在每个下采样块前注入层位条件。

### 4.2 层位适配器 `HorizonAdapter`

```python
class HorizonAdapter(nn.Module):
    def __init__(self, out_channels):
        self.net = nn.Sequential(
            nn.Conv2d(1, out_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
```

输入输出：

| 输入 | 输出 |
|------|------|
| `[B, 1, h, w]` | `[B, C_level, h, w]` |

作用：

1. 将一通道层位图升维到 U-Net 特征通道数；
2. 使用小卷积学习条件特征；
3. 通过零初始化最后一层，使训练开始时条件分支近似不影响原始 U-Net。

零初始化的意义：

```text
训练初期 ≈ 原始无条件 U-Net
随后逐步学习如何使用层位条件
```

这样可以减少训练不稳定。

### 4.3 多尺度注入层级

基础 U-Net 配置来自 `config.py`：

```python
block_out_channels = (64, 128, 256, 256)
```

对应 U-Net 下采样路径的特征尺度大致为：

| 层级 | 空间尺度 | 注入通道数 | 结构含义 |
|------|----------|------------|----------|
| Level 0 | 56x56 | 64 | 细节层位、薄层边界 |
| Level 1 | 28x28 | 64 | 中尺度层位连续性 |
| Level 2 | 14x14 | 128 | 较大构造边界 |
| Level 3 | 7x7 | 256 | 大尺度构造趋势 |

代码中适配器通道数为：

```python
block_out = list(unet_config["block_out_channels"])
inject_channels = [block_out[0]] + block_out[:-1]
```

即：

```text
[64, 64, 128, 256]
```

这是因为条件注入发生在每个 `down_block` 执行之前，此时特征通道数对应当前输入特征，而不是当前 block 输出特征。

### 4.4 条件注入位置

在 `MultiScaleHorizonUNet.forward` 中：

```python
sample = unet.conv_in(sample)

for level, downsample_block in enumerate(unet.down_blocks):
    cond = self.horizon_adapters[level](horizon, sample.shape[-2:])
    sample = sample + self.condition_scale * cond
    sample, res_samples = downsample_block(hidden_states=sample, temb=emb)
```

即：

```text
每个 down block 前：
当前 U-Net 特征 + 对应尺度层位条件特征
```

这样做的原因：

1. 下采样路径决定后续编码到瓶颈的信息；
2. 早期注入能影响局部纹理和边界；
3. 深层注入能影响整体结构和大尺度趋势；
4. 跳连中也会携带受条件影响的多尺度特征。

### 4.5 条件强度 `condition_scale`

```python
sample = sample + self.condition_scale * cond
```

`condition_scale` 控制层位条件对 U-Net 特征的影响强度。

默认：

```text
condition_scale = 1.0
```

建议范围：

```text
0.5 <= condition_scale <= 2.0
```

如果生成结果不明显跟随层位条件，可以适当增大；如果生成结果过度贴合边界、纹理不自然，可以减小。

### 4.6 保存与加载

该 wrapper 提供：

```python
save_pretrained(save_dir)
from_pretrained(load_dir)
```

保存内容：

```text
config.json
pytorch_model.bin
```

`config.json` 中包含：

```json
{
  "unet_config": ...,
  "condition_channels": 1,
  "condition_scale": 1.0,
  "_class_name": "MultiScaleHorizonUNet"
}
```

---

## 五、训练流程 (`train_horizon_latent_diffusion.py`)

### 5.1 训练入口

```powershell
python -m ldm5.train_horizon_latent_diffusion `
  --ae_dir outputs/autoencoder/final `
  --output_dir outputs_horizon_cond `
  --batch_size 4 `
  --num_epochs 50 `
  --device cuda
```

### 5.2 Autoencoder 冻结

训练脚本加载已有 AE：

```python
ae = AutoencoderKL.from_pretrained(args.ae_dir).to(args.device)
ae.eval()
for param in ae.parameters():
    param.requires_grad_(False)
```

该阶段不会更新 AE 参数。AE 只用于：

1. 将真实五通道样本编码到 latent；
2. 采样后将 generated latent 解码回五通道图。

### 5.3 latent 形状探测

```python
probe = dataset[0].unsqueeze(0).to(args.device)
z = ae.encode(probe).latent_dist.sample()
cfg.unet.sample_size = z.shape[-1]
cfg.unet.in_channels = z.shape[1]
cfg.unet.out_channels = z.shape[1]
```

实际 latent 形状为：

```text
[1, 8, 56, 56]
```

因此条件 U-Net 的输入输出仍然是：

```text
[B, 8, 56, 56] → [B, 8, 56, 56]
```

层位条件为：

```text
[B, 1, 56, 56]
```

### 5.4 训练 batch 流程

每个 batch 的训练过程：

```python
batch = batch.to(device)

latents = ae.encode(batch).latent_dist.sample()
horizon = horizon_condition(batch, latent_size=(56, 56), mode=args.condition_mode)

noise = torch.randn_like(latents)
timesteps = torch.randint(0, 1000, (B,))
noisy_latents = scheduler.add_noise(latents, noise, timesteps)

pred = unet(noisy_latents, timesteps, horizon=horizon).sample
loss = F.mse_loss(pred, noise)
```

完整逻辑：

```text
真实五通道数据
        ↓
冻结 AE 编码为 latent
        ↓
从同一真实数据提取层位条件图
        ↓
DDPM 前向加噪
        ↓
条件 U-Net 预测噪声
        ↓
MSE(pred_noise, true_noise)
```

### 5.5 条件模式

训练脚本支持：

```powershell
--condition_mode soft
--condition_mode label
```

两种模式区别：

| 模式 | 条件图 | 特点 |
|------|--------|------|
| `soft` | 连续层位响应 | 信息更丰富，默认推荐 |
| `label` | 二值层位标签 | 更接近分割图，但阈值敏感 |

默认使用：

```text
condition_mode = soft
```

### 5.6 层位标签保存

通过：

```powershell
--save_horizon_labels
```

可将提取出的层位图保存到：

```text
outputs_horizon_cond/horizon_latent_diffusion/horizon_soft_labels/
```

在一次实验中共保存 192 个 `.npy` 层位图，与训练样本数量一致。

---

## 六、采样与可视化 (`visualize_horizon_diffusion.py`)

### 6.1 可视化入口

```powershell
python -m ldm5.visualize_horizon_diffusion `
  --ae_dir outputs/autoencoder/final `
  --unet_dir outputs_horizon_cond/horizon_latent_diffusion/final `
  --output_dir outputs_horizon_cond `
  --save_dir outputs_horizon_cond/horizon_evaluation `
  --num_samples 6 `
  --num_inference_steps 100 `
  --device cuda
```

### 6.2 条件来源

可视化脚本从真实样本中抽取若干样本：

```python
real_norm = torch.stack([dataset[int(i)] for i in idxs])
cond = horizon_condition(real_norm, latent_size=56, mode=args.condition_mode)
```

这些真实样本只用于提取层位条件。生成过程仍然从随机噪声 latent 开始：

```python
latents = torch.randn(shape, device=args.device)
```

### 6.3 条件采样流程

```python
for timestep in scheduler.timesteps:
    pred = unet(latents, timestep, horizon=cond).sample
    latents = scheduler.step(pred, timestep, latents).prev_sample

gen_norm = ae.decode(latents).sample.cpu()
```

即：

```text
随机 latent 噪声
        ↓
条件 U-Net 多步去噪
        ↓
生成 latent
        ↓
冻结 AE 解码
        ↓
五通道生成数据
```

### 6.4 输出文件

脚本输出：

```text
condition_real_samples.png
horizon_conditions.png
horizon_conditioned_samples.png
horizon_visualize_arrays.npz
```

含义：

| 文件 | 内容 |
|------|------|
| `condition_real_samples.png` | 用于提取条件的真实五通道样本 |
| `horizon_conditions.png` | 输入给 U-Net 的层位条件图 |
| `horizon_conditioned_samples.png` | 按该层位条件生成的五通道样本 |
| `horizon_visualize_arrays.npz` | 真实样本、生成样本、条件图的原始数组 |

---

## 七、与文本/类别嵌入方案的区别

### 7.1 类别嵌入方式

如果使用 `UNet2DModel` 的 `class_labels`，条件通常是一个离散类别：

```text
class_id → embedding → 加到 timestep embedding
```

优点：

- 改动小；
- 适合明确类别标签。

缺点：

- 难表达连续层位几何；
- 空间位置信息弱；
- 无法告诉模型“边界具体在哪里”；
- 对小数据集，类别划分容易稀疏。

### 7.2 多尺度层位图注入方式

本方案直接把层位图作为空间条件：

```text
horizon map [B, 1, 56, 56]
```

并注入 U-Net 多个空间尺度。

优势：

- 保留层位空间位置；
- 保留边界连续性；
- 浅层和深层分别接收不同尺度条件；
- 更适合地震剖面这类空间结构强的数据。

### 7.3 与 cross-attention 的区别

Stable Diffusion 常用 cross-attention 注入文本条件。但当前层位条件本质上是二维结构图，不是 token 序列。

因此，本方案选择卷积适配器注入，而不是 cross-attention：

| 方式 | 更适合的条件 | 本任务适配性 |
|------|--------------|--------------|
| cross-attention | 文本、token、全局语义 | 中等 |
| 多尺度卷积注入 | 图像、mask、边界图、层位图 | 高 |

---

## 八、实验结果说明

一次完整训练配置：

```text
GPU: cuda
batch_size: 4
epochs: 50
condition_mode: soft
condition_scale: 1.0
AE: outputs/autoencoder/final，只加载冻结
条件 U-Net: outputs_horizon_cond/horizon_latent_diffusion/final
```

训练日志路径：

```text
outputs_horizon_cond/logs/horizon_latent_diffusion/20260527_145659/
```

最终模型路径：

```text
outputs_horizon_cond/horizon_latent_diffusion/final/
```

训练损失在 50 epoch 内从约 `0.7283` 降到约 `0.2529`。该数值是噪声预测 MSE，与无条件 LDM 的训练损失量级接近，说明多尺度条件分支没有破坏 DDPM 噪声预测训练流程。

可视化路径：

```text
outputs_horizon_cond/horizon_evaluation/
```

包含：

```text
condition_real_samples.png
horizon_conditions.png
horizon_conditioned_samples.png
```

---

## 九、推荐命令

### 9.1 Smoke Test

```powershell
python -m ldm5.train_horizon_latent_diffusion `
  --ae_dir outputs/autoencoder/final `
  --output_dir outputs_horizon_cond `
  --batch_size 4 `
  --device cuda `
  --smoke `
  --save_horizon_labels
```

用途：

- 检查 AE 加载；
- 检查层位条件提取；
- 检查条件 U-Net 前向与反传；
- 可顺便生成层位标签文件。

### 9.2 正式训练

```powershell
python -m ldm5.train_horizon_latent_diffusion `
  --ae_dir outputs/autoencoder/final `
  --output_dir outputs_horizon_cond `
  --batch_size 4 `
  --num_epochs 50 `
  --device cuda `
  --save_every 10
```

### 9.3 使用二值层位标签训练

```powershell
python -m ldm5.train_horizon_latent_diffusion `
  --ae_dir outputs/autoencoder/final `
  --output_dir outputs_horizon_cond_label `
  --batch_size 4 `
  --num_epochs 50 `
  --device cuda `
  --condition_mode label `
  --label_quantile 0.75
```

### 9.4 条件采样可视化

```powershell
python -m ldm5.visualize_horizon_diffusion `
  --ae_dir outputs/autoencoder/final `
  --unet_dir outputs_horizon_cond/horizon_latent_diffusion/final `
  --output_dir outputs_horizon_cond `
  --save_dir outputs_horizon_cond/horizon_evaluation `
  --num_samples 6 `
  --num_inference_steps 100 `
  --device cuda
```

---

## 十、注意事项

### 10.1 该方案需要采样时提供层位条件

条件 U-Net 的 forward 签名为：

```python
unet(noisy_latents, timesteps, horizon=condition)
```

因此，推理时也必须提供 `horizon`。如果希望无条件生成，需要使用原始 `UNet2DModel` 或额外实现条件 dropout / classifier-free guidance。

### 10.2 当前没有实现 CFG

当前版本没有做：

```text
conditional - unconditional
```

这种 classifier-free guidance。若后续需要调节层位条件强度，可以进一步加入：

1. 训练时随机置零 `horizon`；
2. 推理时分别跑有条件和无条件预测；
3. 使用 guidance scale 混合。

### 10.3 条件图来自真实样本自动提取

当前训练和可视化中的层位图是从五通道数据自动提取的，不是人工解释层位。

优点：

- 不需要额外标注；
- 与现有数据完全对齐；
- 可快速训练。

限制：

- 层位质量受原始数据噪声影响；
- 强边缘不一定全部是真实地质层位；
- 如果未来有人工层位解释，应优先考虑融合人工标签。

### 10.4 条件分支参数与原 U-Net 一起训练

`MultiScaleHorizonUNet` 中包含完整 `UNet2DModel` 和额外 `HorizonAdapter`。训练时二者都会更新。它不是在已有无条件 U-Net 上直接微调，除非后续手动加载无条件 U-Net 权重。

### 10.5 与 AE 软层位约束可组合

该方案可与 AE 软层位约束组合：

```text
软层位约束 AE 负责更好地编码/解码层位结构
条件 U-Net 负责按给定层位结构生成 latent
```

实际使用时可以：

1. 先训练软层位约束 AE；
2. 冻结该 AE；
3. 再训练多尺度层位条件 U-Net。

---

## 十一、文件变更摘要

| 文件 | 作用 |
|------|------|
| `horizon.py` | 自动提取软层位图、二值层位标签和 latent 尺度条件图 |
| `conditional_unet.py` | 定义 `HorizonAdapter` 与 `MultiScaleHorizonUNet` |
| `train_horizon_latent_diffusion.py` | 训练多尺度层位条件 latent DDPM |
| `visualize_horizon_diffusion.py` | 可视化条件图与条件生成样本 |

---

## 十二、总结

U-Net 多尺度层位条件注入方案的核心思想是：

```text
把层位图作为空间先验，
在 U-Net 下采样路径的多个尺度中持续注入，
让扩散去噪过程从局部细节到全局结构都受到层位约束。
```

它比类别嵌入更适合当前任务，因为层位不是离散语义类别，而是具有明确空间位置的二维地质结构。该方案能够让模型按给定层位图生成五通道地震数据，是从“无条件生成”走向“结构可控生成”的关键一步。
