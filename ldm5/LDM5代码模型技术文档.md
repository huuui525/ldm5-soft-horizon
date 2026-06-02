# LDM5 五通道地震潜在扩散模型 — 代码模型技术文档

---

## 一、整体架构概述

`ldm5` 是一个面向五通道地震属性数据的潜在扩散模型工程。输入数据来自五个属性通道：

```text
dn, gas, gr, vp, vs
```

每个样本最终被组织成 `[5, 224, 224]` 的张量。项目采用两阶段建模思路：

1. **第一阶段：AutoencoderKL 重建与压缩**
   - 将五通道地震数据从像素空间压缩到潜在空间。
   - 默认从 `[B, 5, 224, 224]` 压缩到 `[B, 8, 56, 56]`。
   - 训练目标包含重建、梯度、通道相关性、频率、软层位和层同步等约束。

2. **第二阶段：Latent DDPM 生成**
   - 冻结第一阶段 AE。
   - 将真实样本编码到 latent 后加噪。
   - 训练 U-Net 在 latent 空间预测噪声。
   - 支持普通无条件扩散和多尺度层位条件扩散。

整体流程如下：

```text
五通道 .npy 数据
    ↓
样本匹配、尺寸统一、逐通道归一化
    ↓
AutoencoderKL 编码 / 解码
    ↓
latent 表示 [B, 8, 56, 56]
    ↓
DDPM 加噪与 U-Net 噪声预测
    ↓
反向采样得到 latent
    ↓
AE 解码得到五通道生成样本
```

| 组件 | 输入 | 输出 | 作用 |
|------|------|------|------|
| 数据集 `FiveChannelSeismicDataset` | 五个通道目录中的 `.npy` | `[5, H, W]` 归一化张量 | 统一样本、通道和尺寸 |
| `AutoencoderKL` | `[B, 5, 224, 224]` | latent `[B, 8, 56, 56]` / 重建图 | 学习低维地震表示 |
| 普通 `UNet2DModel` | 加噪 latent + timestep | 噪声预测 | 学习 latent 去噪分布 |
| `MultiScaleHorizonUNet` | 加噪 latent + timestep + 层位图 | 噪声预测 | 按层位结构控制生成 |
| `DDPMScheduler` / `DDIMScheduler` / `DPMSolver` | latent 与预测噪声 | 下一步 latent | 控制扩散加噪和反向采样 |

---

## 二、代码结构与模块职责

| 文件 | 核心对象 | 作用 | 依赖/被调用位置 |
|------|----------|------|----------------|
| `config.py` | `DataConfig`, `AEConfig`, `UNetConfig`, `TrainConfig` | 集中定义数据、模型和训练默认参数 | 所有训练、评估、生成脚本 |
| `data.py` | `FiveChannelSeismicDataset`, `build_dataset` | 发现样本、读取五通道 `.npy`、计算/加载归一化统计量 | 训练、评估、可视化 |
| `models.py` | `build_autoencoder`, `build_unet`, `build_scheduler` | 构建 diffusers 组件 | AE 训练、LDM 训练 |
| `losses.py` | `autoencoder_loss` 及辅助损失 | 定义 AE 阶段的复合损失 | `train_autoencoder.py`, `evaluate.py` |
| `horizon.py` | `soft_horizon_map`, `horizon_condition` | 从五通道数据中提取层位先验 | AE 损失、条件扩散训练/可视化 |
| `conditional_unet.py` | `HorizonAdapter`, `MultiScaleHorizonUNet` | 在 U-Net 下采样路径注入层位条件 | `train_horizon_latent_diffusion.py` |
| `train_autoencoder.py` | `main` | 第一阶段训练 AE | 输出 `outputs/autoencoder` |
| `train_latent_diffusion.py` | `main` | 第二阶段训练普通 latent DDPM | 输出 `outputs/latent_diffusion` |
| `train_horizon_latent_diffusion.py` | `main` | 第二阶段训练层位条件 DDPM | 输出 `outputs/horizon_latent_diffusion` |
| `evaluate.py` | `evaluate_ae`, `evaluate_ldm` | 计算 AE 重建指标和 LDM 分布指标 | 输出 `outputs/evaluation/report.json` |
| `visualize.py` | `grid_figure`, `main` | 可视化真实样本、AE 重建、普通 LDM 生成 | 输出 PNG 和 `.npz` |
| `visualize_horizon_diffusion.py` | `main` | 可视化层位条件、真实样本、条件生成结果 | 输出层位条件生成图 |
| `generate_dataset.py` | `generate_dataset_from_noise`, `save_channel_dataset` | 从纯噪声采样生成新数据集 | 输出五通道目录和元数据 |
| `count_params.py` | `report` | 统计 AE 和 U-Net 参数量 | 独立工具 |
| `bench.py` | 脚本级 benchmark | 粗略测试 AE 单步训练速度 | 独立工具 |
| `logger.py` | `TrainLogger` | 写入训练日志和 CSV 指标 | 训练脚本 |

---

## 三、数据与配置模块

### 3.1 配置对象 `config.py`

`config.py` 使用 `dataclass` 管理默认参数。这样做的好处是训练、评估、生成脚本都从同一个默认配置出发，减少硬编码。

| 配置类 | 关键字段 | 作用 |
|--------|----------|------|
| `DataConfig` | `data_root`, `channels`, `image_size`, `stats_path` | 指定数据路径、五通道名称、输入尺寸和统计量缓存 |
| `AEConfig` | `in_channels=5`, `latent_channels=8`, `block_out_channels=(64,128,256)` | 定义五通道 AE 的输入输出和压缩结构 |
| `UNetConfig` | `sample_size=56`, `in_channels=8`, `block_out_channels=(64,128,256,256)` | 定义 latent U-Net 的空间尺寸和通道规模 |
| `TrainConfig` | `batch_size`, `num_epochs`, `learning_rate`, `lambda_*`, `num_train_timesteps` | 定义训练超参数和损失权重 |

从代码可见，AE 默认下采样因子是 4，因此 `224 / 4 = 56`，对应 latent 空间分辨率。

### 3.2 样本发现与通道匹配 `data.py`

`discover_sample_ids` 会遍历每个通道目录，找出所有通道都存在的样本 ID。它使用文件名交集保证同一个样本在五个通道中都有对应 `.npy` 文件。

```text
data_root/
  dn/  000001.npy ...
  gas/ 000001.npy ...
  gr/  000001.npy ...
  vp/  000001.npy ...
  vs/  000001.npy ...
```

该设计避免了某个通道缺样本导致训练时通道错位。若某个通道目录不存在，会直接抛出 `FileNotFoundError`；若没有公共样本，则抛出 `RuntimeError`。

### 3.3 数组规范化 `_normalize_array`

`_normalize_array` 将原始 `.npy` 统一成 `[H, W]` 的 `float32`：

- 二维数组直接使用。
- 三维数组只接受单通道形式，如 `[1, H, W]` 或 `[H, W, 1]`。
- 多通道三维数组会被认为有歧义并报错。

这一步的原理是：五通道由五个文件夹显式提供，而不是从单个 `.npy` 的第三维拆出来，因此每个单通道文件必须能明确还原为二维剖面。

### 3.4 逐通道统计量与 Z-score 归一化

`compute_channel_stats` 对每个通道独立计算均值和标准差，并缓存到 `outputs/norm_stats.json`。训练时使用：

```text
x_norm = (x - mean) / std
```

逐通道归一化的意义是：`dn/gas/gr/vp/vs` 的物理量纲和数值范围不同，如果直接混合训练，数值范围大的通道会主导损失。

`FiveChannelSeismicDataset.__getitem__` 的核心流程为：

```text
读取同一 sample_id 的五个通道
    ↓
堆叠为 [5, H, W]
    ↓
必要时双线性插值到 224×224
    ↓
逐通道 Z-score 归一化
```

`denormalize` 则用于评估和可视化，把归一化空间的数据还原到原始物理量纲。

---

## 四、模型与算法核心模块

### 4.1 `models.py` / `build_autoencoder`

`build_autoencoder` 构建 Hugging Face diffusers 的 `AutoencoderKL`。它不是普通自编码器，而是带有概率潜变量分布的 VAE 风格自编码器。

| 阶段 | 输入 | 输出 | 说明 |
|------|------|------|------|
| 编码器 | `[B, 5, 224, 224]` | latent 分布 | 输出可采样的潜在分布 |
| latent 采样 | latent 分布 | `[B, 8, 56, 56]` | 作为压缩表示 |
| 解码器 | `[B, 8, 56, 56]` | `[B, 5, 224, 224]` | 重建五通道地震图 |

代码中会检查：

```text
expected_factor = 2 ** (len(block_out_channels) - 1)
```

默认 `block_out_channels=(64,128,256)`，因此有 3 个 block，实际下采样次数为 2 次，压缩因子为 `2^2 = 4`。这个检查可以防止配置中的 `downsample_factor` 与网络结构不一致。

### 4.2 `models.py` / `build_unet`

`build_unet` 构建 `UNet2DModel`，用于 latent 空间的 DDPM 噪声预测。

输入不是原始五通道图像，而是 AE 编码后的 latent：

```text
noisy_latents [B, 8, 56, 56] + timestep
    ↓
UNet2DModel
    ↓
predicted_noise [B, 8, 56, 56]
```

默认 U-Net 中间包含注意力块：

- 下采样第 3 个 block 为 `AttnDownBlock2D`。
- 上采样第 2 个 block 为 `AttnUpBlock2D`。

注意力块的作用是增强远距离结构关系建模，对地震剖面中连续层位和较大构造边界有帮助。

### 4.3 `horizon.py` / 软层位图提取

`soft_horizon_map` 从五通道图中提取一张一通道层位响应图。核心思想是：近水平层位边界通常表现为深度方向的突变，因此计算垂向差分：

```text
E_c(i,j) = |X_c(i,j) - X_c(i-1,j)|
H(i,j) = mean_c E_c(i,j)
```

其中 `c` 是通道索引。多个通道取平均后得到跨通道共享的层位响应。最后通过 `normalize_map` 做单样本标准化，使不同样本的层位强度在统一尺度上比较。

`horizon_label_map` 进一步把软图转换为二值标签图：对每个样本取分位数阈值，默认 `quantile=0.75`，高于阈值的位置视作层位边界。

`horizon_condition` 则把软图或标签图插值到 latent 分辨率，供条件扩散 U-Net 使用。

### 4.4 `losses.py` / AE 复合损失

AE 阶段不是只做像素重建，而是组合多个约束：

| 损失 | 代码函数 | 原理 | 作用 |
|------|----------|------|------|
| 重建损失 | `reconstruction_loss` | `L1(pred, target)` | 保证整体数值接近 |
| 梯度损失 | `gradient_loss` | 匹配横向和纵向差分 | 保留边缘、断层、反射界面 |
| 通道相关损失 | `channel_correlation_loss` | 匹配每个样本的 5×5 通道相关矩阵 | 保持属性通道之间的物理关联 |
| 频率损失 | `frequency_loss` | 匹配二维 FFT 幅值 | 约束频谱结构 |
| 软层位损失 | `horizon_loss` | 匹配多尺度层位响应图 | 提升层位边界保真度 |
| 层同步损失 | `layer_sync_loss` | 匹配各通道垂向边缘 | 保持五通道层位突变同步 |

`autoencoder_loss` 根据配置中的 `lambda_*` 动态启用各项损失。默认启用梯度、通道相关、软层位和层同步，频率损失默认关闭。

训练脚本中还额外加入：

```text
total = loss + 1e-6 * KL
```

KL 项用于约束 AE latent 分布，但权重非常小，说明当前更重视重建质量和结构保真。

### 4.5 `conditional_unet.py` / 多尺度层位条件 U-Net

`MultiScaleHorizonUNet` 是对 diffusers `UNet2DModel` 的包装。它保留原 U-Net 的主体结构，同时在每个 down block 前加入层位条件。

核心组件是 `HorizonAdapter`：

```text
一通道 horizon map
    ↓
Conv 3×3 + SiLU + Conv 1×1
    ↓
与当前 U-Net feature 相同通道数的条件特征
```

然后在每个下采样层执行：

```text
sample = sample + condition_scale * adapter(horizon)
```

这种方式的意义是：

- 浅层注入高分辨率层位细节。
- 深层注入下采样后的整体结构边界。
- U-Net 仍然学习噪声预测，但预测过程受层位图引导。

`HorizonAdapter` 最后一层卷积初始化为零，这样模型刚开始训练时接近原始 U-Net，不会因为新条件分支随机扰动过大而破坏训练稳定性。

---

## 五、训练、推理与生成流程

### 5.1 第一阶段：`train_autoencoder.py`

AE 训练流程如下：

```text
读取五通道 batch
    ↓
AutoencoderKL.encode 得到 posterior
    ↓
posterior.sample 得到 latent
    ↓
AutoencoderKL.decode 得到 recon
    ↓
计算复合重建损失 + KL
    ↓
AdamW 更新 AE
    ↓
写日志、保存 checkpoint
```

关键实现点：

- 支持命令行覆盖数据路径、batch size、epoch、学习率和各损失权重。
- 默认使用 CUDA AMP 混合精度，`--no_amp` 可关闭。
- `--smoke` 可只跑一个 batch，用于快速验证环境和代码链路。
- 输出目录为 `outputs/autoencoder/`，最终模型保存到 `outputs/autoencoder/final`。
- 配置和统计量写入 `config.json`，便于复现实验。

### 5.2 第二阶段普通扩散：`train_latent_diffusion.py`

普通 latent DDPM 训练流程：

```text
加载并冻结 AE
    ↓
真实五通道 batch 编码为 latent
    ↓
随机采样 timestep
    ↓
DDPMScheduler.add_noise 加噪
    ↓
U-Net 预测噪声
    ↓
MSE(pred, noise)
    ↓
更新 U-Net
```

该脚本会先用一个样本探测真实 latent shape，并自动更新 `cfg.unet.sample_size/in_channels/out_channels`。这可以避免 AE 配置变化后 U-Net 尺寸仍保持旧值。

### 5.3 第二阶段层位条件扩散：`train_horizon_latent_diffusion.py`

层位条件扩散在普通 DDPM 基础上多了一条条件分支：

```text
真实五通道 batch
    ↓
提取 horizon_condition
    ↓
编码为 latent 并加噪
    ↓
MultiScaleHorizonUNet(noisy_latents, timestep, horizon)
    ↓
预测噪声并计算 MSE
```

可选参数包括：

- `--condition_mode soft|label`：使用软层位图或二值层位标签。
- `--label_quantile`：二值标签阈值分位数。
- `--condition_scale`：条件注入强度。
- `--save_horizon_labels`：将提取出的层位图保存为 `.npy` 便于检查。

从代码可见，该条件图来自真实样本本身，因此训练目标是让模型学会“在给定层位结构时生成符合该结构的数据”。

### 5.4 生成数据集：`generate_dataset.py`

该脚本从纯 latent 噪声生成新的五通道样本。流程如下：

```text
随机 latent 噪声
    ↓
DDPM/DDIM/DPM 反向采样
    ↓
AE.decode
    ↓
归一化空间五通道样本
    ↓
denormalize 回原始物理量纲
    ↓
保存 combined、逐通道目录、预览图、metadata
```

支持三种采样器：

| 参数 | 采样器 | 特点 |
|------|--------|------|
| `ddpm` | `DDPMScheduler` | 随机性强，符合原始 DDPM 流程 |
| `ddim` | `DDIMScheduler` | 可更快、可确定性采样 |
| `dpm` | `DPMSolverMultistepScheduler` | 多步求解器，通常用于加速 |

输出目录包含：

- `combined/`：每个样本保存完整 `[5,H,W]`。
- `dn/gas/gr/vp/vs/`：按通道拆分保存。
- `samples_original.npy`：原始量纲数组。
- `samples_normalized.npy`：归一化空间数组。
- `metadata.json`：采样参数、模型路径、统计量和保存路径。
- `gen_<scheduler>_grid.png`：生成样本预览图。

---

## 六、评估、可视化与工具脚本

### 6.1 `evaluate.py`

`evaluate_ae` 计算 AE 重建指标：

- 每通道 L1。
- 每通道 MSE。
- 每通道 PSNR。
- 每通道 SSIM。
- 软层位 L1。
- 层同步 L1。

`evaluate_ldm` 用生成样本和真实样本的逐通道均值/标准差差异评估分布是否接近。它不是完整生成质量评价，但可以快速判断生成结果是否出现明显均值漂移或方差坍塌。

输出为：

```text
outputs/evaluation/report.json
outputs/evaluation/ldm_samples_normalized.npy
```

### 6.2 `visualize.py`

该脚本生成三类图：

- `real_samples.png`：真实五通道样本。
- `ae_reconstruction.png`：真实样本与 AE 重建对比。
- `ae_horizon_reconstruction.png`：真实/重建层位图和误差。
- `ldm_samples.png`：普通 latent DDPM 生成样本。

同时保存 `visualize_arrays.npz`，便于后续重新画图或做定量分析。

### 6.3 `visualize_horizon_diffusion.py`

该脚本用于检查条件扩散：

```text
选取真实样本
    ↓
提取层位条件
    ↓
从随机 latent 采样
    ↓
使用 MultiScaleHorizonUNet 按条件生成
    ↓
保存真实图、条件图、生成图
```

输出包括：

- `condition_real_samples.png`
- `horizon_conditions.png`
- `horizon_conditioned_samples.png`
- `horizon_visualize_arrays.npz`

### 6.4 `logger.py`

`TrainLogger` 为每次训练建立时间戳目录，并输出：

- `train.log`：人工可读日志。
- `metrics.csv`：step 级指标。
- `epoch_metrics.csv`：epoch 均值指标。

这使训练曲线可以用 Excel、Python 或其他工具直接分析。

### 6.5 `count_params.py` 与 `bench.py`

`count_params.py` 用于统计 AE、U-Net 和总模型参数量，可选择将结果导出为 JSON。它不依赖数据集，适合在调整配置后快速比较模型规模。

`bench.py` 构造随机 `[8,5,224,224]` 输入并执行 AE 训练步，用于粗略隔离 GPU 计算速度。它不评估模型质量，只用于性能诊断。

---

## 七、当前实现特点与局限

### 7.1 实现特点

1. **代码分层清晰**
   - 数据、模型、损失、训练、评估、可视化分文件组织，便于维护。

2. **采用 latent diffusion，降低生成建模难度**
   - 先把五通道图压缩到 latent，再训练扩散模型，比直接在 224×224 原图空间扩散更省显存和计算。

3. **面向地震结构加入专门约束**
   - AE 阶段加入软层位损失和层同步损失。
   - 条件扩散阶段加入多尺度层位条件注入。

4. **输出链路比较完整**
   - 支持训练、评估、可视化、生成数据集、统计参数和 benchmark。

5. **支持快速 smoke test**
   - 训练脚本可以只跑一个 batch，便于排查环境问题。

### 7.2 当前局限

1. **数据划分不足**
   - 代码当前主要在训练集上评估 AE，缺少显式 train/val/test 划分。

2. **生成质量指标偏基础**
   - LDM 评估主要比较均值和方差，缺少更强的结构相似性、频谱、层位一致性或地质合理性指标。

3. **条件生成暂未在生成数据集脚本中集成**
   - `generate_dataset.py` 只加载普通 `UNet2DModel`，没有生成层位条件数据集的路径。

4. **配置管理仍偏脚本化**
   - 配置来自默认 dataclass 和命令行参数，缺少统一 YAML/JSON 实验配置入口。

5. **缺少自动化测试**
   - 当前没有测试用例覆盖数据读取、shape 检查、损失函数、模型前向和保存加载。

6. **默认路径存在编码/显示风险**
   - `DataConfig.data_root` 中默认字符串在当前环境显示为乱码，建议统一改成 ASCII 路径或明确文档说明。

---

## 八、可行优化方案

| 方向 | 问题/动机 | 优化方案 | 预期收益 | 风险/成本 | 验证方式 |
|------|-----------|----------|----------|-----------|----------|
| 数据划分 | 训练集评估不能代表泛化 | 在 `data.py` 增加固定 train/val/test split，支持 seed 和 split 文件 | 更准确判断 AE 与 LDM 泛化能力 | 需要调整训练/评估入口 | 对比训练集和验证集指标 |
| 评估指标 | 均值/方差不足以评价地震结构 | 增加频谱差异、层位图相似度、通道相关矩阵差异、结构连续性指标 | 更贴合地震数据质量 | 指标设计需验证有效性 | 与可视化结果和专家判断对齐 |
| 条件生成 | 条件扩散只能可视化，不能批量生成数据集 | 新增 `generate_horizon_dataset.py` 或扩展 `generate_dataset.py` 支持 `MultiScaleHorizonUNet` | 能按给定层位结构批量生成样本 | 需要定义条件来源 | 比较生成图与条件图的层位一致性 |
| 模型结构 | U-Net 条件只注入 down blocks | 尝试在 mid/up blocks 注入条件，或使用 cross-attention/ControlNet 风格分支 | 提升条件控制强度 | 参数量和训练复杂度增加 | 做 ablation：down-only vs down+mid vs full |
| 损失权重 | 固定 `lambda_*` 可能不是最优 | 加入损失权重 warmup 或动态调度 | 训练前期更稳定，后期更重结构 | 调参成本增加 | 观察重建、层位、PSNR 曲线 |
| AE latent | 默认 latent_channels=8 未系统比较 | 实验 4/8/16 通道和不同 block 宽度 | 找到压缩率和重建质量平衡 | 多组实验成本 | 比较重建指标、扩散训练 loss、采样图 |
| 采样效率 | DDPM 采样步数多 | 训练/生成统一支持 DDIM/DPM，并记录采样器配置 | 提升生成速度 | 不同采样器质量可能不同 | 相同步数下比较质量和耗时 |
| 工程配置 | 命令行参数分散 | 增加统一实验配置文件加载，如 `--config experiment.yaml` | 实验可复现、易管理 | 需要兼容现有 CLI | 检查 config dump 与命令行覆盖 |
| 日志分析 | CSV 需要手动画图 | 增加训练曲线自动绘图脚本 | 便于汇报和排查 | 实现成本低 | 自动输出 loss/PSNR/horizon 曲线 |
| 自动化测试 | 代码修改后缺少回归保障 | 增加最小测试：数据 shape、损失非 NaN、AE/UNet 前向、保存加载 | 降低后续改动风险 | 需要构造临时小数据 | `pytest` smoke tests |

---

## 九、建议的实验推进顺序

1. **先补验证集**
   - 这是判断优化是否有效的前提。

2. **再补结构化评估**
   - 优先加入层位图相似度、通道相关矩阵差异和频谱差异。

3. **做 AE 损失权重消融**
   - 对比无层位、软层位、层同步、多尺度层位组合。

4. **做条件扩散消融**
   - 对比普通 U-Net、down-block 条件注入、更多位置条件注入。

5. **最后扩展批量条件生成**
   - 在模型质量和条件有效性确认后，再生成大规模数据集更稳妥。

---

## 十、总结

从代码可见，`ldm5` 已经形成了一个较完整的五通道地震潜在扩散工程：它先用 `AutoencoderKL` 学习五通道数据的压缩表示，再在 latent 空间训练扩散 U-Net；同时又针对地震层位结构加入了 AE 软层位约束和 U-Net 多尺度层位条件注入。

当前代码的主要价值在于：不仅能生成五通道地震属性样本，还把“层位边界、跨通道同步、结构连续性”这些地震数据中的关键结构信息显式放进训练目标和条件生成路径中。后续如果补齐验证集、结构化指标、条件批量生成和自动化测试，整个工程会更适合长期实验迭代和正式汇报展示。
