# LDM5 Soft Horizon Autoencoder

五通道地震属性数据的潜在扩散模型实验项目。当前主线版本聚焦于 **Autoencoder 软层位约束改进**：在 AE 重构训练中加入软层位图损失和跨通道层位同步损失，以提升重构图中的层位边界、薄层结构和五通道同步突变信息。

## 项目亮点

- 五通道地震属性数据：`dn / gas / gr / vp / vs`
- 两阶段 LDM 流程：`AutoencoderKL` + latent-space `UNet2DModel`
- AE 软层位约束：
  - `horizon_loss`：约束共享软层位图
  - `layer_sync_loss`：约束五通道边界同步
  - `channel_correlation_loss`：补充五通道相关性约束
- 提供训练、评估、可视化脚本
- 附带轻量实验结果和技术文档

## 目录结构

```text
.
├── ldm5/
│   ├── data.py
│   ├── models.py
│   ├── losses.py
│   ├── horizon.py
│   ├── train_autoencoder.py
│   ├── train_latent_diffusion.py
│   ├── evaluate.py
│   ├── visualize.py
│   ├── AE软层位约束改进技术文档.md
│   └── 模型原理与代码文档.md
├── docs/
│   ├── AE软层位约束改进技术文档.md
│   └── UNet多尺度层位条件注入技术文档.md
├── results/
│   └── soft_horizon_ae/
│       ├── report.json
│       ├── real_samples.png
│       ├── ae_reconstruction.png
│       └── ae_horizon_reconstruction.png
├── requirements.txt
└── README.md
```

## 环境安装

建议使用 Python 3.10+ 和 CUDA 版本 PyTorch。

```powershell
pip install -r requirements.txt
```

如果需要安装特定 CUDA 版本的 PyTorch，请优先参考 PyTorch 官方安装命令，然后再安装其余依赖。

## 数据格式

数据根目录下应包含五个通道子目录：

```text
标签/
├── dn/
├── gas/
├── gr/
├── vp/
└── vs/
```

每个通道目录中使用同名 `.npy` 文件表示同一个样本的不同物理属性：

```text
dn/000001.npy
gas/000001.npy
gr/000001.npy
vp/000001.npy
vs/000001.npy
```

训练数据和模型权重默认不提交到 Git 仓库，请放在本地或通过 Release/外部存储管理。

## 训练软层位约束 AE

```powershell
python -m ldm5.train_autoencoder `
  --data_root "标签" `
  --output_dir outputs_horizon `
  --batch_size 4 `
  --num_epochs 50 `
  --device cuda `
  --save_every 10
```

默认损失权重：

```text
lambda_gradient = 0.1
lambda_channel_corr = 0.05
lambda_frequency = 0.0
lambda_horizon = 0.2
lambda_layer_sync = 0.05
```

## 评估

```powershell
python -m ldm5.evaluate `
  --ae_dir outputs_horizon/autoencoder/final `
  --output_dir outputs_horizon `
  --batch_size 4 `
  --num_eval_batches 24 `
  --save_dir outputs_horizon/evaluation `
  --skip_ldm
```

关键指标：

- `l1_mean`
- `psnr_db_mean`
- `ssim_mean`
- `horizon_l1_mean`
- `layer_sync_l1_mean`

## 可视化

```powershell
python -m ldm5.visualize `
  --ae_dir outputs_horizon/autoencoder/final `
  --output_dir outputs_horizon `
  --save_dir outputs_horizon/evaluation `
  --num_samples 6 `
  --device cuda `
  --skip_ldm
```

会输出：

- `real_samples.png`
- `ae_reconstruction.png`
- `ae_horizon_reconstruction.png`

## 轻量结果

本仓库保留了轻量结果文件：

```text
results/soft_horizon_ae/
```

其中 `report.json` 记录了一次软层位约束 AE 评估结果，PNG 文件用于展示真实样本、AE 重构和软层位图重构对比。

## 文档

详细说明见：

- [AE软层位约束改进技术文档](docs/AE软层位约束改进技术文档.md)
- [UNet多尺度层位条件注入技术文档](docs/UNet多尺度层位条件注入技术文档.md)
- [完整 LDM5 模型原理与代码文档](ldm5/模型原理与代码文档.md)

## 注意事项

- `outputs/`、`outputs_horizon/`、`outputs_horizon_cond/` 中的大模型权重不建议提交到 Git。
- `.npy` 数据集默认不提交。
- 如需共享模型权重，建议使用 GitHub Release、Git LFS 或外部对象存储。
