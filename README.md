 # 实验五：多模态情感分类（最终代码已master分支中为准）  

## 1. 项目概览

本项目旨在解决文本和图像的多模态情感分类任务。通过对模型架构和训练策略进行一系列的 **SOTA 级优化**，我们将模型的验证集准确率从基线的 $\approx 68.50\%$ 提升到了 **$75.75\%$**。

### 核心亮点

1.  **深度融合架构：** 采用 2 层 **Transformer 交叉注意力** 模块实现序列级的双向深度语义对齐。
2.  **骨干网络升级：** 将编码器升级至 **RoBERTa-wwm-ext** (Text) + **ResNet50** (Image)，保障了高质量的特征输入。
3.  **SOTA 训练策略：** 引入 **分层学习率** ($1\text{e}-5$ vs $5\text{e}-4$)、**Warmup 调度器** 和 **强正则化**（Dropout, Label Smoothing），解决了高参数量模型在小数据集上的过拟合和训练稳定性问题。

## 2. 环境配置与依赖

本项目基于 PyTorch 实现，并依赖 HuggingFace Transformers 库。请确保您的环境安装了 NVIDIA 驱动和 CUDA。

### 2.1 依赖安装

请使用以下命令安装所有依赖。所有环境信息已记录在 `requirements.txt` 中。

```bash
pip install -r requirements.txt
```

### 2.2 硬件要求

*   **GPU 显存：** 至少 **8GB VRAM**（本项目在Batch Size=16时测试通过，更大的Batch Size需要更多显存）。

## 3. 代码文件结构

```
master
├── models/
│   └── base_models.py       # 【核心】所有模型（Text/Image/EarlyFusion/TransformerFusion）定义
├── scripts/
│   ├── data_loader_fusion.py  # 【核心】RoBERTa/Fusion模型的序列数据加载器
│   ├── data_loader.py  # 【核心】基线模型数据加载器
│   ├── train.py  # 【核心】基线模型训练与评估入口
│   └── train_fusion.py        # 【核心】RoBERTa/Fusion模型训练与评估，最终结果预测入口 
├── data/                    # 原始图片和文本数据（本地存储，Git 已忽略）
├── train.txt
├── .gitignore      # 上传要求
├── test_without_label.txt
├── requirements.txt
├── shiyanjieguo.txt  # 最终预测结果文件
└── README.md                
```

## 4. 执行你代码的完整流程

### 步骤 1：数据准备

1.  将 `实验五数据.zip` **解压**。
2.  将所有 **.jpg** 和 **.txt** 文件放入项目根目录下的 **`data/`** 文件夹中。
3.  确保 `train.txt` 和 `test_without_label.txt` 文件位于项目根目录。

### 步骤 2：运行训练、评估与预测

运行以下脚本即可完成所有任务：

*   运行所有**基线模型**（Text, Image, Early Fusion）的训练和验证。由train.py进行
*   运行**最终优化模型**（RoBERTa+ResNet50）的训练，并自动加载最佳权重。由train_fusion.py进行
*   对 `test_without_label.txt` 进行预测，并生成结果文件。

```bash
#基线模型结果运行（需取消train.py中的测试代码注释）
python -m scripts.train
# 融合模型最终结果运行
python -m scripts.train_fusion
```

### 步骤 3：查看结果

*   **训练输出：** 脚本会实时打印所有模型的 Val Acc。
*   **最佳模型保存：** 最佳模型权重保存在 `models/best_roberta_fusion.pth`。
*   **最终提交结果：** 最终的预测结果文件将生成在：`shiyanjieguo.txt`

## 5. 参考文献与技术实现

本实验的架构和策略参考了以下 SOTA 深度学习实践，以达到性能突破：

| 技术点 | 对应代码 / 作用 | 参考文献或出处 |
| :--- | :--- | :--- |
| **文本编码器** | `hfl/chinese-roberta-wwm-ext` | RoBERTa: A robustly optimized BERT pretraining approach. |
| **图像编码器** | `models.resnet50` | Deep residual learning for image recognition. |
| **交叉注意力** | `CrossAttentionBlock` / `FusionEncoder` | Vision-and-Language Transformer (ViLT, UNITER) 架构。 |
| **分层学习率** | `torch.optim.AdamW` + `get_linear_schedule_with_warmup` | HuggingFace Transformers 库及其官方 SOTA 模型训练策略。 |
| **正则化** | `nn.Dropout(0.5)` + `nn.CrossEntropyLoss(label_smoothing=0.1)` | 通用深度学习优化实践，用于对抗高参数量模型的过拟合。 |
```
