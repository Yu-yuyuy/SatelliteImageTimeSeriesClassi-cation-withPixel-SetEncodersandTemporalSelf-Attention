# PSE-TAE 遥感影像时空分类模型复现

本仓库为课程作业项目，围绕 CVPR 2020 论文 **Satellite Image Time Series Classification with Pixel-Set Encoders and Temporal Self-Attention** 进行代码复现、训练验证与结果可视化。项目基于 PSE（Pixel-Set Encoder）与 TAE（Temporal Attention Encoder）架构，面向 Sentinel-2 卫星影像时间序列数据开展地物分类实验，重点验证该模型在遥感影像时空分类任务中的有效性、稳定性与可复现性。
## 小组成员信息

| 角色 | 姓名 | 学号 |
| --- | --- | --- |
| 组长 | 王少艺 | 2025303120175 |
| 组员 | 余娟 | 2025303110117 |
| 组员 | 丁红俊 | 2025303110115 |
* 论文名称：Satellite Image Time Series Classification with Pixel-Set Encoders and Temporal Self-Attention
* 论文会议：CVPR 2020 Oral
* 论文 DOI：10.1109/CVPR42600.2020.01233
* 论文链接：https://openaccess.thecvf.com/content_CVPR_2020/html/Garnot_Satellite_Image_Time_Series_Classification_With_Pixel-Set_Encoders_and_Temporal_CVPR_2020_paper.html
* 原始代码仓库：https://github.com/VSainteuf/pytorch-psetae

> 说明：本仓库是在原始开源项目基础上的课程复现版本，主要补充了环境配置、训练过程验证、推理结果输出、指标统计和可视化分析等内容。

---

## 1. 项目简介

遥感影像时间序列数据能够反映地物在不同时间节点上的光谱变化特征，是农业地块识别、土地覆盖分类和地表动态监测中的重要数据来源。传统卷积神经网络通常依赖规则影像块进行空间特征提取，而 Sentinel-2 等中分辨率遥感影像中，同一地块内部像元数量有限且空间排列不规则。PSE-TAE 模型通过 Pixel-Set Encoder 对无序像元集合进行空间特征编码，并利用 Temporal Attention Encoder 捕捉多时相观测之间的时间依赖关系，从而实现高效的遥感影像时空分类。

本复现项目主要完成以下工作：

1. 搭建 PSE-TAE 模型运行环境；
2. 准备 Sentinel-2 Pixel-Set 数据集；
3. 执行 5 折交叉验证训练；
4. 保存模型权重、训练日志、测试指标和混淆矩阵；
5. 基于训练日志与测试结果制作可视化图表；
6. 总结复现过程中遇到的依赖兼容与硬件适配问题。

---

## 2. 代码核心功能

本项目面向 Sentinel-2 遥感影像数据集的多类别地物分类任务，核心流程由 `train.py` 和 `inference.py` 两个主脚本构成。

### 2.1 模型结构

模型主要包括三个部分：

* **Pixel-Set Encoder, PSE**：对每个地块内部的像素集合进行编码，提取空间维度上的统计与聚合特征；
* **Temporal Attention Encoder, TAE**：基于自注意力机制建模多时相遥感观测之间的时间依赖关系；
* **MLP Classifier**：将时空特征映射到最终地物类别，实现多类别分类。

模型整体结构如下：

![PSE-TAE architecture](./graphics/PSETAE.png)

### 2.2 训练与推理流程

`train.py` 用于模型训练与验证，主要功能包括：

* 读取 Pixel-Set 数据集；
* 按照指定类别筛选样本；
* 执行 5 折交叉验证；
* 记录每轮训练损失、准确率和 IoU；
* 保存最优模型权重；
* 输出测试指标与混淆矩阵。

`inference.py` 用于模型推理，主要功能包括：

* 加载训练得到的模型权重；
* 对全量数据集进行批量预测；
* 输出样本 ID、真实标签和预测标签；
* 为后续分类结果分析和可视化提供数据支持。

---

## 3. 环境配置

本项目复现过程中主要使用 Python、PyTorch、torchnet、scikit-learn 等依赖库。由于原始项目发布时间较早，部分依赖与新版 PyTorch 存在兼容问题，因此建议按照固定版本配置环境。

### 3.1 推荐环境

```text
Python == 3.6.6
PyTorch == 1.1.0
CUDA == 10.0
torchnet
numpy
pandas
scikit-learn
tqdm
```

### 3.2 安装依赖

可以使用以下命令安装依赖：

```bash
pip install -r requirements.txt
```

如果使用 UV 管理环境，可根据 `requirements.txt` 创建一致的运行环境。

---

## 4. 数据集说明

本项目使用原论文提供的 Pixel-Set 数据集。由于完整数据集体积较大，仓库中不直接包含 `data/` 文件夹。请用户根据需要自行下载数据，并放置到项目根目录下。

### 4.1 数据下载

原论文数据集下载地址如下：

* Pixel-Set dataset：https://zenodo.org/record/5815488
* Pixel-Patch dataset：https://zenodo.org/record/5815523
* Toy dataset：http://recherche.ign.fr/llandrieu/TAE/S2-2017-T31TFM-PixelSet-TOY.zip

如需使用本课程复现实验所用数据，请从以下链接下载：

```text
数据下载链接：请在此处填写网盘或服务器链接
提取码：如有，请填写
```

下载完成后，请将数据解压到项目根目录下的 `data/` 文件夹中。

### 4.2 数据组织结构

数据文件夹建议组织如下：

```text
data/
├── normalisation_values.pkl
├── DATA/
│   ├── sample0.npy
│   ├── sample1.npy
│   └── ...
└── META/
    ├── labels.json
    ├── dates.json
    └── geomfeat.json
```

其中：

* `DATA/` 存储每个样本对应的 `.npy` 文件；
* `META/labels.json` 存储样本标签；
* `META/dates.json` 存储多时相观测日期；
* `META/geomfeat.json` 存储可选的几何或静态特征；
* `normalisation_values.pkl` 存储归一化所需的均值和标准差。

---

## 5. 仓库结构

本仓库主要文件结构如下：

```text
.
├── dataset.py              # 数据集读取与预处理
├── train.py                # 模型训练与交叉验证
├── inference.py            # 模型推理
├── requirements.txt        # 项目依赖
├── models/                 # PSE、TAE 和 PSE-TAE 模型结构
├── learning/               # 训练、损失函数和评价指标相关工具
├── preprocessing/          # 数据预处理脚本
├── graphics/               # 模型结构图及可视化图像
├── vis_resluts/            # 复现结果可视化图表
├── .gitignore              # 忽略数据集、模型权重和训练输出等大文件
└── README.md               # 项目说明文档
```

---

## 6. 运行方法

### 6.1 训练模型

运行 `train.py` 进行模型训练：

```bash
python train.py --dataset_folder ./data
```

训练脚本将自动执行 5 折交叉验证，并在对应输出目录中保存训练日志、模型权重、测试指标和混淆矩阵。

### 6.2 模型推理

训练完成后，可以使用 `inference.py` 进行推理：

```bash
python inference.py --dataset_folder ./data --weight_dir ./results/Fold_1
```

其中，`--weight_dir` 为已训练模型权重所在文件夹。

### 6.3 结果输出

训练和推理完成后，主要输出文件包括：

```text
results/
├── Fold_1/
│   ├── trainlog.json
│   ├── test_metrics.json
│   ├── confusion_matrix.npy
│   └── model.pth.tar
├── Fold_2/
├── Fold_3/
├── Fold_4/
└── Fold_5/
```

---

## 7. 复现结果与可视化

本项目基于模型训练输出的核心数据文件制作了两类可视化图表，分别从训练过程和最终测试性能两个方面展示模型复现效果。

### 7.1 单折训练过程可视化

基于每折 `trainlog.json` 中记录的 epoch 级数据，分别绘制训练损失、分类准确率和 IoU 随训练轮次变化的曲线，用于观察模型收敛过程和训练稳定性。

<img width="2556" height="878" alt="Fold training curve 1" src="https://github.com/user-attachments/assets/934e0843-b072-4a04-9623-de9c376b3015" />

<img width="2556" height="873" alt="Fold training curve 2" src="https://github.com/user-attachments/assets/31f6c3f2-368c-4bf3-9558-c053a53a9bed" />

<img width="1278" height="439" alt="Fold training curve 3" src="https://github.com/user-attachments/assets/0dbcb4f4-9236-45cb-9d64-708d251326a5c56" />

<img width="2553" height="870" alt="Fold training curve 4" src="https://github.com/user-attachments/assets/fe60feb7-8f33-4b24-b233-30ae467d5491" />

<img width="2556" height="870" alt="Fold training curve 5" src="https://github.com/user-attachments/assets/26b9a658-4f46-4b8c-9b3c-78606f55ca4b" />

### 7.2 5 折测试指标汇总

基于各折 `test_metrics.json` 文件，统计模型在独立测试集上的 Accuracy、Loss 和 mIoU 指标。实验结果表明，模型在不同数据划分下具有较好的稳定性。

| 指标            | 结果范围            | 说明             |
| ------------- | --------------- | -------------- |
| Test Accuracy | 93.66% - 94.11% | 反映模型整体分类正确率    |
| Test Loss     | 0.1426 - 0.1511 | 反映模型在测试集上的损失水平 |
| Test mIoU     | 0.4342 - 0.4508 | 反映不同类别空间分类稳定性  |

<img width="2331" height="606" alt="5-fold test metrics" src="https://github.com/user-attachments/assets/bd458c97-9e1d-4690-801f-d00aafa15b6c" />

---

## 8. 小组完成的主要工作

本课程复现项目围绕“模型可运行、结果可追溯、图表可验证”的目标开展，主要完成了以下工作：

1. **项目克隆与环境标准化配置**
   基于原始开源仓库搭建运行环境，梳理 PyTorch、torchnet、scikit-learn、tqdm 等核心依赖版本，并形成可复现的环境配置方案。

2. **数据准备与训练验证**
   下载并整理 Sentinel-2 Pixel-Set 数据集，按照代码中的 `sub_classes` 参数筛选 20 类地物标签，运行 `train.py` 执行 5 折交叉验证训练，并保存训练日志、模型权重、测试指标和混淆矩阵。

3. **模型推理与结果输出**
   运行 `inference.py` 加载训练权重，对数据集进行批量推理，输出样本 ID、真实标签和预测标签，为后续结果分析提供支撑。

4. **结果可视化与复现验证**
   基于 `trainlog.json` 和 `test_metrics.json` 绘制训练过程曲线和测试指标汇总图，验证模型训练过程、测试性能和可视化结果之间的一致性。

---

## 9. 复现过程中遇到的问题及解决方法

### 9.1 torchnet 与新版 PyTorch 兼容问题

原始项目依赖的 torchnet 库与新版 PyTorch 存在部分接口不兼容问题，导致 `ClassErrorMeter` 等指标计算模块在训练过程中报错。复现过程中通过调整 PyTorch 版本并适配相关调用逻辑，保证训练指标能够正常统计。

### 9.2 CUDA 算力匹配问题

在不同设备上运行模型时，可能出现 CUDA 版本与 PyTorch 预编译版本不匹配的问题，例如：

```text
CUDA error: no kernel image is available for execution on the device
```

该问题可通过检查 GPU 算力、CUDA 版本和 PyTorch 版本之间的对应关系解决。建议按照推荐环境配置运行，避免由于硬件适配问题导致训练失败。

### 9.3 依赖版本精准性问题

部分 Python 包的新版接口与原始代码不完全兼容，例如 scikit-learn、tqdm 等。因此，复现过程中建议锁定依赖版本，并通过 `requirements.txt` 统一管理环境，减少不同设备之间的运行差异。

---

## 10. 注意事项

1. 本仓库不上传完整 `data/` 数据集，用户需要自行下载并放置到项目根目录下；
2. `results/`、`output/`、`runs/` 等训练输出文件夹默认不纳入 Git 管理；
3. `.pt`、`.pth`、`.npy`、`.npz`、`.pkl`、`.zip` 等大文件默认被 `.gitignore` 忽略；
4. 如需复现实验结果，请确保数据结构、依赖版本和运行参数与本文档保持一致。

---

## 11. 致谢与引用

本项目基于以下论文和开源仓库完成复现：

```bibtex
@inproceedings{garnot2020satellite,
  title={Satellite Image Time Series Classification With Pixel-Set Encoders and Temporal Self-Attention},
  author={Garnot, Vivien Sainte Fare and Landrieu, Loic and Giordano, Sebastien and Chehata, Nesrine},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={12325--12334},
  year={2020}
}
```

原始代码仓库：

```text
https://github.com/VSainteuf/pytorch-psetae
```

本仓库仅用于课程学习、论文复现与实验验证。

