import matplotlib.pyplot as plt
import numpy as np

# -------------------------- 1. 基础配置（与前4个fold完全一致） --------------------------
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 优先显示中文，无中文则用英文
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.style.use('seaborn-v0_8-whitegrid')  # 保持统一的样式

# -------------------------- 2. fold5 数据准备（已保留四位小数） --------------------------
fold5_data = {
    "1": {"train_loss": 0.3180, "train_accuracy": 89.4050, "train_IoU": 0.2628,
          "val_accuracy": 92.4039, "val_loss": 0.1976, "val_IoU": 0.3210},
    "2": {"train_loss": 0.2187, "train_accuracy": 91.7125, "train_IoU": 0.3313,
          "val_accuracy": 93.0230, "val_loss": 0.1791, "val_IoU": 0.3936},
    "3": {"train_loss": 0.1949, "train_accuracy": 92.4865, "train_IoU": 0.3632,
          "val_accuracy": 92.9995, "val_loss": 0.1732, "val_IoU": 0.3996},
    "4": {"train_loss": 0.1821, "train_accuracy": 92.9236, "train_IoU": 0.3806,
          "val_accuracy": 93.4410, "val_loss": 0.1625, "val_IoU": 0.4204},
    "5": {"train_loss": 0.1735, "train_accuracy": 93.1030, "train_IoU": 0.3918,
          "val_accuracy": 93.7152, "val_loss": 0.1519, "val_IoU": 0.4281},
    "6": {"train_loss": 0.1671, "train_accuracy": 93.3451, "train_IoU": 0.4018,
          "val_accuracy": 93.7466, "val_loss": 0.1558, "val_IoU": 0.4088},
    "7": {"train_loss": 0.1612, "train_accuracy": 93.4870, "train_IoU": 0.4124,
          "val_accuracy": 93.6813, "val_loss": 0.1495, "val_IoU": 0.4449},
    "8": {"train_loss": 0.1581, "train_accuracy": 93.5976, "train_IoU": 0.4168,
          "val_accuracy": 93.9425, "val_loss": 0.1494, "val_IoU": 0.4290},
    "9": {"train_loss": 0.1523, "train_accuracy": 93.7430, "train_IoU": 0.4242,
          "val_accuracy": 93.7884, "val_loss": 0.1481, "val_IoU": 0.4476},
    "10": {"train_loss": 0.1506, "train_accuracy": 93.8466, "train_IoU": 0.4273,
           "val_accuracy": 94.0731, "val_loss": 0.1419, "val_IoU": 0.4364}
}

# 提取epochs和各项指标数据
epochs = list(map(int, fold5_data.keys()))
train_loss = [fold5_data[str(e)]["train_loss"] for e in epochs]
val_loss = [fold5_data[str(e)]["val_loss"] for e in epochs]
train_acc = [fold5_data[str(e)]["train_accuracy"] for e in epochs]
val_acc = [fold5_data[str(e)]["val_accuracy"] for e in epochs]
train_iou = [fold5_data[str(e)]["train_IoU"] for e in epochs]
val_iou = [fold5_data[str(e)]["val_IoU"] for e in epochs]

# -------------------------- 3. 绘制折线图（1行3列横向布局，样式完全统一） --------------------------
# 创建1行3列的子图，画布尺寸和前四个fold保持一致
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Fold5 Training and Validation Metrics (1-10 Epochs)', fontsize=18, fontweight='bold', y=0.98)

# 子图1：Loss曲线（配色、样式与前4个fold一致）
ax1.plot(epochs, train_loss, 'b-o', label='Train Loss', linewidth=2, markersize=5, alpha=0.8)
ax1.plot(epochs, val_loss, 'r-s', label='Val Loss', linewidth=2, markersize=5, alpha=0.8)
ax1.set_title('Loss', fontsize=14, fontweight='bold')
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Loss Value', fontsize=12)
ax1.set_xticks(epochs)
ax1.legend(loc='upper right', fontsize=10)
ax1.grid(True, alpha=0.3)

# 子图2：Accuracy曲线（配色、样式与前4个fold一致）
ax2.plot(epochs, train_acc, 'b-o', label='Train Accuracy', linewidth=2, markersize=5, alpha=0.8)
ax2.plot(epochs, val_acc, 'r-s', label='Val Accuracy', linewidth=2, markersize=5, alpha=0.8)
ax2.set_title('Accuracy', fontsize=14, fontweight='bold')
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Accuracy (%)', fontsize=12)
ax2.set_xticks(epochs)
ax2.legend(loc='lower right', fontsize=10)
ax2.grid(True, alpha=0.3)

# 子图3：IoU曲线（配色、样式与前4个fold一致）
ax3.plot(epochs, train_iou, 'b-o', label='Train IoU', linewidth=2, markersize=5, alpha=0.8)
ax3.plot(epochs, val_iou, 'r-s', label='Val IoU', linewidth=2, markersize=5, alpha=0.8)
ax3.set_title('IoU', fontsize=14, fontweight='bold')
ax3.set_xlabel('Epoch', fontsize=12)
ax3.set_ylabel('IoU Value', fontsize=12)
ax3.set_xticks(epochs)
ax3.legend(loc='lower right', fontsize=10)
ax3.grid(True, alpha=0.3)

# -------------------------- 4. 保存和显示图片 --------------------------
# 调整子图间距，避免标题/标签重叠
plt.tight_layout(rect=[0, 0, 1, 0.96])

# 保存高清图片（300dpi），文件名标注fold5区分
plt.savefig('fold5_training_metrics_horizontal.png', dpi=300, bbox_inches='tight')
print("fold5横向布局图片已保存为: fold5_training_metrics_horizontal.png")

# 显示图片
plt.show()