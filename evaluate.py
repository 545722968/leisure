import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from sklearn.preprocessing import LabelBinarizer

# 1. 数据加载与预处理
df = pd.read_csv('fer2013_plus_final.csv')


# 预处理函数（与训练时相同）
def preprocess_pixels(pixel_str):
    pixel_list = list(map(int, pixel_str.split()))
    return np.array(pixel_list).reshape(48, 48, 1)


X = np.array([preprocess_pixels(pixels) for pixels in df['pixels']])
X = X / 255.0  # 保持相同的归一化方式

# 2. 标签处理
lb = LabelBinarizer()
y = lb.fit_transform(df['emotion'])

# 在评估代码中添加情绪标签映射（根据FER2013的标准标签顺序）
emotion_labels = {
    0: 'angry',
    1: 'disgust',
    2: 'fear',
    3: 'happy',
    4: 'sad',
    5: 'surprise',
    6: 'neutral'
}

# 3. 数据集划分（保持与训练时相同的划分逻辑）
X_test = X[df['Usage'] == 'PrivateTest']
y_test = y[df['Usage'] == 'PrivateTest']

# 4. 加载预训练模型
model = load_model('vgg_model.h5')

# 5. 基础评估
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f'[基础指标] 测试集准确率: {test_acc:.4f}, 损失值: {test_loss:.4f}')

# 6. 详细评估
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)

# 分类报告
print("\n[分类报告]")
print(classification_report(
    y_test_classes,
    y_pred_classes,
    target_names=list(emotion_labels.values())  # 使用自定义的字符串标签
))
# 混淆矩阵可视化
plt.figure(figsize=(12, 10))
cm = confusion_matrix(y_test_classes, y_pred_classes)
sns.heatmap(cm, annot=True, fmt='d',
            cmap='Blues',
            xticklabels=emotion_labels.values(),  # 使用自定义标签
            yticklabels=emotion_labels.values())  # 使用自定义标签
plt.xlabel('Prediction labels')
plt.ylabel('Real labels')
plt.title('Confusion matrix')
plt.show()

# 7. 错误样本分析
error_indices = np.where(y_pred_classes != y_test_classes)[0]
if len(error_indices) > 0:
    print(f"\n[错误分析] 共发现 {len(error_indices)} 个错误样本")

    # 可视化随机9个错误样本
    sample_idx = np.random.choice(error_indices, 9)
    plt.figure(figsize=(10, 10))
    for i, idx in enumerate(sample_idx):
        plt.subplot(3, 3, i + 1)
        plt.imshow(X_test[idx].reshape(48, 48), cmap='gray')
        plt.title(f'True: {lb.classes_[y_test_classes[idx]]}\nPrediction: {lb.classes_[y_pred_classes[idx]]}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()
else:
    print("\n[完美预测] 所有测试样本均预测正确！")