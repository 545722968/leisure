import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import os


#  数据预处理
# fer2013数据集格式：pixels列为逗号分隔的48x48灰度像素值，emotion为标签
df_fer2013 = pd.read_csv("fer2013.csv")
df_ferplus = pd.read_csv("fer2013_plus.csv")

# 定义有效表情列与fer2013标签编码的映射
emotion_columns = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear']
emotion_mapping = {
    'neutral': 6,
    'happiness': 3,
    'surprise': 5,
    'sadness': 4,
    'anger': 0,
    'disgust': 1,
    'fear': 2
}


def generate_label(row):
    """根据投票规则生成标签"""
    # 提取无效标记的票数
    unknown_votes = row['unknown']
    nf_votes = row['NF']

    # 提取有效表情票数
    effective_votes = row[emotion_columns].values.astype(int)

    # 计算全局最大票数（包括unknown和NF）
    all_votes = np.concatenate([effective_votes, [unknown_votes, nf_votes]])
    max_all_votes = np.max(all_votes)

    # 条件1：如果unknown或NF是最高票，直接无效
    if unknown_votes == max_all_votes or nf_votes == max_all_votes:
        return None

    # 条件2：有效表情票数总和为0（理论上不会发生，但保留检查）
    if np.sum(effective_votes) == 0:
        return None

    # 条件3：有效表情中存在并列最高票
    max_effective_votes = np.max(effective_votes)
    max_indices = np.where(effective_votes == max_effective_votes)[0]
    if len(max_indices) > 1:
        return None

    # 返回有效标签
    emotion = emotion_columns[max_indices[0]]
    return emotion_mapping[emotion]


# 生成新标签列
df_ferplus['emotion_new'] = df_ferplus.apply(generate_label, axis=1)

# 合并数据集（假设行号对齐）
df_combined = pd.DataFrame({
    'pixels': df_fer2013['pixels'],
    'Usage': df_ferplus['Usage'],
    'emotion_original': df_fer2013['emotion'],
    'emotion_new': df_ferplus['emotion_new']
})

# 过滤无效数据（保留有效标签）
df_combined = df_combined.dropna(subset=['emotion_new']).reset_index(drop=True)

# 保存最终数据集（仅保留新标签）
df_combined[['pixels', 'Usage', 'emotion_new']].rename(
    columns={'emotion_new': 'emotion'}
).to_csv("fer2013_plus_final.csv", index=False)

# 数据清洗统计
print(f"原始样本数: {len(df_ferplus)}")
print(f"有效样本数: {len(df_combined)}")
print(f"过滤比例: {1 - len(df_combined)/len(df_ferplus):.2%}")

# 输出标签分布对比
print("\n原始标签分布:")
print(df_fer2013['emotion'].value_counts().sort_index())
print("\n清洗后标签分布:")
print(df_combined['emotion_new'].value_counts().sort_index())
# 标签分布对比
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
# 原始分布
df_fer2013['emotion'].value_counts().sort_index().plot(
    kind='bar', ax=ax1, title='Original Labels'
)
# 清洗后分布
df_combined['emotion_new'].value_counts().sort_index().plot(
    kind='bar', ax=ax2, title='Cleaned Labels', color='orange'
)
plt.show()


# 配置GPU显存自动增长
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# 常量定义
IMG_SIZE = 224
BATCH_SIZE = 16  # 减小批量大小以节省内存
EPOCHS_INITIAL = 30
EPOCHS_FINE_TUNE = 20


# --------------------------------------------------
# 1. 数据加载与预处理（生成器模式）
# --------------------------------------------------
def dataframe_generator(df, batch_size, augment=False):
    """生成器返回 (images, labels, sample_weights) """
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    ) if augment else ImageDataGenerator()

    while True:
        for i in range(0, len(df), batch_size):
            batch_df = df.iloc[i:i + batch_size]
            images = []
            labels = []
            sample_weights = []
            for _, row in batch_df.iterrows():
                # 解析像素
                img = np.fromstring(row['pixels'], sep=' ', dtype=np.uint8).reshape(48, 48)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)
                img = np.stack([img] * 3, axis=-1).astype(np.float32)
                img = (img / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
                images.append(img)
                labels.append(row['emotion'])
                sample_weights.append(row['sample_weight'])

            images = np.array(images)
            labels = np.array(labels)
            sample_weights = np.array(sample_weights)

            if augment:
                # 数据增强（确保增强后的数据与权重对应）
                gen = datagen.flow(images, labels, sample_weight=sample_weights, batch_size=batch_size)
                aug_images, aug_labels, aug_weights = next(gen)
                yield aug_images, aug_labels, aug_weights
            else:
                yield images, labels, sample_weights

# 加载元数据（仅读取文件名和标签）
df = pd.read_csv("fer2013_plus_final.csv")

# 划分训练集和验证集
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['emotion'], random_state=42)

# 创建生成器
train_gen = dataframe_generator(train_df, BATCH_SIZE, augment=True)
val_gen = dataframe_generator(val_df, BATCH_SIZE, augment=False)

# --------------------------------------------------
# 2. 数据增强
# --------------------------------------------------
train_datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

# 将生成器与数据增强结合
def augmented_generator(generator):
    for x, y in generator:
        yield train_datagen.flow(x, y, batch_size=BATCH_SIZE).next()

train_aug_gen = augmented_generator(train_gen)



# --------------------------------------------------
# 3. 构建VGG模型
# --------------------------------------------------
def build_vgg_model():
    base_model = VGG16(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    base_model.trainable = False  # 初始阶段冻结基础模型

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu',
                     kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        layers.Dropout(0.5),
        layers.Dense(7, activation='softmax')
    ])
    return model


model = build_vgg_model()

# --------------------------------------------------
# 4. 处理类别不平衡 类别权重计算
# --------------------------------------------------
# 计算类别权重
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_df['emotion']),
    y=train_df['emotion']
)
class_weights_dict = dict(enumerate(class_weights))

# 为训练集和验证集添加样本权重列
train_df['sample_weight'] = train_df['emotion'].map(class_weights_dict)
val_df['sample_weight'] = val_df['emotion'].map(class_weights_dict)

# --------------------------------------------------
# 5. 训练配置
# --------------------------------------------------
# 初始训练（冻结基础模型）
model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 回调函数
early_stop = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=8,
    restore_best_weights=True
)
reduce_lr = callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-6
)

# 第一阶段训练
print("\n=== 初始训练阶段 ===")
# 使用生成器进行训练（自动从生成器获取样本权重）
history_initial = model.fit(
    train_gen,
    steps_per_epoch=len(train_df) // BATCH_SIZE,
    epochs=EPOCHS_INITIAL,
    validation_data=val_gen,
    validation_steps=len(val_df) // BATCH_SIZE,
    callbacks=[early_stop, reduce_lr]
)
# --------------------------------------------------
# 6. 微调模型
# --------------------------------------------------
# 解冻部分卷积层
base_model = model.layers[0]
base_model.trainable = True
for layer in base_model.layers[:-4]:  # 解冻最后4层
    layer.trainable = False

# 重新编译（更低学习率）
model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 微调训练
print("\n=== 微调训练阶段 ===")
history_fine = model.fit(
    train_aug_gen,
    steps_per_epoch=len(train_df) // BATCH_SIZE,    # 计算每epoch步数
    epochs=EPOCHS_FINE_TUNE,
    validation_data=val_gen,
    validation_steps=len(val_df) // BATCH_SIZE,  # 验证步数
    class_weight=class_weights,
    callbacks=[early_stop, reduce_lr]
)

# --------------------------------------------------
# 7. 保存模型
# --------------------------------------------------
model.save('fer_vgg_final.h5')  # H5格式
# model.save('fer_vgg_final')  # SavedModel格式


# --------------------------------------------------
# 8. 可视化评估
# --------------------------------------------------
def plot_history(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Accuracy Curve')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Loss Curve')
    plt.legend()
    plt.show()


# 绘制训练曲线
print("\n训练曲线:")
plot_history(history_initial)
plot_history(history_fine)

# 评估需使用生成器生成预测结果
val_gen_eval = dataframe_generator(val_df, BATCH_SIZE, augment=False)
y_pred = model.predict(val_gen_eval, steps=len(val_df)//BATCH_SIZE + 1)
y_pred_classes = np.argmax(y_pred[:len(val_df)], axis=1)  # 截断多余预测

# 混淆矩阵与分类报告
from sklearn.metrics import confusion_matrix, classification_report
print("\n分类报告:")
print(classification_report(val_df['emotion'].values[:len(y_pred_classes)], y_pred_classes))
