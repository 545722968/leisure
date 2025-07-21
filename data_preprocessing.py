import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def load_and_preprocess_data():
    # 数据增强
    datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=25,  # 增大旋转范围
        width_shift_range=0.2,  # 增大平移幅度
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        brightness_range=[0.7, 1.3],  # 亮度调整
        channel_shift_range=50,  # 通道偏移增强
        horizontal_flip=True,
        fill_mode='nearest'
    )
    # 加载原始数据
    df = pd.read_csv('fer2013.csv')

    # 数据解析
    pixels = df['pixels'].apply(lambda x: np.fromstring(x, sep=' ', dtype=np.uint8))
    X = np.vstack(pixels.values)
    y = df['emotion'].values

    # 调整维度 (48x48 灰度图)
    X = X.reshape(-1, 48, 48, 1)

    # 数据拆分
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    return datagen, X_train, X_test, y_train, y_test

def analyze_data_distribution(y_train, y_test):
    # 统计类别分布
    train_counts = np.bincount(y_train)
    test_counts = np.bincount(y_test)
    classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

    # 绘制分布图
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.bar(classes, train_counts)
    plt.title('Training Set Distribution')
    plt.xticks(rotation=45)

    plt.subplot(1, 2, 2)
    plt.bar(classes, test_counts)
    plt.title('Test Set Distribution')
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig('data_distribution.png')
    plt.show()
