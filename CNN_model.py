import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.layers import BatchNormalization, GlobalAveragePooling2D
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 加载数据
df = pd.read_csv('fer2013_plus_final.csv')

# 预处理像素数据
def preprocess_pixels(pixel_str):
    pixel_list = list(map(int, pixel_str.split()))
    return np.array(pixel_list).reshape(48, 48, 1)  # 假设图像尺寸为48x48

X = np.array([preprocess_pixels(pixels) for pixels in df['pixels']])
X = X / 255.0  # 归一化

# 处理标签
lb = LabelBinarizer()
y = lb.fit_transform(df['emotion'])

# 根据Usage划分数据集
X_train = X[df['Usage'] == 'Training']
X_val = X[df['Usage'] == 'PublicTest']
X_test = X[df['Usage'] == 'PrivateTest']

y_train = y[df['Usage'] == 'Training']
y_val = y[df['Usage'] == 'PublicTest']
y_test = y[df['Usage'] == 'PrivateTest']

# 数据增强
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)
datagen.fit(X_train)

# 构建CNN模型
model = Sequential([
        # Block 1
        Conv2D(64, (3, 3), activation='relu', input_shape=(48, 48, 1)),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        # Block 2
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.35),

        # Block 3
        Conv2D(256, (3, 3), activation='relu'),
        BatchNormalization(),
        GlobalAveragePooling2D(),
        Dropout(0.5),

        Dense(512, activation='relu'),
        Dense(7, activation='softmax')
    ])

# 编译模型
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 训练模型
early_stop = EarlyStopping(monitor='val_loss', patience=5)
history = model.fit(datagen.flow(X_train, y_train, batch_size=64),
                    epochs=50,
                    validation_data=(X_val, y_val),
                    callbacks=[early_stop])

# 保存模型
model.save('fer2013_plus_CNNmodel2.h5')

# 评估模型
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f'Test accuracy: {test_acc:.4f}')

# 绘制训练曲线
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Accuracy Curves')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss Curves')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
