import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

BASE_DIR = 'accessories_data'    # папка, где лежат Augmented и Original
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 50                      # ← теперь 50, но EarlyStopping остановит раньше
NUM_CLASSES = 10

# ===================== ГЕНЕРАТОРЫ =====================
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    validation_split=0.3
)

train_generator = datagen.flow_from_directory(
    BASE_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_generator = datagen.flow_from_directory(
    BASE_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

test_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.15)
test_generator = test_datagen.flow_from_directory(
    BASE_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

class_names = list(train_generator.class_indices.keys())
print("Классы:", class_names)
print(f"Обучающих изображений: {train_generator.samples}")
print(f"Валидационных: {val_generator.samples}")

# ===================== МОДЕЛЬ =====================
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(256, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(512, activation='relu'),
    layers.Dense(NUM_CLASSES, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ===================== КОЛБЭКИ =====================
checkpoint = ModelCheckpoint(
    'models/best_accessories.h5',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=8,                # ждём 8 эпох без улучшения
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_accuracy',
    factor=0.5,                # уменьшаем скорость обучения вдвое
    patience=4,                # если 4 эпохи без улучшения
    verbose=1,
    min_lr=1e-6
)

# ===================== ОБУЧЕНИЕ =====================
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=val_generator,
    validation_steps=val_generator.samples // BATCH_SIZE,
    callbacks=[checkpoint, early_stop, reduce_lr]
)

# ===================== СОХРАНЕНИЕ =====================
model.save('models/final_accessories.h5')

# ===================== ОЦЕНКА НА ТЕСТЕ =====================
test_loss, test_acc = model.evaluate(test_generator, steps=test_generator.samples // BATCH_SIZE)
print(f"Точность на тестовой выборке: {test_acc:.4f}")

# ===================== ГРАФИКИ =====================
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Точность')
plt.xlabel('Эпоха')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Потери')
plt.xlabel('Эпоха')
plt.legend()

plt.savefig('training_accessories.png')
plt.show()

# ===================== СПИСОК КЛАССОВ =====================
with open('models/class_names_accessories.txt', 'w') as f:
    for name in class_names:
        f.write(name + '\n')

print("✅ Модель готова! Файлы сохранены в папке models/")