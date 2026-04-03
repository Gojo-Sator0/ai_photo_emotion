"""
Навчання моделі на датасеті з Kaggle
"""
import os
import numpy as np
import cv2
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def create_better_model():
    """Покращена модель для реальних результатів"""
    model = models.Sequential([
        # Conv Block 1
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(48, 48, 1)),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Conv Block 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Conv Block 3
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Dense layers
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(7, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_on_fer2013():
    """Навчання на FER-2013 датасеті"""
    
    print("="*60)
    print("🎓 ІНСТРУКЦІЯ ДЛЯ ЗАВАНТАЖЕННЯ ДАТАСЕТУ FER-2013")
    print("="*60)
    
    print("""
1️⃣  Перейдіть на: https://www.kaggle.com/datasets/msambare/fer2013

2️⃣  Натисніть "Download" (потребує Kaggle акаунту)

3️⃣  Розпакуйте архів у папку: D:\\Project\\AI photo emotion\\data\\
    
    Структура повинна бути:
    data/
    ├── train/
    │   ├── angry/
    │   ├── disgust/
    │   ├── fear/
    │   ├── happy/
    │   ├── sad/
    │   ├── surprise/
    │   └── neutral/
    └── test/
        └── (аналогічно)

4️⃣  Запустіть: python train_from_kaggle.py --train

5️⃣  Після завершення запустіть: python main.py
    """)
    
    # Перевірка наявності датасету
    train_path = 'data/train'
    
    if not os.path.exists(train_path):
        print("❌ Датасет не знайдено!")
        print("📂 Очікував папку: data/train/")
        return False
    
    print("\n✅ Датасет знайдено! Початок навчання...\n")
    
    # Створення моделі
    model = create_better_model()
    print("📊 Архітектура моделі:")
    model.summary()
    
    # Data augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Завантаження даних
    train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=(48, 48),
        batch_size=64,
        color_mode='grayscale',
        class_mode='categorical'
    )
    
    test_path = 'data/test'
    if os.path.exists(test_path):
        test_generator = test_datagen.flow_from_directory(
            test_path,
            target_size=(48, 48),
            batch_size=64,
            color_mode='grayscale',
            class_mode='categorical'
        )
    else:
        test_generator = None
    
    # Навчання
    print("\n🎯 Початок навчання...\n")
    
    history = model.fit(
        train_generator,
        epochs=50,
        validation_data=test_generator,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            )
        ]
    )
    
    # Збереження
    os.makedirs('models', exist_ok=True)
    model.save('models/emotion_model.h5')
    print("✅ Модель збережена: models/emotion_model.h5")
    
    return True

if __name__ == "__main__":
    import sys
    import tensorflow as tf
    
    if '--train' in sys.argv:
        train_on_fer2013()
    else:
        print("Використання: python train_from_kaggle.py --train")
        train_on_fer2013()
