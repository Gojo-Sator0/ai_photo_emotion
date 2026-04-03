"""
Скрипт для створення та збереження моделі розпізнавання емоцій
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from utils.constants import MODELS_DIR, MODEL_PATH, EMOTION_LABELS, INPUT_SHAPE


def create_emotion_model():
    """
    Створення CNN моделі для розпізнавання емоцій
    
    Returns:
        tensorflow.keras.Model: скомпільована модель
    """
    model = models.Sequential([
        # Перший конволюційний блок
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=INPUT_SHAPE),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Другий конволюційний блок
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Третій конволюційний блок
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Четвертий конволюційний блок
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Повнозв'язні шари
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        
        # Вихідний шар
        layers.Dense(len(EMOTION_LABELS), activation='softmax')
    ])
    
    # Компіляція моделі
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def create_dummy_data():
    """
    Створення тестових даних для демонстрації
    
    Returns:
        tuple: (X_train, y_train, X_test, y_test)
    """
    print("📊 Створення тестових даних...")
    
    # Генерація випадкових даних
    num_samples = 1000
    X_train = np.random.random((num_samples, 48, 48, 1)).astype('float32')
    X_test = np.random.random((200, 48, 48, 1)).astype('float32')
    
    # Випадкові лейбли
    y_train = tf.keras.utils.to_categorical(
        np.random.randint(0, len(EMOTION_LABELS), num_samples),
        len(EMOTION_LABELS)
    )
    y_test = tf.keras.utils.to_categorical(
        np.random.randint(0, len(EMOTION_LABELS), 200),
        len(EMOTION_LABELS)
    )
    
    return X_train, y_train, X_test, y_test


def create_and_save_model():
    """Створення, навчання та збереження моделі"""
    try:
        print("🤖 Створення моделі розпізнавання емоцій...")
        
        # Створення папки для моделей
        os.makedirs(MODELS_DIR, exist_ok=True)
        
        # Створення моделі
        model = create_emotion_model()
        
        print("📋 Архітектура моделі:")
        model.summary()
        
        # Створення тестових даних
        X_train, y_train, X_test, y_test = create_dummy_data()
        
        # Швидке навчання на тестових даних
        print("🎯 Початок навчання моделі...")
        
        history = model.fit(
            X_train, y_train,
            batch_size=32,
            epochs=5,  # Мало епох для швидкого тесту
            validation_data=(X_test, y_test),
            verbose=1
        )
        
        # Збереження моделі
        model.save(MODEL_PATH)
        print(f"✅ Модель збережена: {MODEL_PATH}")
        
        # Тестування моделі
        print("🧪 Тестування моделі...")
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
        print(f"📈 Точність на тестових даних: {test_accuracy:.2%}")
        
        return True
        
    except Exception as e:
        print(f"❌ Помилка створення моделі: {e}")
        return False


if __name__ == "__main__":
    create_and_save_model()
