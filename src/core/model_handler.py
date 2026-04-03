"""
Модуль для роботи з моделлю машинного навчання
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from utils.constants import MODEL_PATH, EMOTION_LABELS


class ModelHandler:
    """Клас для роботи з моделлю емоцій"""
    
    def __init__(self):
        """Ініціалізація обробника моделі"""
        self.model = None
        self.is_loaded = False
        self.load_model()
    
    def load_model(self):
        """Завантаження моделі"""
        try:
            if os.path.exists(MODEL_PATH):
                # Вимкнути попередження TensorFlow
                tf.get_logger().setLevel('ERROR')
                
                self.model = load_model(MODEL_PATH)
                self.is_loaded = True
                print(f"✅ Модель завантажена: {MODEL_PATH}")
                
                # Інформація про модель
                print(f"📊 Архітектура моделі:")
                print(f"   - Вхідна форма: {self.model.input_shape}")
                print(f"   - Кількість класів: {len(EMOTION_LABELS)}")
                
            else:
                print(f"⚠️ Файл моделі не знайдено: {MODEL_PATH}")
                self.is_loaded = False
                
        except Exception as e:
            print(f"❌ Помилка завантаження моделі: {e}")
            self.is_loaded = False
    
    def predict_emotion(self, processed_image):
        """
        Прогнозування емоції
        
        Args:
            processed_image (numpy.ndarray): оброблене зображення
            
        Returns:
            tuple: (predicted_emotion, confidence, all_probabilities)
        """
        try:
            if self.is_loaded and self.model is not None:
                # Реальний прогноз
                predictions = self.model.predict(processed_image, verbose=0)
                probabilities = predictions[0]
            else:
                # Демо режим з випадковими значеннями
                print("🔄 Демо режим: використання випадкових значень")
                probabilities = np.random.random(len(EMOTION_LABELS))
                probabilities = probabilities / probabilities.sum()
            
            # Визначення найймовірнішої емоції
            predicted_idx = np.argmax(probabilities)
            predicted_emotion = EMOTION_LABELS[predicted_idx]
            confidence = probabilities[predicted_idx] * 100
            
            return predicted_emotion, confidence, probabilities
            
        except Exception as e:
            raise Exception(f"Помилка прогнозування: {str(e)}")
    
    def get_model_info(self):
        """Отримання інформації про модель"""
        if self.is_loaded:
            return {
                'loaded': True,
                'input_shape': self.model.input_shape,
                'output_shape': self.model.output_shape,
                'parameters': self.model.count_params()
            }
        else:
            return {'loaded': False}
