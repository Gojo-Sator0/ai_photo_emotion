"""
Модуль для обробки зображень
"""
import cv2
import numpy as np
from PIL import Image, ImageTk
from utils.constants import IMAGE_SIZE


class ImageProcessor:
    """Клас для обробки зображень"""
    
    @staticmethod
    def preprocess_for_model(image):
        """
        Попередня обробка зображення для моделі
        
        Args:
            image (numpy.ndarray): вхідне зображення
            
        Returns:
            numpy.ndarray: оброблене зображення
        """
        try:
            # Конвертація в градації сірого
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image.copy()
            
            # Зміна розміру
            resized = cv2.resize(gray, IMAGE_SIZE)
            
            # Нормалізація
            normalized = resized.astype('float32') / 255.0
            
            # Додавання розмірностей для моделі
            processed = np.expand_dims(normalized, axis=0)
            processed = np.expand_dims(processed, axis=-1)
            
            return processed
            
        except Exception as e:
            raise Exception(f"Помилка обробки зображення: {str(e)}")
    
    @staticmethod
    def prepare_for_display(image_path, max_size=(400, 400)):
        """
        Підготовка зображення для відображення в UI
        
        Args:
            image_path (str): шлях до зображення
            max_size (tuple): максимальний розмір
            
        Returns:
            tuple: (PIL.Image, ImageTk.PhotoImage, numpy.ndarray)
        """
        try:
            # Завантаження
            pil_image = Image.open(image_path)
            original_array = np.array(pil_image)
            
            # Зменшення для відображення
            display_image = pil_image.copy()
            display_image.thumbnail(max_size, Image.Resampling.LANCZOS)
            
            # Конвертація для tkinter
            photo = ImageTk.PhotoImage(display_image)
            
            return pil_image, photo, original_array
            
        except Exception as e:
            raise Exception(f"Помилка підготовки зображення: {str(e)}")
    
    @staticmethod
    def detect_face(image):
        """
        Виявлення обличчя на зображенні (опціонально)
        
        Args:
            image (numpy.ndarray): зображення
            
        Returns:
            numpy.ndarray: обрізане зображення з обличчям або оригінал
        """
        try:
            # Завантаження каскаду Хаара для виявлення облич
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            
            # Конвертація в сірі тони, якщо потрібно
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
            
            # Виявлення облич
            faces = face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=5, 
                minSize=(30, 30)
            )
            
            if len(faces) > 0:
                # Беремо перше знайдене обличчя
                (x, y, w, h) = faces[0]
                face_roi = image[y:y+h, x:x+w]
                return face_roi
            else:
                # Якщо обличчя не знайдено, повертаємо оригінал
                return image
                
        except Exception as e:
            print(f"⚠️ Помилка виявлення обличчя: {e}")
            return image
