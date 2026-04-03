"""
Константи програми
"""
import os

# Шляхи
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
MODEL_PATH = os.path.join(MODELS_DIR, 'emotion_model.h5')

# Емоції FER-2013
EMOTION_LABELS = [
    'Angry',     # Гнів
    'Disgust',   # Огида  
    'Fear',      # Страх
    'Happy',     # Радість
    'Sad',       # Сум
    'Surprise',  # Здивування
    'Neutral'    # Нейтральний
]

# Кольори емоцій
EMOTION_COLORS = {
    'Angry': '#FF6B6B',      # Червоний
    'Disgust': '#4ECDC4',    # Бірюзовий
    'Fear': '#45B7D1',       # Синій
    'Happy': '#96CEB4',      # Зелений
    'Sad': '#FFEAA7',        # Жовтий
    'Surprise': '#DDA0DD',   # Фіолетовий
    'Neutral': '#74B9FF'     # Блакитний
}

# Параметри зображення
IMAGE_SIZE = (48, 48)
INPUT_SHAPE = (48, 48, 1)

# UI налаштування
WINDOW_SIZE = "900x700"
MAIN_COLOR = '#2c3e50'
BG_COLOR = '#f0f0f0'
