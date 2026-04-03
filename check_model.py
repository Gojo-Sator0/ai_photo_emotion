import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Використовуємо legacy tf_keras
import tensorflow as tf
os.environ['TF_USE_LEGACY_KERAS'] = '1'
import tf_keras

model_path = 'models/model_v6_23.hdf5'

if not os.path.exists(model_path):
    print(f"❌ Модель не знайдена: {model_path}")
    exit()

print(f"✅ Модель знайдена: {model_path}")

try:
    model = tf_keras.models.load_model(model_path, compile=False)
    print("✅ Модель успішно завантажена!")
except Exception as e:
    print(f"❌ Помилка завантаження: {e}")
    exit()

# Тестовий вхід
test_input = np.random.rand(1, 48, 48, 1).astype(np.float32)
predictions = model.predict(test_input, verbose=0)
probabilities = predictions[0]

emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

print("\n📊 Розподіл ймовірностей:")
for emotion, prob in zip(emotions, probabilities):
    bar = '█' * int(prob * 30)
    print(f"{emotion:<12} {prob*100:5.2f}% {bar}")

max_prob = max(probabilities)
max_emotion = emotions[list(probabilities).index(max_prob)]

print(f"\n🏆 Найвища ймовірність: {max_emotion} ({max_prob*100:.2f}%)")

if max_prob > 0.5:
    print("✅ Модель добре навчена!")
else:
    print("⚠️ Для випадкового зображення це нормально — тестуй з реальним обличчям!")

print(f"\n📐 Input shape:  {model.input_shape}")
print(f"📐 Output shape: {model.output_shape}")
print(f"🔢 Параметри:    {model.count_params():,}")
