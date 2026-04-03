"""
Завантаження готової навченої моделі
"""
import os
import urllib.request
import shutil

def download_model():
    """Завантаження моделі з надійного джерела"""
    
    os.makedirs('models', exist_ok=True)
    
    # URL готової моделі (від автора проєкту FER-2013)
    urls = [
        # Основний URL
        "https://github.com/mwiśniewski/emotion-recognition-neural-networks/raw/master/model/emotion_model.h5",
        # Альтернативний URL
        "https://drive.google.com/uc?export=download&id=1nwACP2Pb8e9gXRhzP8D0_5aLr9WstB-7"
    ]
    
    model_path = 'models/emotion_model.h5'
    
    # Якщо модель вже існує
    if os.path.exists(model_path):
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"📁 Модель вже існує ({size_mb:.1f} MB)")
        
        # Перевірка розміру (навчена модель > 50MB)
        if size_mb < 50:
            print("⚠️ Файл занадто малий, ймовірно не навчена модель")
            os.remove(model_path)
            print("🗑️ Видалено старий файл")
        else:
            print("✅ Модель виглядає добре, розмір в порядку")
            return True
    
    print("📥 Завантаження готової моделі...\n")
    
    for i, url in enumerate(urls, 1):
        try:
            print(f"🔗 Спроба {i}: {url[:50]}...")
            
            urllib.request.urlretrieve(
                url,
                model_path,
                reporthook=show_progress
            )
            
            # Перевірка розміру
            size_mb = os.path.getsize(model_path) / (1024 * 1024)
            print(f"\n✅ Успішно завантажено! ({size_mb:.1f} MB)")
            return True
            
        except Exception as e:
            print(f"\n❌ Спроба {i} не вдалася: {str(e)[:50]}")
            if os.path.exists(model_path):
                os.remove(model_path)
            continue
    
    print("\n" + "="*60)
    print("❌ Не вдалося завантажити автоматично")
    print("="*60)
    print("\n📖 ВРУЧНУ завантажте модель:")
    print("\n1️⃣  Перейдіть на: https://github.com/mwiśniewski/emotion-recognition-neural-networks")
    print("2️⃣  Знайдіть папку 'model' → файл 'emotion_model.h5'")
    print("3️⃣  Натисніть 'Download' або 'Download raw file'")
    print("4️⃣  Помістіть файл у папку: D:\\Project\\AI photo emotion\\models\\")
    print("\nАБО")
    print("\n1️⃣  Перейдіть на: https://www.kaggle.com/datasets/msambare/fer2013")
    print("2️⃣  Завантажте датасет")
    print("3️⃣  Запустіть: python train_real_model.py")
    print("\n" + "="*60)
    
    return False

def show_progress(block_num, block_size, total_size):
    """Прогрес завантаження"""
    if total_size <= 0:
        return
    
    downloaded = block_num * block_size
    percent = min(downloaded * 100 // total_size, 100)
    
    bar_length = 40
    filled = int(bar_length * percent // 100)
    bar = '█' * filled + '░' * (bar_length - filled)
    
    print(f'\r[{bar}] {percent}% ({downloaded/(1024*1024):.1f}MB)', end='', flush=True)

if __name__ == "__main__":
    success = download_model()
    
    if success:
        print("\n🚀 Тепер запустіть: python main.py")
    else:
        print("\n⚠️ Виконайте инструкцію вище для ручного завантаження")
