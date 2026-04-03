"""
Перевірка структури датасету
"""
import os
from pathlib import Path

def verify_dataset():
    """Перевірка датасету"""
    
    print("🔍 Перевірка структури датасету...\n")
    
    # Перевірка основної папки
    if not os.path.exists('data'):
        print("❌ Папка 'data' не знайдена!")
        print("✅ Створіть папку: D:\\Project\\AI photo emotion\\data\\")
        return False
    
    print("✅ Папка data знайдена")
    
    # Перевірка train та test
    for folder in ['train', 'test']:
        folder_path = f'data/{folder}'
        
        if not os.path.exists(folder_path):
            print(f"❌ Папка 'data/{folder}' не знайдена!")
            return False
        
        print(f"✅ Папка data/{folder} знайдена")
        
        # Перевірка емоцій
        emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        
        for emotion in emotions:
            emotion_path = f'{folder_path}/{emotion}'
            
            if not os.path.exists(emotion_path):
                print(f"   ❌ Папка {emotion} не знайдена в {folder}")
                return False
            
            # Підрахунок зображень
            images = [f for f in os.listdir(emotion_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
            count = len(images)
            
            if count == 0:
                print(f"   ❌ {emotion}: 0 зображень (помилка!)")
                return False
            
            print(f"   ✅ {emotion}: {count} зображень")
    
    print("\n" + "="*50)
    print("✅ ВСЕ ПРАВИЛЬНО! Датасет готовий до навчання")
    print("="*50)
    
    # Загальна статистика
    total_train = 0
    total_test = 0
    
    for emotion in emotions:
        train_count = len(os.listdir(f'data/train/{emotion}'))
        test_count = len(os.listdir(f'data/test/{emotion}'))
        total_train += train_count
        total_test += test_count
    
    print(f"\n📊 СТАТИСТИКА:")
    print(f"   Train датасет: {total_train} зображень")
    print(f"   Test датасет: {total_test} зображень")
    print(f"   Всього: {total_train + total_test} зображень")
    
    return True

if __name__ == "__main__":
    success = verify_dataset()
    
    if success:
        print("\n🚀 Тепер запустіть: python train_from_kaggle.py --train")
    else:
        print("\n❌ Виправте структуру датасету та спробуйте знову")
