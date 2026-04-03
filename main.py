"""
Головний файл програми розпізнавання емоцій (режим фото)
"""
import sys
import os
import tkinter as tk
from tkinter import messagebox

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from gui.main_window import EmotionRecognitionApp


def main():
    """Запуск програми"""
    try:
        model_path = os.path.join('models', 'model_v6_23.hdf5')
        if not os.path.exists(model_path):
            messagebox.showerror(
                "Модель не знайдена",
                f"Файл моделі не знайдено:\n{model_path}\n\nПереконайся що файл існує."
            )
            return

        root = tk.Tk()
        app = EmotionRecognitionApp(root)
        root.mainloop()

    except KeyboardInterrupt:
        print("\n👋 Програма завершена користувачем")
    except Exception as e:
        print(f"❌ Критична помилка: {e}")
        messagebox.showerror("Помилка", f"Не вдалося запустити програму:\n{e}")


if __name__ == "__main__":
    main()
