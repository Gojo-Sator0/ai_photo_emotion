"""
GUI для розпізнавання емоцій на фотографіях
"""
import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import cv2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import tf_keras

EMOTIONS    = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
EMOTIONS_UA = ['Злість', 'Відраза', 'Страх', 'Радість', 'Сум', 'Подив', 'Нейтрально']

EMOTION_COLORS = {
    'Angry':    '#FF4444',
    'Disgust':  '#AA44FF',
    'Fear':     '#FF8800',
    'Happy':    '#44DD44',
    'Sad':      '#4488FF',
    'Surprise': '#FFDD00',
    'Neutral':  '#AAAAAA',
}

MODEL_PATH = os.path.join('models', 'model_v6_23.hdf5')


class EmotionRecognitionApp:

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("🎭 Розпізнавання емоцій")
        self.root.geometry("900x650")
        self.root.resizable(True, True)
        self.root.configure(bg='#1E1E2E')

        self.model = None
        self.current_image_path = None
        self.tk_image = None

        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

        self._build_ui()
        self._load_model()

    # ──────────────────────────────────────────────────────────
    # UI
    # ──────────────────────────────────────────────────────────

    def _build_ui(self):
        # Заголовок
        header = tk.Frame(self.root, bg='#2E2E4E', pady=10)
        header.pack(fill='x')

        tk.Label(
            header, text="🎭 Розпізнавання емоцій на фото",
            font=('Segoe UI', 18, 'bold'),
            bg='#2E2E4E', fg='#CDD6F4'
        ).pack()

        tk.Label(
            header,
            text="Завантажте фото → Програма знайде обличчя та визначить емоції",
            font=('Segoe UI', 10),
            bg='#2E2E4E', fg='#888'
        ).pack()

        # Контент
        content = tk.Frame(self.root, bg='#1E1E2E')
        content.pack(fill='both', expand=True, padx=15, pady=10)

        # Ліва панель — зображення
        left = tk.Frame(content, bg='#1E1E2E')
        left.pack(side='left', fill='both', expand=True)

        self.image_frame = tk.Label(
            left,
            text="📂 Натисніть «Відкрити фото»\nщоб завантажити зображення",
            bg='#2E2E4E', fg='#888',
            font=('Segoe UI', 12),
            width=50, height=20,
            relief='groove', bd=2
        )
        self.image_frame.pack(fill='both', expand=True, padx=(0, 10))

        # Права панель — результати
        right = tk.Frame(content, bg='#1E1E2E', width=260)
        right.pack(side='right', fill='y')
        right.pack_propagate(False)

        tk.Label(
            right, text="📊 Результати аналізу",
            font=('Segoe UI', 13, 'bold'),
            bg='#1E1E2E', fg='#CDD6F4'
        ).pack(anchor='w', pady=(0, 8))

        self.model_status = tk.Label(
            right, text="⏳ Завантаження моделі...",
            font=('Segoe UI', 9),
            bg='#1E1E2E', fg='#F38BA8'
        )
        self.model_status.pack(anchor='w', pady=(0, 10))

        self.main_emotion_label = tk.Label(
            right, text="—",
            font=('Segoe UI', 26, 'bold'),
            bg='#1E1E2E', fg='#CDD6F4'
        )
        self.main_emotion_label.pack(pady=(5, 0))

        self.confidence_label = tk.Label(
            right, text="",
            font=('Segoe UI', 13),
            bg='#1E1E2E', fg='#A6E3A1'
        )
        self.confidence_label.pack(pady=(0, 15))

        tk.Label(
            right, text="Розподіл ймовірностей:",
            font=('Segoe UI', 10, 'bold'),
            bg='#1E1E2E', fg='#888'
        ).pack(anchor='w')

        self.bars_frame = tk.Frame(right, bg='#1E1E2E')
        self.bars_frame.pack(fill='x', pady=5)

        self._build_bars()

        self.faces_label = tk.Label(
            right, text="",
            font=('Segoe UI', 10),
            bg='#1E1E2E', fg='#89DCEB'
        )
        self.faces_label.pack(anchor='w', pady=(10, 0))

        # Кнопки
        btn_frame = tk.Frame(self.root, bg='#1E1E2E', pady=10)
        btn_frame.pack(fill='x', padx=15)

        btn_style = dict(
            font=('Segoe UI', 11, 'bold'),
            bd=0, padx=20, pady=10,
            cursor='hand2', relief='flat'
        )

        tk.Button(
            btn_frame, text="📂  Відкрити фото",
            bg='#89B4FA', fg='#1E1E2E',
            command=self._open_image,
            **btn_style
        ).pack(side='left', padx=(0, 10))

        self.analyze_btn = tk.Button(
            btn_frame, text="🔍  Аналізувати",
            bg='#A6E3A1', fg='#1E1E2E',
            command=self._analyze_image,
            state='disabled',
            **btn_style
        )
        self.analyze_btn.pack(side='left', padx=(0, 10))

        tk.Button(
            btn_frame, text="🗑  Очистити",
            bg='#F38BA8', fg='#1E1E2E',
            command=self._clear,
            **btn_style
        ).pack(side='left')

        # Статус-рядок
        self.status_bar = tk.Label(
            self.root, text="Готово до роботи",
            font=('Segoe UI', 9),
            bg='#2E2E4E', fg='#888',
            anchor='w', padx=10
        )
        self.status_bar.pack(fill='x', side='bottom')

    def _build_bars(self):
        self.bar_widgets = {}
        for emo, emo_ua in zip(EMOTIONS, EMOTIONS_UA):
            row = tk.Frame(self.bars_frame, bg='#1E1E2E')
            row.pack(fill='x', pady=2)

            tk.Label(
                row, text=f"{emo_ua:<12}",
                font=('Consolas', 9),
                bg='#1E1E2E', fg='#CDD6F4',
                width=12, anchor='w'
            ).pack(side='left')

            canvas = tk.Canvas(
                row, height=14, bg='#313244',
                bd=0, highlightthickness=0, width=120
            )
            canvas.pack(side='left', padx=3)

            pct_label = tk.Label(
                row, text="0%",
                font=('Consolas', 9),
                bg='#1E1E2E', fg='#888', width=5
            )
            pct_label.pack(side='left')

            self.bar_widgets[emo] = (canvas, pct_label)

    # ──────────────────────────────────────────────────────────
    # Модель
    # ──────────────────────────────────────────────────────────

    def _load_model(self):
        self._set_status("⏳ Завантаження моделі...")
        try:
            self.model = tf_keras.models.load_model(MODEL_PATH, compile=False)
            self.model_status.config(text="✅ Модель завантажена", fg='#A6E3A1')
            self._set_status("✅ Модель готова. Відкрийте фото для аналізу.")
        except Exception as e:
            self.model_status.config(text="❌ Помилка моделі", fg='#F38BA8')
            self._set_status(f"❌ Помилка: {e}")
            messagebox.showerror("Помилка моделі", str(e))

    # ──────────────────────────────────────────────────────────
    # Відкриття фото
    # ──────────────────────────────────────────────────────────

    def _open_image(self):
        path = filedialog.askopenfilename(
            title="Оберіть фото",
            filetypes=[
                ("Зображення", "*.jpg *.jpeg *.png *.bmp *.webp *.tiff"),
                ("Всі файли", "*.*")
            ]
        )
        if not path:
            return

        self.current_image_path = path
        self._display_image(path)
        self.analyze_btn.config(state='normal')
        self._set_status(f"📂 Завантажено: {os.path.basename(path)}")
        self._reset_results()

    def _display_image(self, path, faces=None):
        img = cv2.imread(path)
        if img is None:
            messagebox.showerror("Помилка", "Не вдалося відкрити зображення")
            return

        if faces is not None:
            for (x, y, w, h), emo, conf in faces:
                color_bgr = self._hex_to_bgr(EMOTION_COLORS.get(emo, '#FFFFFF'))
                cv2.rectangle(img, (x, y), (x + w, y + h), color_bgr, 3)
                label_text = f"{emo} {conf:.0f}%"
                (tw, th), _ = cv2.getTextSize(
                    label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
                )
                cv2.rectangle(img, (x, y - th - 12), (x + tw + 6, y), color_bgr, -1)
                cv2.putText(
                    img, label_text, (x + 3, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
                )

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        pil_img.thumbnail((580, 480), Image.LANCZOS)

        self.tk_image = ImageTk.PhotoImage(pil_img)
        self.image_frame.config(image=self.tk_image, text='')

    # ──────────────────────────────────────────────────────────
    # Аналіз
    # ──────────────────────────────────────────────────────────

    def _analyze_image(self):
        if not self.current_image_path or self.model is None:
            return

        self._set_status("🔍 Аналіз зображення...")
        self.root.update()

        img = cv2.imread(self.current_image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces_rects = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )

        if len(faces_rects) == 0:
            self._set_status("⚠️ Облич не знайдено. Спробуйте інше фото.")
            messagebox.showwarning(
                "Обличчя не знайдено",
                "На цьому фото не вдалося знайти обличчя.\n\n"
                "Поради:\n• Фото має бути чітким\n"
                "• Обличчя має бути повернуте до камери\n"
                "• Достатнє освітлення"
            )
            return

        faces_result = []
        last_preds = None

        for (x, y, w, h) in faces_rects:
            roi = gray[y:y + h, x:x + w]
            roi = cv2.resize(roi, (48, 48))
            roi = roi.astype('float32') / 255.0
            roi = np.expand_dims(roi, axis=0)
            roi = np.expand_dims(roi, axis=-1)

            preds = self.model.predict(roi, verbose=0)[0]
            max_idx = int(np.argmax(preds))
            emo = EMOTIONS[max_idx]
            conf = float(preds[max_idx]) * 100

            faces_result.append(((x, y, w, h), emo, conf))
            last_preds = preds

        self._display_image(self.current_image_path, faces_result)

        first_emo = faces_result[0][1]
        first_conf = faces_result[0][2]
        first_emo_ua = EMOTIONS_UA[EMOTIONS.index(first_emo)]

        self.main_emotion_label.config(
            text=first_emo_ua,
            fg=EMOTION_COLORS.get(first_emo, '#CDD6F4')
        )
        self.confidence_label.config(text=f"{first_conf:.1f}% впевненість")
        self.faces_label.config(text=f"👤 Знайдено облич: {len(faces_rects)}")
        self._update_bars(last_preds)

        self._set_status(
            f"✅ Готово! Знайдено {len(faces_rects)} обличч(я). "
            f"Головна емоція: {first_emo_ua} ({first_conf:.1f}%)"
        )

    # ──────────────────────────────────────────────────────────
    # Допоміжні методи
    # ──────────────────────────────────────────────────────────

    def _update_bars(self, predictions):
        for emo, prob in zip(EMOTIONS, predictions):
            canvas, pct_label = self.bar_widgets[emo]
            width = int(prob * 120)
            color = EMOTION_COLORS.get(emo, '#888888')
            canvas.delete('all')
            if width > 0:
                canvas.create_rectangle(0, 0, width, 14, fill=color, outline='')
            pct_label.config(text=f"{prob * 100:.0f}%")

    def _reset_results(self):
        self.main_emotion_label.config(text="—", fg='#CDD6F4')
        self.confidence_label.config(text="")
        self.faces_label.config(text="")
        for emo in EMOTIONS:
            canvas, pct_label = self.bar_widgets[emo]
            canvas.delete('all')
            pct_label.config(text="0%")

    def _clear(self):
        self.current_image_path = None
        self.tk_image = None
        self.image_frame.config(
            image='',
            text="📂 Натисніть «Відкрити фото»\nщоб завантажити зображення"
        )
        self.analyze_btn.config(state='disabled')
        self._reset_results()
        self._set_status("🗑 Очищено")

    def _set_status(self, text: str):
        self.status_bar.config(text=text)
        self.root.update_idletasks()

    @staticmethod
    def _hex_to_bgr(hex_color: str):
        h = hex_color.lstrip('#')
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return (b, g, r)
