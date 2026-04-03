"""
Стилі та налаштування GUI
"""
from utils.constants import MAIN_COLOR, BG_COLOR


class AppStyles:
    """Стилі програми"""
    
    # Кольори
    COLORS = {
        'primary': MAIN_COLOR,
        'background': BG_COLOR,
        'white': '#ffffff',
        'success': '#27ae60',
        'info': '#3498db',
        'warning': '#f39c12',
        'error': '#e74c3c',
        'text_light': '#7f8c8d',
        'text_dark': '#2c3e50'
    }
    
    # Шрифти
    FONTS = {
        'title': ('Arial', 18, 'bold'),
        'header': ('Arial', 14, 'bold'),
        'button': ('Arial', 12, 'bold'),
        'normal': ('Arial', 10),
        'small': ('Arial', 9),
        'result': ('Arial', 12, 'bold')
    }
    
    # Кнопки
    BUTTON_STYLE = {
        'relief': 'flat',
        'cursor': 'hand2',
        'pady': 10,
        'font': FONTS['button']
    }
    
    @classmethod
    def get_button_style(cls, button_type='default'):
        """Отримання стилю кнопки"""
        base_style = cls.BUTTON_STYLE.copy()
        
        if button_type == 'primary':
            base_style.update({
                'bg': cls.COLORS['info'],
                'fg': cls.COLORS['white']
            })
        elif button_type == 'success':
            base_style.update({
                'bg': cls.COLORS['success'],
                'fg': cls.COLORS['white']
            })
        elif button_type == 'warning':
            base_style.update({
                'bg': cls.COLORS['warning'],
                'fg': cls.COLORS['white']
            })
        
        return base_style
