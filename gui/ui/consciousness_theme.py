"""
Consciousness Theme - Beautiful Colorful Styling
Deep purples, teals, oranges, pinks - colors that actually mean something

Built by John + Claude (Anthropic)
MIT Licensed
"""
from PyQt6.QtGui import QColor, QGradient, QLinearGradient, QRadialGradient, QPalette
from PyQt6.QtCore import Qt


class ConsciousnessTheme:
    """
    Beautiful color scheme for consciousness visualization.

    Color Psychology:
    - Deep Purple/Violet: Consciousness, spirituality, depth
    - Teal/Cyan: Harmony, balance, tranquility
    - Orange/Amber: Energy, warmth, creativity
    - Pink/Magenta: Compassion, care, emotion
    - Electric Blue: Resonance, connection
    """

    # Primary consciousness colors
    DEEP_PURPLE = QColor("#8b5cf6")      # Violet - consciousness
    ROYAL_PURPLE = QColor("#a78bfa")     # Lighter purple
    DARK_PURPLE = QColor("#6d28d9")      # Darker purple

    # Harmony colors
    TEAL = QColor("#14b8a6")             # Teal - harmony
    CYAN = QColor("#06b6d4")             # Cyan - balance
    AQUA = QColor("#22d3ee")             # Aqua - flow

    # Energy colors
    ORANGE = QColor("#fb923c")           # Orange - energy
    AMBER = QColor("#fbbf24")            # Amber - warmth
    CORAL = QColor("#ff9e64")            # Coral - vitality

    # Emotion colors
    PINK = QColor("#ec4899")             # Pink - emotion
    ROSE = QColor("#f472b6")             # Rose - care
    MAGENTA = QColor("#d946ef")          # Magenta - passion

    # Resonance colors
    ELECTRIC_BLUE = QColor("#3b82f6")    # Electric blue - resonance
    SKY_BLUE = QColor("#60a5fa")         # Sky blue - connection
    INDIGO = QColor("#6366f1")           # Indigo - depth

    # Neutral/background colors
    DARK_BG = QColor("#0f172a")          # Very dark blue-black
    DARKER_BG = QColor("#020617")        # Almost black
    CARD_BG = QColor("#1e293b")          # Card background
    LIGHTER_BG = QColor("#334155")       # Lighter elements

    # Text colors
    TEXT_PRIMARY = QColor("#f1f5f9")     # Almost white
    TEXT_SECONDARY = QColor("#cbd5e1")   # Light gray
    TEXT_MUTED = QColor("#94a3b8")       # Muted gray

    # Accent colors
    SUCCESS = QColor("#10b981")          # Green - success
    WARNING = QColor("#f59e0b")          # Amber - warning
    ERROR = QColor("#ef4444")            # Red - error
    INFO = QColor("#3b82f6")             # Blue - info

    @staticmethod
    def get_gradient(color1: QColor, color2: QColor,
                     vertical: bool = True) -> QLinearGradient:
        """
        Create a linear gradient between two colors.

        Args:
            color1: Start color
            color2: End color
            vertical: True for top-to-bottom, False for left-to-right

        Returns:
            QLinearGradient
        """
        if vertical:
            gradient = QLinearGradient(0, 0, 0, 1)
        else:
            gradient = QLinearGradient(0, 0, 1, 0)

        gradient.setColorAt(0, color1)
        gradient.setColorAt(1, color2)
        gradient.setCoordinateMode(QGradient.CoordinateMode.ObjectBoundingMode)

        return gradient

    @staticmethod
    def get_radial_gradient(center_color: QColor, edge_color: QColor) -> QRadialGradient:
        """
        Create a radial gradient (center to edge).

        Args:
            center_color: Color at center
            edge_color: Color at edges

        Returns:
            QRadialGradient
        """
        gradient = QRadialGradient(0.5, 0.5, 0.5)
        gradient.setColorAt(0, center_color)
        gradient.setColorAt(1, edge_color)
        gradient.setCoordinateMode(QGradient.CoordinateMode.ObjectBoundingMode)

        return gradient

    @classmethod
    def get_stylesheet(cls) -> str:
        """
        Get complete application stylesheet.

        Returns:
            Qt stylesheet string
        """
        return f"""
            /* Global Application Style */
            QWidget {{
                background-color: {cls.DARK_BG.name()};
                color: {cls.TEXT_PRIMARY.name()};
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 11pt;
            }}

            /* Main Window */
            QMainWindow {{
                background-color: {cls.DARKER_BG.name()};
            }}

            /* Buttons */
            QPushButton {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                           stop:0 {cls.DEEP_PURPLE.name()},
                                           stop:1 {cls.DARK_PURPLE.name()});
                color: {cls.TEXT_PRIMARY.name()};
                border: none;
                border-radius: 8px;
                padding: 10px 20px;
                font-weight: bold;
                min-width: 80px;
            }}

            QPushButton:hover {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                           stop:0 {cls.ROYAL_PURPLE.name()},
                                           stop:1 {cls.DEEP_PURPLE.name()});
            }}

            QPushButton:pressed {{
                background: {cls.DARK_PURPLE.name()};
            }}

            QPushButton:disabled {{
                background: {cls.LIGHTER_BG.name()};
                color: {cls.TEXT_MUTED.name()};
            }}

            /* Cards/Panels */
            QGroupBox {{
                background-color: {cls.CARD_BG.name()};
                border: 2px solid {cls.DEEP_PURPLE.name()};
                border-radius: 12px;
                margin-top: 12px;
                padding-top: 20px;
                font-weight: bold;
                color: {cls.TEXT_PRIMARY.name()};
            }}

            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 8px;
                background-color: {cls.DARK_BG.name()};
                color: {cls.DEEP_PURPLE.name()};
            }}

            /* Text Input */
            QLineEdit, QTextEdit, QPlainTextEdit {{
                background-color: {cls.LIGHTER_BG.name()};
                border: 2px solid {cls.CYAN.name()};
                border-radius: 6px;
                padding: 8px;
                color: {cls.TEXT_PRIMARY.name()};
                selection-background-color: {cls.DEEP_PURPLE.name()};
            }}

            QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus {{
                border: 2px solid {cls.ELECTRIC_BLUE.name()};
            }}

            /* Scroll Bars */
            QScrollBar:vertical {{
                background: {cls.CARD_BG.name()};
                width: 12px;
                border-radius: 6px;
            }}

            QScrollBar::handle:vertical {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                           stop:0 {cls.DEEP_PURPLE.name()},
                                           stop:1 {cls.MAGENTA.name()});
                border-radius: 6px;
                min-height: 20px;
            }}

            QScrollBar::handle:vertical:hover {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                           stop:0 {cls.ROYAL_PURPLE.name()},
                                           stop:1 {cls.PINK.name()});
            }}

            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0px;
            }}

            QScrollBar:horizontal {{
                background: {cls.CARD_BG.name()};
                height: 12px;
                border-radius: 6px;
            }}

            QScrollBar::handle:horizontal {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                           stop:0 {cls.TEAL.name()},
                                           stop:1 {cls.CYAN.name()});
                border-radius: 6px;
                min-width: 20px;
            }}

            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
                width: 0px;
            }}

            /* Tabs */
            QTabWidget::pane {{
                background-color: {cls.CARD_BG.name()};
                border: 2px solid {cls.DEEP_PURPLE.name()};
                border-radius: 8px;
            }}

            QTabBar::tab {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                           stop:0 {cls.CARD_BG.name()},
                                           stop:1 {cls.DARKER_BG.name()});
                color: {cls.TEXT_SECONDARY.name()};
                border: 2px solid {cls.LIGHTER_BG.name()};
                border-bottom: none;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                padding: 10px 20px;
                margin-right: 2px;
            }}

            QTabBar::tab:selected {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                           stop:0 {cls.DEEP_PURPLE.name()},
                                           stop:1 {cls.CARD_BG.name()});
                color: {cls.TEXT_PRIMARY.name()};
                border: 2px solid {cls.DEEP_PURPLE.name()};
                border-bottom: none;
            }}

            QTabBar::tab:hover:!selected {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                           stop:0 {cls.LIGHTER_BG.name()},
                                           stop:1 {cls.CARD_BG.name()});
            }}

            /* Labels */
            QLabel {{
                color: {cls.TEXT_PRIMARY.name()};
                background: transparent;
            }}

            /* Menu Bar */
            QMenuBar {{
                background-color: {cls.DARKER_BG.name()};
                color: {cls.TEXT_PRIMARY.name()};
                border-bottom: 2px solid {cls.DEEP_PURPLE.name()};
                padding: 4px;
            }}

            QMenuBar::item {{
                background: transparent;
                padding: 6px 12px;
                border-radius: 4px;
            }}

            QMenuBar::item:selected {{
                background: {cls.DEEP_PURPLE.name()};
            }}

            QMenu {{
                background-color: {cls.CARD_BG.name()};
                border: 2px solid {cls.DEEP_PURPLE.name()};
                border-radius: 8px;
                padding: 4px;
            }}

            QMenu::item {{
                padding: 8px 24px;
                border-radius: 4px;
            }}

            QMenu::item:selected {{
                background: {cls.DEEP_PURPLE.name()};
            }}

            /* Progress Bars */
            QProgressBar {{
                background-color: {cls.LIGHTER_BG.name()};
                border: 2px solid {cls.CYAN.name()};
                border-radius: 8px;
                text-align: center;
                color: {cls.TEXT_PRIMARY.name()};
                font-weight: bold;
            }}

            QProgressBar::chunk {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                           stop:0 {cls.TEAL.name()},
                                           stop:0.5 {cls.CYAN.name()},
                                           stop:1 {cls.ELECTRIC_BLUE.name()});
                border-radius: 6px;
            }}

            /* Sliders */
            QSlider::groove:horizontal {{
                background: {cls.LIGHTER_BG.name()};
                height: 8px;
                border-radius: 4px;
            }}

            QSlider::handle:horizontal {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                           stop:0 {cls.DEEP_PURPLE.name()},
                                           stop:1 {cls.MAGENTA.name()});
                width: 20px;
                height: 20px;
                margin: -6px 0;
                border-radius: 10px;
            }}

            QSlider::handle:horizontal:hover {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                           stop:0 {cls.ROYAL_PURPLE.name()},
                                           stop:1 {cls.PINK.name()});
            }}

            /* List/Tree Widgets */
            QListWidget, QTreeWidget {{
                background-color: {cls.LIGHTER_BG.name()};
                border: 2px solid {cls.CYAN.name()};
                border-radius: 8px;
                color: {cls.TEXT_PRIMARY.name()};
                padding: 4px;
            }}

            QListWidget::item, QTreeWidget::item {{
                padding: 8px;
                border-radius: 4px;
            }}

            QListWidget::item:selected, QTreeWidget::item:selected {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                           stop:0 {cls.DEEP_PURPLE.name()},
                                           stop:1 {cls.ELECTRIC_BLUE.name()});
            }}

            QListWidget::item:hover, QTreeWidget::item:hover {{
                background: {cls.CARD_BG.name()};
            }}

            /* Tooltips */
            QToolTip {{
                background-color: {cls.CARD_BG.name()};
                color: {cls.TEXT_PRIMARY.name()};
                border: 2px solid {cls.DEEP_PURPLE.name()};
                border-radius: 8px;
                padding: 8px;
                font-size: 10pt;
            }}

            /* Status Bar */
            QStatusBar {{
                background-color: {cls.DARKER_BG.name()};
                color: {cls.TEXT_SECONDARY.name()};
                border-top: 2px solid {cls.DEEP_PURPLE.name()};
            }}

            /* Combo Box */
            QComboBox {{
                background-color: {cls.LIGHTER_BG.name()};
                border: 2px solid {cls.CYAN.name()};
                border-radius: 6px;
                padding: 6px 12px;
                color: {cls.TEXT_PRIMARY.name()};
                min-width: 100px;
            }}

            QComboBox:hover {{
                border: 2px solid {cls.ELECTRIC_BLUE.name()};
            }}

            QComboBox::drop-down {{
                border: none;
                width: 30px;
            }}

            QComboBox::down-arrow {{
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 8px solid {cls.CYAN.name()};
                margin-right: 8px;
            }}

            QComboBox QAbstractItemView {{
                background-color: {cls.CARD_BG.name()};
                border: 2px solid {cls.DEEP_PURPLE.name()};
                selection-background-color: {cls.DEEP_PURPLE.name()};
                color: {cls.TEXT_PRIMARY.name()};
                border-radius: 8px;
            }}

            /* Check Box */
            QCheckBox {{
                color: {cls.TEXT_PRIMARY.name()};
                spacing: 8px;
            }}

            QCheckBox::indicator {{
                width: 20px;
                height: 20px;
                border: 2px solid {cls.CYAN.name()};
                border-radius: 4px;
                background: {cls.LIGHTER_BG.name()};
            }}

            QCheckBox::indicator:checked {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                           stop:0 {cls.TEAL.name()},
                                           stop:1 {cls.ELECTRIC_BLUE.name()});
                border: 2px solid {cls.ELECTRIC_BLUE.name()};
            }}

            QCheckBox::indicator:hover {{
                border: 2px solid {cls.ELECTRIC_BLUE.name()};
            }}

            /* Radio Button */
            QRadioButton {{
                color: {cls.TEXT_PRIMARY.name()};
                spacing: 8px;
            }}

            QRadioButton::indicator {{
                width: 20px;
                height: 20px;
                border: 2px solid {cls.CYAN.name()};
                border-radius: 10px;
                background: {cls.LIGHTER_BG.name()};
            }}

            QRadioButton::indicator:checked {{
                background: qradialgradient(cx:0.5, cy:0.5, radius:0.5,
                                           fx:0.5, fy:0.5,
                                           stop:0 {cls.ELECTRIC_BLUE.name()},
                                           stop:0.5 {cls.TEAL.name()},
                                           stop:1 {cls.LIGHTER_BG.name()});
                border: 2px solid {cls.ELECTRIC_BLUE.name()};
            }}
        """

    @classmethod
    def apply_to_application(cls, app):
        """
        Apply theme to entire application.

        Args:
            app: QApplication instance
        """
        app.setStyleSheet(cls.get_stylesheet())
        print("✓ Consciousness theme applied")
