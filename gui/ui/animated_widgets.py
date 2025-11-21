"""
Animated Widgets - Beautiful UI animations for consciousness platform
Typing effects, fade-ins, smooth transitions

Built by John + Claude (Anthropic)
MIT Licensed
"""
from PyQt6.QtWidgets import QTextEdit, QProgressBar, QGraphicsOpacityEffect
from PyQt6.QtCore import QTimer, QPropertyAnimation, QEasingCurve, pyqtProperty, QObject
from PyQt6.QtGui import QTextCursor, QColor
from typing import Optional


class TypingTextEdit(QTextEdit):
    """
    Text edit widget with typewriter effect.
    Animates text appearing character by character.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)

        # Typing animation state
        self.is_typing = False
        self.typing_text = ""
        self.typing_index = 0
        self.typing_speed = 20  # ms per character

        # Typing timer
        self.typing_timer = QTimer()
        self.typing_timer.timeout.connect(self._type_next_char)

    def type_text(self, text: str, speed: int = 20):
        """
        Animate typing of text.

        Args:
            text: Text to type
            speed: Milliseconds per character
        """
        self.typing_text = text
        self.typing_index = 0
        self.typing_speed = speed
        self.is_typing = True

        # Clear current text
        self.clear()

        # Start typing
        self.typing_timer.start(speed)

    def append_instant(self, text: str):
        """Append text without animation."""
        self.moveCursor(QTextCursor.MoveOperation.End)
        self.insertPlainText(text)
        self.moveCursor(QTextCursor.MoveOperation.End)

    def _type_next_char(self):
        """Type the next character."""
        if self.typing_index < len(self.typing_text):
            # Get next character
            char = self.typing_text[self.typing_index]

            # Insert character
            self.moveCursor(QTextCursor.MoveOperation.End)
            self.insertPlainText(char)
            self.moveCursor(QTextCursor.MoveOperation.End)

            self.typing_index += 1
        else:
            # Typing complete
            self.typing_timer.stop()
            self.is_typing = False

    def stop_typing(self):
        """Stop typing animation and show all remaining text."""
        if self.is_typing:
            self.typing_timer.stop()

            # Show remaining text instantly
            remaining = self.typing_text[self.typing_index:]
            if remaining:
                self.append_instant(remaining)

            self.is_typing = False


class FadeInWidget(QObject):
    """
    Helper to add fade-in animation to any widget.
    """

    def __init__(self, widget):
        super().__init__(widget)
        self.widget = widget

        # Create opacity effect
        self.opacity_effect = QGraphicsOpacityEffect(widget)
        self.widget.setGraphicsEffect(self.opacity_effect)

        # Create animation
        self.animation = QPropertyAnimation(self.opacity_effect, b"opacity")
        self.animation.setDuration(500)  # 500ms
        self.animation.setStartValue(0.0)
        self.animation.setEndValue(1.0)
        self.animation.setEasingCurve(QEasingCurve.Type.InOutQuad)

    def fade_in(self, duration: int = 500):
        """
        Fade in the widget.

        Args:
            duration: Animation duration in milliseconds
        """
        self.animation.setDuration(duration)
        self.animation.start()

    def fade_out(self, duration: int = 500):
        """
        Fade out the widget.

        Args:
            duration: Animation duration in milliseconds
        """
        self.animation.setDuration(duration)
        self.animation.setStartValue(1.0)
        self.animation.setEndValue(0.0)
        self.animation.start()


class PulsingProgressBar(QProgressBar):
    """
    Progress bar with pulsing glow animation.
    Perfect for consciousness metrics visualization.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        # Pulse state
        self.pulse_direction = 1  # 1 for up, -1 for down
        self.pulse_intensity = 0.0  # 0.0 to 1.0
        self.base_color = QColor("#53bba5")

        # Pulse timer
        self.pulse_timer = QTimer()
        self.pulse_timer.timeout.connect(self._update_pulse)
        self.pulse_timer.start(50)  # 20 FPS

    def set_base_color(self, color: QColor):
        """Set the base color for the progress bar."""
        self.base_color = color
        self._update_style()

    def _update_pulse(self):
        """Update pulse animation."""
        # Update intensity
        self.pulse_intensity += 0.05 * self.pulse_direction

        # Reverse direction at bounds
        if self.pulse_intensity >= 1.0:
            self.pulse_intensity = 1.0
            self.pulse_direction = -1
        elif self.pulse_intensity <= 0.0:
            self.pulse_intensity = 0.0
            self.pulse_direction = 1

        # Update style
        self._update_style()

    def _update_style(self):
        """Update stylesheet with current pulse intensity."""
        # Calculate glow color
        glow_alpha = int(100 + (self.pulse_intensity * 155))

        # Create gradient effect
        color_name = self.base_color.name()
        lighter = self.base_color.lighter(120).name()

        self.setStyleSheet(f"""
            QProgressBar {{
                background-color: #414868;
                border: 2px solid {color_name};
                border-radius: 5px;
                text-align: center;
                color: #f7f7f7;
                font-weight: bold;
            }}
            QProgressBar::chunk {{
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 {color_name},
                    stop:1 {lighter}
                );
                border-radius: 3px;
            }}
        """)

    def set_value_animated(self, value: int, duration: int = 300):
        """
        Set value with smooth animation.

        Args:
            value: Target value
            duration: Animation duration in ms
        """
        # Create animation
        self.animation = QPropertyAnimation(self, b"value")
        self.animation.setDuration(duration)
        self.animation.setStartValue(self.value())
        self.animation.setEndValue(value)
        self.animation.setEasingCurve(QEasingCurve.Type.InOutQuad)
        self.animation.start()

    # Property for animation
    def get_value(self):
        return super().value()

    def set_value(self, value):
        super().setValue(value)

    value_prop = pyqtProperty(int, get_value, set_value)


class SmoothScrollArea:
    """
    Helper to add smooth scrolling to scroll areas.
    """

    def __init__(self, scroll_area):
        self.scroll_area = scroll_area
        self.target_value = 0
        self.current_value = 0

        # Smooth scroll timer
        self.scroll_timer = QTimer()
        self.scroll_timer.timeout.connect(self._update_scroll)
        self.scroll_timer.start(16)  # ~60 FPS

    def scroll_to_bottom(self, smooth: bool = True):
        """
        Scroll to bottom of area.

        Args:
            smooth: Use smooth animation (True) or instant (False)
        """
        scrollbar = self.scroll_area.verticalScrollBar()
        self.target_value = scrollbar.maximum()

        if not smooth:
            scrollbar.setValue(self.target_value)
            self.current_value = self.target_value

    def _update_scroll(self):
        """Update smooth scroll animation."""
        scrollbar = self.scroll_area.verticalScrollBar()
        self.current_value = scrollbar.value()

        if abs(self.target_value - self.current_value) > 1:
            # Smooth interpolation
            diff = self.target_value - self.current_value
            step = diff * 0.2  # 20% per frame

            new_value = int(self.current_value + step)
            scrollbar.setValue(new_value)
        else:
            # Snap to target when close enough
            if self.target_value != self.current_value:
                scrollbar.setValue(self.target_value)


class ColorTransition:
    """
    Helper for smooth color transitions.
    Useful for consciousness-based color changes.
    """

    def __init__(self, widget, property_name: str = "color"):
        self.widget = widget
        self.property_name = property_name

        self.start_color = QColor("#ffffff")
        self.end_color = QColor("#ffffff")
        self.current_progress = 0.0

        # Transition timer
        self.transition_timer = QTimer()
        self.transition_timer.timeout.connect(self._update_transition)

    def transition_to(self, target_color: QColor, duration: int = 1000):
        """
        Transition to target color.

        Args:
            target_color: Target color
            duration: Transition duration in ms
        """
        # Get current color from widget stylesheet
        self.start_color = self._get_current_color()
        self.end_color = target_color
        self.current_progress = 0.0

        # Calculate step size
        self.step_size = 1.0 / (duration / 16)  # 60 FPS

        # Start timer
        if not self.transition_timer.isActive():
            self.transition_timer.start(16)

    def _update_transition(self):
        """Update color transition."""
        self.current_progress += self.step_size

        if self.current_progress >= 1.0:
            # Transition complete
            self.current_progress = 1.0
            self.transition_timer.stop()

        # Interpolate color
        current_color = self._interpolate_color(
            self.start_color,
            self.end_color,
            self.current_progress
        )

        # Apply to widget (would need to be customized per widget type)
        self._apply_color(current_color)

    def _interpolate_color(self, color1: QColor, color2: QColor, t: float) -> QColor:
        """Interpolate between two colors."""
        r = int(color1.red() + (color2.red() - color1.red()) * t)
        g = int(color1.green() + (color2.green() - color1.green()) * t)
        b = int(color1.blue() + (color2.blue() - color1.blue()) * t)
        a = int(color1.alpha() + (color2.alpha() - color1.alpha()) * t)

        return QColor(r, g, b, a)

    def _get_current_color(self) -> QColor:
        """Get current color from widget (placeholder)."""
        return self.start_color

    def _apply_color(self, color: QColor):
        """Apply color to widget (placeholder - override in subclass)."""
        pass
