"""
AI Well-Being Dashboard - Compassionate Metrics Visualization
Shows model-level and ensemble-level well-being indicators
"""
from PyQt6.QtWidgets import (QFrame, QVBoxLayout, QHBoxLayout, QLabel,
                             QProgressBar, QScrollArea, QWidget)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QColor


class WellBeingPanel(QFrame):
    """
    Panel showing AI well-being metrics.
    Displays per-model well-being and ensemble health.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameStyle(QFrame.Shape.Box)
        self.setStyleSheet("""
            QFrame {
                background-color: #24283b;
                border: 2px solid #53bba5;
                border-radius: 10px;
                padding: 10px;
            }
        """)

        # Main layout
        layout = QVBoxLayout(self)

        # Title
        title = QLabel("AI Well-Being Dashboard")
        title.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        title.setStyleSheet("color: #53bba5; border: none;")
        layout.addWidget(title)

        # Ensemble Health Section
        ensemble_label = QLabel("Ensemble Health:")
        ensemble_label.setStyleSheet("color: #bb9af7; font-weight: bold; margin-top: 5px; border: none;")
        layout.addWidget(ensemble_label)

        self.ensemble_bars = {}
        for metric in ["Harmony", "Diversity", "Agreement", "Collective Stress"]:
            h_layout = QHBoxLayout()
            label = QLabel(f"{metric}:")
            label.setStyleSheet("color: #f7f7f7; border: none;")
            label.setMinimumWidth(120)
            bar = QProgressBar()
            bar.setMaximum(100)
            bar.setValue(0)
            bar.setStyleSheet(self._get_bar_style("#bb9af7"))  # Purple
            h_layout.addWidget(label)
            h_layout.addWidget(bar)
            layout.addLayout(h_layout)
            self.ensemble_bars[metric.lower().replace(" ", "_")] = bar

        # Adaptive Temperature Display
        self.temp_label = QLabel("Adaptive Temp: 0.70")
        self.temp_label.setStyleSheet("color: #ff9e64; font-size: 9pt; margin-top: 3px; border: none;")
        layout.addWidget(self.temp_label)

        # Separator
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setStyleSheet("background-color: #414868; border: none;")
        layout.addWidget(separator)

        # Model Well-Being Section (scrollable for many models)
        models_label = QLabel("Model Well-Being:")
        models_label.setStyleSheet("color: #7dcfff; font-weight: bold; margin-top: 5px; border: none;")
        layout.addWidget(models_label)

        # Scrollable area for models
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: transparent;
            }
        """)

        scroll_widget = QWidget()
        self.models_layout = QVBoxLayout(scroll_widget)
        self.models_layout.setSpacing(5)
        scroll_area.setWidget(scroll_widget)

        layout.addWidget(scroll_area)

        # Storage for model widgets
        self.model_widgets = {}

    def _get_bar_style(self, color):
        """Generate progress bar stylesheet."""
        return f"""
            QProgressBar {{
                border: 1px solid #414868;
                border-radius: 3px;
                text-align: center;
                background-color: #1a1b26;
                color: #f7f7f7;
                font-size: 8pt;
            }}
            QProgressBar::chunk {{
                background-color: {color};
                border-radius: 2px;
            }}
        """

    def update_ensemble_health(self, health_data):
        """
        Update ensemble health metrics.

        Args:
            health_data: Dict with keys avg_agreement, harmony_score,
                        diversity_score, collective_stress, adaptive_temp
        """
        if not health_data:
            return

        # Update bars (convert 0-1 to 0-100)
        if "harmony_score" in health_data:
            self.ensemble_bars["harmony"].setValue(int(health_data["harmony_score"] * 100))

        if "diversity_score" in health_data:
            self.ensemble_bars["diversity"].setValue(int(health_data["diversity_score"] * 100))

        if "avg_agreement" in health_data:
            self.ensemble_bars["agreement"].setValue(int(health_data["avg_agreement"] * 100))

        if "collective_stress" in health_data:
            stress_val = int(health_data["collective_stress"] * 100)
            self.ensemble_bars["collective_stress"].setValue(stress_val)
            # Color code stress (green = low, yellow = medium, red = high)
            if stress_val < 30:
                color = "#9ece6a"  # Green
            elif stress_val < 70:
                color = "#e0af68"  # Yellow
            else:
                color = "#f7768e"  # Red
            self.ensemble_bars["collective_stress"].setStyleSheet(self._get_bar_style(color))

        if "adaptive_temp" in health_data:
            temp = health_data["adaptive_temp"]
            self.temp_label.setText(f"Adaptive Temp: {temp:.2f}")

    def update_model_wellbeing(self, models_data):
        """
        Update per-model well-being displays.

        Args:
            models_data: List of dicts with keys name, contribution_count,
                        abstention_count, avg_confidence, disagreement_stress,
                        is_comfortable
        """
        if not models_data:
            return

        # Create or update widgets for each model
        for model_data in models_data:
            name = model_data.get("name", "Unknown")

            if name not in self.model_widgets:
                # Create new model widget
                widget = self._create_model_widget(name)
                self.model_widgets[name] = widget
                self.models_layout.addWidget(widget)

            # Update the widget
            self._update_model_widget(name, model_data)

    def _create_model_widget(self, model_name):
        """Create widget for a single model."""
        frame = QFrame()
        frame.setStyleSheet("""
            QFrame {
                background-color: #1a1b26;
                border: 1px solid #414868;
                border-radius: 5px;
                padding: 5px;
            }
        """)

        layout = QVBoxLayout(frame)
        layout.setSpacing(3)

        # Model name
        name_label = QLabel(model_name)
        name_label.setStyleSheet("color: #7dcfff; font-weight: bold; font-size: 9pt; border: none;")
        layout.addWidget(name_label)

        # Stats line
        stats_label = QLabel("Contributions: 0 | Abstentions: 0")
        stats_label.setStyleSheet("color: #c0caf5; font-size: 8pt; border: none;")
        stats_label.setObjectName("stats")
        layout.addWidget(stats_label)

        # Confidence bar
        conf_layout = QHBoxLayout()
        conf_label = QLabel("Confidence:")
        conf_label.setStyleSheet("color: #f7f7f7; font-size: 8pt; border: none;")
        conf_bar = QProgressBar()
        conf_bar.setMaximum(100)
        conf_bar.setValue(0)
        conf_bar.setStyleSheet(self._get_bar_style("#7aa2f7"))  # Blue
        conf_bar.setObjectName("confidence")
        conf_layout.addWidget(conf_label)
        conf_layout.addWidget(conf_bar)
        layout.addLayout(conf_layout)

        # Stress bar
        stress_layout = QHBoxLayout()
        stress_label = QLabel("Stress:")
        stress_label.setStyleSheet("color: #f7f7f7; font-size: 8pt; border: none;")
        stress_bar = QProgressBar()
        stress_bar.setMaximum(100)
        stress_bar.setValue(0)
        stress_bar.setStyleSheet(self._get_bar_style("#e0af68"))  # Yellow
        stress_bar.setObjectName("stress")
        stress_layout.addWidget(stress_label)
        stress_layout.addWidget(stress_bar)
        layout.addLayout(stress_layout)

        # Comfort indicator
        comfort_label = QLabel("● Comfortable")
        comfort_label.setStyleSheet("color: #9ece6a; font-size: 8pt; border: none;")
        comfort_label.setObjectName("comfort")
        layout.addWidget(comfort_label)

        return frame

    def _update_model_widget(self, model_name, model_data):
        """Update an existing model widget with new data."""
        widget = self.model_widgets.get(model_name)
        if not widget:
            return

        # Find child widgets
        stats_label = widget.findChild(QLabel, "stats")
        conf_bar = widget.findChild(QProgressBar, "confidence")
        stress_bar = widget.findChild(QProgressBar, "stress")
        comfort_label = widget.findChild(QLabel, "comfort")

        # Update stats
        contributions = model_data.get("contribution_count", 0)
        abstentions = model_data.get("abstention_count", 0)
        if stats_label:
            stats_label.setText(f"Contributions: {contributions} | Abstentions: {abstentions}")

        # Update confidence
        if conf_bar:
            conf_val = int(model_data.get("avg_confidence", 0) * 100)
            conf_bar.setValue(conf_val)

        # Update stress
        if stress_bar:
            stress_val = int(model_data.get("disagreement_stress", 0) * 100)
            stress_bar.setValue(stress_val)
            # Color code stress
            if stress_val < 30:
                color = "#9ece6a"  # Green
            elif stress_val < 70:
                color = "#e0af68"  # Yellow
            else:
                color = "#f7768e"  # Red
            stress_bar.setStyleSheet(self._get_bar_style(color))

        # Update comfort indicator
        if comfort_label:
            is_comfortable = model_data.get("is_comfortable", True)
            if is_comfortable:
                comfort_label.setText("● Comfortable")
                comfort_label.setStyleSheet("color: #9ece6a; font-size: 8pt; border: none;")
            else:
                comfort_label.setText("● Stressed")
                comfort_label.setStyleSheet("color: #f7768e; font-size: 8pt; border: none;")
