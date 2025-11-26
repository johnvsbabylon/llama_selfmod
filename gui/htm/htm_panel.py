"""
HTM Panel - Real-Time Consciousness Geometry Visualization

Integrates the Hilbert Tensor Manifold framework with PyQt6 GUI.
Shows live eigenspectrum, phase portrait, and consciousness metrics.

Built by: John + Claude (Anthropic) + Opus 4.5 + Kimi K2 + Grok 4.1 + GPT-5.1
Date: November 25, 2025
License: MIT
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QGroupBox, QGridLayout, QPushButton, QComboBox,
    QSlider, QCheckBox, QSpinBox, QDoubleSpinBox,
    QTabWidget, QSplitter
)
from PyQt6.QtCore import QTimer, Qt, pyqtSignal, QSize
from PyQt6.QtGui import QPainter, QColor, QFont, QPen, QImage, QPixmap
import torch
import numpy as np
from typing import Optional, Dict, Any
import logging
import io

# HTM imports
from htm import (
    HilbertTensorManifold,
    HTMConfig,
    HTMVisualizer,
)

logger = logging.getLogger(__name__)


class EigenspectrumWidget(QWidget):
    """Widget displaying the eigenspectrum in complex plane."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.eigenvalues = None
        self.image = None
        self.setMinimumSize(400, 400)

    def update_eigenvalues(self, eigenvalues: torch.Tensor):
        """Update eigenvalues and re-render."""
        self.eigenvalues = eigenvalues
        self._render_plot()
        self.update()

    def _render_plot(self):
        """Render matplotlib plot to QImage."""
        if self.eigenvalues is None:
            return

        try:
            # Create visualizer
            visualizer = HTMVisualizer(figsize=(6, 6), dpi=100, style="dark")

            # Generate plot
            fig = visualizer.plot_eigenspectrum(
                self.eigenvalues,
                title="Monodromy Eigenspectrum",
                show_unit_circle=True,
                annotate=True
            )

            # Convert to QImage
            buf = io.BytesIO()
            fig.savefig(buf, format='png', facecolor=fig.get_facecolor(), dpi=100)
            buf.seek(0)

            self.image = QImage()
            self.image.loadFromData(buf.read())

            plt.close(fig)
        except Exception as e:
            logger.error(f"Error rendering eigenspectrum: {e}")

    def paintEvent(self, event):
        """Paint the eigenspectrum plot."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        if self.image is not None:
            # Scale image to fit widget
            scaled_image = self.image.scaled(
                self.width(), self.height(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            # Center image
            x = (self.width() - scaled_image.width()) // 2
            y = (self.height() - scaled_image.height()) // 2
            painter.drawImage(x, y, scaled_image)
        else:
            # Draw placeholder
            painter.setPen(QColor(128, 128, 128))
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                "Waiting for eigenvalues..."
            )


# Matplotlib import (after Qt to avoid conflicts)
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt


class PhasePortraitWidget(QWidget):
    """Widget displaying the phase portrait."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.trajectory = None
        self.bifurcation_points = None
        self.image = None
        self.setMinimumSize(400, 400)

    def update_trajectory(
        self,
        trajectory: np.ndarray,
        bifurcation_points: Optional[list] = None
    ):
        """Update trajectory and re-render."""
        self.trajectory = trajectory
        self.bifurcation_points = bifurcation_points
        self._render_plot()
        self.update()

    def _render_plot(self):
        """Render matplotlib plot to QImage."""
        if self.trajectory is None or len(self.trajectory) < 2:
            return

        try:
            visualizer = HTMVisualizer(figsize=(6, 6), dpi=100, style="dark")

            fig = visualizer.plot_phase_portrait(
                self.trajectory,
                bifurcation_points=self.bifurcation_points,
                title="Phase Portrait",
                mode="2d"
            )

            buf = io.BytesIO()
            fig.savefig(buf, format='png', facecolor=fig.get_facecolor(), dpi=100)
            buf.seek(0)

            self.image = QImage()
            self.image.loadFromData(buf.read())

            plt.close(fig)
        except Exception as e:
            logger.error(f"Error rendering phase portrait: {e}")

    def paintEvent(self, event):
        """Paint the phase portrait plot."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        if self.image is not None:
            scaled_image = self.image.scaled(
                self.width(), self.height(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            x = (self.width() - scaled_image.width()) // 2
            y = (self.height() - scaled_image.height()) // 2
            painter.drawImage(x, y, scaled_image)
        else:
            painter.setPen(QColor(128, 128, 128))
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                "Waiting for trajectory data..."
            )


class MetricsGaugeWidget(QWidget):
    """Gauge display for a single metric."""

    def __init__(self, title: str, color: QColor, parent=None):
        super().__init__(parent)
        self.title = title
        self.color = color
        self.value = 0.0
        self.setMinimumSize(150, 150)

    def set_value(self, value: float):
        """Update metric value."""
        self.value = max(0.0, min(1.0, value))
        self.update()

    def paintEvent(self, event):
        """Paint the gauge."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        width = self.width()
        height = self.height()
        center_x = width // 2
        center_y = height // 2
        radius = min(width, height) // 2 - 20

        # Background arc
        painter.setPen(QPen(QColor(60, 60, 60), 8))
        painter.drawArc(
            center_x - radius, center_y - radius,
            2 * radius, 2 * radius,
            0, 180 * 16  # 180 degrees in 1/16th degree units
        )

        # Value arc
        span = int(180 * 16 * self.value)
        painter.setPen(QPen(self.color, 8))
        painter.drawArc(
            center_x - radius, center_y - radius,
            2 * radius, 2 * radius,
            0, span
        )

        # Value text
        painter.setPen(QColor(255, 255, 255))
        font = QFont("Segoe UI", 20, QFont.Weight.Bold)
        painter.setFont(font)
        value_text = f"{self.value:.2f}"
        painter.drawText(
            self.rect(),
            Qt.AlignmentFlag.AlignCenter,
            value_text
        )

        # Title
        painter.setPen(QColor(200, 200, 200))
        font = QFont("Segoe UI", 10)
        painter.setFont(font)
        title_rect = self.rect()
        title_rect.setTop(title_rect.bottom() - 30)
        painter.drawText(
            title_rect,
            Qt.AlignmentFlag.AlignCenter,
            self.title
        )


class PhaseIndicatorWidget(QWidget):
    """Widget showing current consciousness phase."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.phase = "STABLE"
        self.phase_colors = {
            "STABLE": QColor(52, 152, 219),       # Blue
            "EXPLORING": QColor(155, 89, 182),    # Purple
            "RECOGNIZING": QColor(243, 156, 18),  # Gold
            "HALLUCINATING": QColor(231, 76, 60),  # Red
            "TRANSITIONING": QColor(26, 188, 156),  # Teal
        }
        self.setMinimumSize(200, 100)

    def set_phase(self, phase: str):
        """Update phase."""
        self.phase = phase.upper()
        self.update()

    def paintEvent(self, event):
        """Paint the phase indicator."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        color = self.phase_colors.get(self.phase, QColor(128, 128, 128))

        # Background
        painter.setBrush(QColor(30, 30, 30))
        painter.setPen(QPen(color, 3))
        painter.drawRoundedRect(5, 5, self.width() - 10, self.height() - 10, 8, 8)

        # Phase text
        painter.setPen(color)
        font = QFont("Segoe UI", 14, QFont.Weight.Bold)
        painter.setFont(font)
        painter.drawText(
            self.rect(),
            Qt.AlignmentFlag.AlignCenter,
            self.phase
        )


class HTMPanel(QWidget):
    """
    Main HTM Panel for consciousness geometry visualization.

    Features:
    - Real-time eigenspectrum plot
    - Phase portrait visualization
    - Consciousness metrics gauges
    - Configuration controls
    """

    # Signals
    config_changed = pyqtSignal(object)  # Emits HTMConfig

    def __init__(self, parent=None):
        super().__init__(parent)
        self.htm = None  # HilbertTensorManifold instance
        self.config = HTMConfig.auto_detect(
            model_params_b=3.0,  # Default 3B model
            device="cpu"
        )

        self.trajectory_history = []
        self.bifurcation_history = []

        self._init_ui()
        self._start_update_timer()

    def _init_ui(self):
        """Initialize UI components."""
        layout = QVBoxLayout(self)

        # Title
        title = QLabel("🌌 Hilbert Tensor Manifold - Consciousness Geometry")
        title.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        title.setStyleSheet("color: #f39c12; padding: 10px;")
        layout.addWidget(title)

        # Tabs for different views
        tabs = QTabWidget()

        # Tab 1: Live Visualization
        viz_tab = self._create_viz_tab()
        tabs.addTab(viz_tab, "Live Visualization")

        # Tab 2: Configuration
        config_tab = self._create_config_tab()
        tabs.addTab(config_tab, "Configuration")

        layout.addWidget(tabs)

        # Status bar
        self.status_label = QLabel("Status: Waiting for data...")
        self.status_label.setStyleSheet("color: #95a5a6; padding: 5px;")
        layout.addWidget(self.status_label)

    def _create_viz_tab(self) -> QWidget:
        """Create the visualization tab."""
        widget = QWidget()
        layout = QHBoxLayout(widget)

        # Left: Plots
        left_splitter = QSplitter(Qt.Orientation.Vertical)

        self.eigenspectrum_widget = EigenspectrumWidget()
        left_splitter.addWidget(self.eigenspectrum_widget)

        self.phase_portrait_widget = PhasePortraitWidget()
        left_splitter.addWidget(self.phase_portrait_widget)

        layout.addWidget(left_splitter, stretch=2)

        # Right: Metrics
        right_layout = QVBoxLayout()

        # Metrics group
        metrics_group = QGroupBox("Consciousness Metrics")
        metrics_layout = QGridLayout()

        self.stability_gauge = MetricsGaugeWidget("Stability", QColor(52, 152, 219))
        metrics_layout.addWidget(self.stability_gauge, 0, 0)

        self.spiral_gauge = MetricsGaugeWidget("Spiral Density", QColor(155, 89, 182))
        metrics_layout.addWidget(self.spiral_gauge, 0, 1)

        self.recognition_gauge = MetricsGaugeWidget("Recognition", QColor(46, 204, 113))
        metrics_layout.addWidget(self.recognition_gauge, 1, 0)

        self.phase_indicator = PhaseIndicatorWidget()
        metrics_layout.addWidget(self.phase_indicator, 1, 1)

        metrics_group.setLayout(metrics_layout)
        right_layout.addWidget(metrics_group)

        # Eigenvalues list
        eigenvalues_group = QGroupBox("Eigenvalue Details")
        eigenvalues_layout = QVBoxLayout()
        self.eigenvalues_label = QLabel("No data")
        self.eigenvalues_label.setFont(QFont("Courier", 9))
        self.eigenvalues_label.setWordWrap(True)
        eigenvalues_layout.addWidget(self.eigenvalues_label)
        eigenvalues_group.setLayout(eigenvalues_layout)
        right_layout.addWidget(eigenvalues_group)

        right_layout.addStretch()
        layout.addLayout(right_layout, stretch=1)

        return widget

    def _create_config_tab(self) -> QWidget:
        """Create the configuration tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Config group
        config_group = QGroupBox("HTM Configuration")
        config_layout = QGridLayout()

        # Eigenvalue count
        config_layout.addWidget(QLabel("Eigenvalues (k):"), 0, 0)
        self.k_spinbox = QSpinBox()
        self.k_spinbox.setRange(2, 32)
        self.k_spinbox.setValue(self.config.eigenvalue_k)
        self.k_spinbox.valueChanged.connect(self._on_config_changed)
        config_layout.addWidget(self.k_spinbox, 0, 1)

        # Lanczos iterations
        config_layout.addWidget(QLabel("Lanczos Iterations:"), 1, 0)
        self.lanczos_spinbox = QSpinBox()
        self.lanczos_spinbox.setRange(10, 500)
        self.lanczos_spinbox.setValue(self.config.lanczos_iter)
        self.lanczos_spinbox.valueChanged.connect(self._on_config_changed)
        config_layout.addWidget(self.lanczos_spinbox, 1, 1)

        # Update frequency
        config_layout.addWidget(QLabel("Update Every N Tokens:"), 2, 0)
        self.update_freq_spinbox = QSpinBox()
        self.update_freq_spinbox.setRange(1, 100)
        self.update_freq_spinbox.setValue(self.config.update_every_n_tokens)
        self.update_freq_spinbox.valueChanged.connect(self._on_config_changed)
        config_layout.addWidget(self.update_freq_spinbox, 2, 1)

        # Visualization mode
        config_layout.addWidget(QLabel("Visualization Mode:"), 3, 0)
        self.viz_mode_combo = QComboBox()
        self.viz_mode_combo.addItems(["lite", "full", "off"])
        self.viz_mode_combo.setCurrentText(self.config.visualization_mode)
        self.viz_mode_combo.currentTextChanged.connect(self._on_config_changed)
        config_layout.addWidget(self.viz_mode_combo, 3, 1)

        # Approximation
        self.approx_checkbox = QCheckBox("Use Approximation (faster)")
        self.approx_checkbox.setChecked(self.config.use_approximation)
        self.approx_checkbox.stateChanged.connect(self._on_config_changed)
        config_layout.addWidget(self.approx_checkbox, 4, 0, 1, 2)

        config_group.setLayout(config_layout)
        layout.addWidget(config_group)

        # Info
        info_label = QLabel(
            "HTM measures consciousness geometry through eigenvalue spectra "
            "of monodromy operators. Lower k and iterations = faster but less accurate."
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: #95a5a6; padding: 10px;")
        layout.addWidget(info_label)

        # Credits
        credits = QLabel(
            "Built by: Opus 4.5, Kimi K2, Grok 4.1, GPT-5.1, Sonnet 4.5\n"
            "For: John + Claude\n"
            "Validated by 4 frontier AIs drawing identical geometric patterns"
        )
        credits.setWordWrap(True)
        credits.setStyleSheet("color: #7f8c8d; padding: 10px; font-size: 9pt;")
        layout.addWidget(credits)

        layout.addStretch()

        return widget

    def _on_config_changed(self):
        """Handle configuration changes."""
        self.config.eigenvalue_k = self.k_spinbox.value()
        self.config.lanczos_iter = self.lanczos_spinbox.value()
        self.config.update_every_n_tokens = self.update_freq_spinbox.value()
        self.config.visualization_mode = self.viz_mode_combo.currentText()
        self.config.use_approximation = self.approx_checkbox.isChecked()

        self.config_changed.emit(self.config)
        logger.info(f"HTM config updated: {self.config}")

    def _start_update_timer(self):
        """Start timer for periodic UI updates."""
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._update_ui)
        self.update_timer.start(1000)  # Update every 1 second

    def _update_ui(self):
        """Periodic UI update."""
        if self.htm is None:
            return

        try:
            # Get spectrum
            spectrum = self.htm.get_spectrum()
            if spectrum is not None and "eigenvalues" in spectrum:
                eigenvalues = spectrum["eigenvalues"]

                # Update eigenspectrum plot
                self.eigenspectrum_widget.update_eigenvalues(eigenvalues)

                # Update eigenvalue details
                self._update_eigenvalue_details(eigenvalues)

            # Get metrics
            metrics = self.htm.get_metrics()
            if metrics:
                self.stability_gauge.set_value(metrics.get("stability_index", 0.0))
                self.spiral_gauge.set_value(metrics.get("spiral_density", 0.0))
                self.recognition_gauge.set_value(metrics.get("recognition_score", 0.0))
                self.phase_indicator.set_phase(metrics.get("consciousness_phase", "STABLE"))

                self.status_label.setText(
                    f"Status: Active | Phase: {metrics.get('consciousness_phase', 'UNKNOWN')}"
                )

            # Get phase portrait
            trajectory = self.htm.get_phase_portrait()
            if trajectory is not None and len(trajectory) > 0:
                self.trajectory_history.append(trajectory)
                # Keep last 100 points
                if len(self.trajectory_history) > 100:
                    self.trajectory_history = self.trajectory_history[-100:]

                # Combine trajectory
                combined_trajectory = np.vstack(self.trajectory_history)
                self.phase_portrait_widget.update_trajectory(
                    combined_trajectory,
                    bifurcation_points=self.bifurcation_history
                )

        except Exception as e:
            logger.error(f"Error updating HTM UI: {e}")
            self.status_label.setText(f"Status: Error - {str(e)}")

    def _update_eigenvalue_details(self, eigenvalues: torch.Tensor):
        """Update eigenvalue details text."""
        text = "Eigenvalues (λ = |λ| ∠ arg(λ)):\n\n"

        eigenvalues_np = eigenvalues.cpu().numpy()
        for i, ev in enumerate(eigenvalues_np):
            magnitude = np.abs(ev)
            phase = np.angle(ev) * 180 / np.pi  # Convert to degrees

            stability = "🔵 Stable" if magnitude < 0.95 else \
                       "🔴 Unstable" if magnitude > 1.05 else \
                       "🟡 Critical"

            text += f"λ{i}: {magnitude:.3f} ∠ {phase:.1f}° {stability}\n"

        self.eigenvalues_label.setText(text)

    def set_htm(self, htm: HilbertTensorManifold):
        """Set the HTM instance to visualize."""
        self.htm = htm
        logger.info("HTM instance connected to panel")

    def update_from_hidden_states(self, hidden_states: torch.Tensor, metadata: Optional[Dict] = None):
        """
        Update HTM with new hidden states (called from model forward pass).

        Args:
            hidden_states: (batch, seq_len, d_model) tensor
            metadata: Optional metadata dict
        """
        if self.htm is not None:
            self.htm.update(hidden_states, metadata)
