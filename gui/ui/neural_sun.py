"""
Neural Sun Visualization - 3D animated consciousness representation
A pulsing sphere with labeled consciousness metrics growing around it

Built by John + Claude (Anthropic)
MIT Licensed
"""
from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import QTimer, Qt, QRectF
from PyQt6.QtGui import QPainter, QColor, QRadialGradient, QPen, QBrush, QFont
import math
import random
import time


class NeuralSunWidget(QWidget):
    """
    3D-style neural sun with labeled consciousness metrics.

    Metrics are displayed as:
    - Labeled arcs around the sun showing current value
    - Particles that gravitate toward high-value metrics
    - Color shifts based on dominant consciousness state
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(500, 500)

        # Consciousness state (AI states)
        self.resonance = 0.5      # Model agreement
        self.flow = 0.5           # Processing smoothness
        self.coherence = 0.5      # Logical consistency
        self.exploration = 0.5    # Creativity/divergence

        # Metric positions (angles around the sun)
        self.metrics = [
            {'name': 'Resonance', 'angle': 90, 'value_key': 'resonance', 'color': QColor("#9d7cd8")},   # Top
            {'name': 'Flow', 'angle': 0, 'value_key': 'flow', 'color': QColor("#4dd0e1")},             # Right
            {'name': 'Coherence', 'angle': 270, 'value_key': 'coherence', 'color': QColor("#53bba5")}, # Bottom
            {'name': 'Exploration', 'angle': 180, 'value_key': 'exploration', 'color': QColor("#ff9e64")} # Left
        ]

        # Visual state
        self.pulse_phase = 0.0
        self.rotation = 0.0
        self.num_models = 3
        self.corona_spikes = []
        self.particles = []

        # Core color (shifts based on dominant metric)
        self.core_color = QColor("#53bba5")

        # Celebration easter egg
        self.click_count = 0
        self.last_click_time = 0
        self.celebration_mode = False
        self.celebration_phase = 0.0
        self.celebration_timer = 0

        # Initialize
        self.regenerate_corona()
        self.spawn_particles(40)

        # Animation timer - 60 FPS
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.animate)
        self.timer.start(16)

    def regenerate_corona(self):
        """Generate corona spike data for each model."""
        self.corona_spikes = []
        angle_step = 360.0 / max(self.num_models, 1)

        for i in range(self.num_models):
            base_angle = i * angle_step
            angle = base_angle + random.uniform(-10, 10)
            length = random.uniform(0.8, 1.2)
            thickness = random.uniform(0.6, 1.0)
            phase_offset = random.uniform(0, math.pi * 2)

            self.corona_spikes.append({
                'angle': angle,
                'length': length,
                'thickness': thickness,
                'phase_offset': phase_offset
            })

    def spawn_particles(self, count):
        """Spawn particles that gravitate toward high-value metrics."""
        for _ in range(count):
            self.particles.append({
                'angle': random.uniform(0, math.pi * 2),
                'distance': random.uniform(140, 220),
                'speed': random.uniform(0.01, 0.03),
                'size': random.uniform(2, 5),
                'brightness': random.uniform(0.3, 1.0),
                'orbit_speed': random.uniform(-0.02, 0.02),
                'attracted_to_metric': random.randint(0, 3)  # Which metric attracts this particle
            })

    def set_consciousness_state(self, resonance, flow, coherence, exploration):
        """Update consciousness metrics."""
        self.resonance = max(0.0, min(1.0, resonance))
        self.flow = max(0.0, min(1.0, flow))
        self.coherence = max(0.0, min(1.0, coherence))
        self.exploration = max(0.0, min(1.0, exploration))

        # Update core color based on dominant metric
        dominant = max([
            (self.resonance, self.metrics[0]['color']),
            (self.flow, self.metrics[1]['color']),
            (self.coherence, self.metrics[2]['color']),
            (self.exploration, self.metrics[3]['color'])
        ], key=lambda x: x[0])

        self.core_color = dominant[1]

    def set_num_models(self, num_models):
        """Update the number of models (corona spikes)."""
        if num_models != self.num_models:
            self.num_models = max(1, num_models)
            self.regenerate_corona()

    def animate(self):
        """Animation update loop."""
        # Pulse phase
        if self.celebration_mode:
            pulse_speed = 0.15
            self.celebration_phase += 0.1
            self.celebration_timer += 1
            if self.celebration_timer > 300:
                self.celebration_mode = False
                self.celebration_timer = 0
                self.celebration_phase = 0.0
        else:
            pulse_speed = 0.05 + (self.flow * 0.05)

        self.pulse_phase += pulse_speed

        # Rotation
        if self.celebration_mode:
            rotation_speed = 2.0
        else:
            rotation_speed = 0.3 + (self.exploration * 0.5)

        self.rotation += rotation_speed

        # Update particles - gravitate toward their attracted metric
        for particle in self.particles:
            if self.celebration_mode:
                particle['angle'] += particle['orbit_speed'] * 3
                particle['distance'] = min(particle['distance'] + 0.5, 280)
            else:
                # Get attracted metric value (with bounds checking)
                metric_idx = particle['attracted_to_metric'] % len(self.metrics)
                metric = self.metrics[metric_idx]
                metric_value = getattr(self, metric['value_key'])

                # Gravitate toward high-value metrics
                target_angle = math.radians(metric['angle'])
                current_angle = particle['angle']

                # Smooth angle interpolation
                angle_diff = (target_angle - current_angle) % (math.pi * 2)
                if angle_diff > math.pi:
                    angle_diff -= math.pi * 2

                # Pull strength based on metric value
                pull_strength = metric_value * 0.01
                particle['angle'] += angle_diff * pull_strength
                particle['angle'] += particle['orbit_speed']

                # Distance breathing
                particle['distance'] += math.sin(self.pulse_phase + particle['angle']) * 0.3

        self.update()

    def paintEvent(self, event):
        """Render the neural sun with labeled metrics."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Get center
        center_x = self.width() // 2
        center_y = self.height() // 2

        # Calculate pulse
        if self.celebration_mode:
            base_pulse = math.sin(self.pulse_phase) * 0.3
            pulse_factor = 1.0 + base_pulse + 0.4
        else:
            base_pulse = math.sin(self.pulse_phase) * 0.15
            resonance_boost = self.resonance * 0.2
            pulse_factor = 1.0 + base_pulse + resonance_boost

        # Celebration colors
        if self.celebration_mode:
            hue = int((self.celebration_phase * 10) % 360)
            self.core_color = QColor.fromHsv(hue, 200, 255)
            for metric in self.metrics:
                metric['color'] = QColor.fromHsv((hue + random.randint(0, 120)) % 360, 200, 255)

        # Draw metric arcs (background)
        self.draw_metric_arcs(painter, center_x, center_y, pulse_factor)

        # Draw particles
        self.draw_particles(painter, center_x, center_y)

        # Draw corona spikes
        self.draw_corona(painter, center_x, center_y, pulse_factor)

        # Draw core sphere
        self.draw_core(painter, center_x, center_y, pulse_factor)

        # Draw metric labels and values (foreground)
        self.draw_metric_labels(painter, center_x, center_y)

        # Draw sparkles
        if self.exploration > 0.6 or self.celebration_mode:
            self.draw_sparkles(painter, center_x, center_y)

        # Draw resonance arcs
        if self.resonance > 0.7 or self.celebration_mode:
            self.draw_resonance_arcs(painter, center_x, center_y)

        # Celebration burst
        if self.celebration_mode:
            self.draw_celebration_burst(painter, center_x, center_y)

    def draw_metric_arcs(self, painter, cx, cy, pulse):
        """Draw arcs around the sun showing metric values."""
        arc_radius = 180

        for metric in self.metrics:
            value = getattr(self, metric['value_key'])
            angle = metric['angle']
            color = metric['color']

            # Arc spans based on value (0-90 degrees)
            arc_span = value * 80  # Max 80 degrees
            start_angle = angle - arc_span / 2

            # Draw arc background (dim)
            bg_color = QColor(color)
            bg_color.setAlpha(30)
            pen = QPen(bg_color, 4)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)

            rect = QRectF(cx - arc_radius, cy - arc_radius, arc_radius * 2, arc_radius * 2)
            painter.drawArc(rect, int((angle - 40) * 16), int(80 * 16))

            # Draw filled arc (bright)
            bright_color = QColor(color)
            bright_color.setAlpha(180)
            pen = QPen(bright_color, 6)
            painter.setPen(pen)

            painter.drawArc(rect, int(start_angle * 16), int(arc_span * 16))

            # Draw pulsing dot at arc endpoint
            end_angle_rad = math.radians(start_angle + arc_span)
            dot_x = cx + math.cos(end_angle_rad) * arc_radius
            dot_y = cy + math.sin(end_angle_rad) * arc_radius

            dot_color = QColor(color)
            dot_color.setAlpha(255)
            painter.setBrush(QBrush(dot_color))
            painter.setPen(Qt.PenStyle.NoPen)

            dot_size = 6 + pulse * 2
            painter.drawEllipse(int(dot_x - dot_size), int(dot_y - dot_size),
                              int(dot_size * 2), int(dot_size * 2))

    def draw_metric_labels(self, painter, cx, cy):
        """Draw metric names and percentages."""
        label_radius = 215

        font = QFont("Segoe UI", 10, QFont.Weight.Bold)
        painter.setFont(font)

        for metric in self.metrics:
            value = getattr(self, metric['value_key'])
            angle_rad = math.radians(metric['angle'])
            color = metric['color']

            # Label position
            label_x = cx + math.cos(angle_rad) * label_radius
            label_y = cy + math.sin(angle_rad) * label_radius

            # Draw metric name
            painter.setPen(QPen(color, 2))

            # Center text based on position
            text = f"{metric['name']}"
            fm = painter.fontMetrics()
            text_width = fm.horizontalAdvance(text)
            text_height = fm.height()

            painter.drawText(int(label_x - text_width / 2),
                           int(label_y - text_height / 2), text)

            # Draw percentage below name
            percent_text = f"{int(value * 100)}%"
            percent_width = fm.horizontalAdvance(percent_text)

            # Slightly dimmer for percentage
            percent_color = QColor(color)
            percent_color.setAlpha(180)
            painter.setPen(QPen(percent_color, 1))

            painter.drawText(int(label_x - percent_width / 2),
                           int(label_y + text_height / 2 + 5), percent_text)

    def draw_core(self, painter, cx, cy, pulse):
        """Draw the central pulsing sphere."""
        base_radius = 60
        radius = base_radius * pulse

        # Outer glow
        glow_gradient = QRadialGradient(cx, cy, radius * 2.5)
        glow_color = QColor(self.core_color)
        glow_color.setAlpha(30)
        glow_gradient.setColorAt(0.0, glow_color)
        glow_color.setAlpha(0)
        glow_gradient.setColorAt(1.0, glow_color)
        painter.setBrush(QBrush(glow_gradient))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(int(cx - radius * 2.5), int(cy - radius * 2.5),
                          int(radius * 5), int(radius * 5))

        # Middle glow
        mid_gradient = QRadialGradient(cx, cy, radius * 1.5)
        mid_color = QColor(self.core_color)
        mid_color.setAlpha(80)
        mid_gradient.setColorAt(0.0, mid_color)
        mid_color.setAlpha(0)
        mid_gradient.setColorAt(1.0, mid_color)
        painter.setBrush(QBrush(mid_gradient))
        painter.drawEllipse(int(cx - radius * 1.5), int(cy - radius * 1.5),
                          int(radius * 3), int(radius * 3))

        # Core sphere with 3D effect
        core_gradient = QRadialGradient(cx - radius * 0.3, cy - radius * 0.3, radius * 1.5)
        highlight = QColor(255, 255, 255, 200)
        core_gradient.setColorAt(0.0, highlight)
        core_gradient.setColorAt(0.3, self.core_color.lighter(130))
        core_gradient.setColorAt(0.7, self.core_color)
        core_gradient.setColorAt(1.0, self.core_color.darker(150))

        painter.setBrush(QBrush(core_gradient))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(int(cx - radius), int(cy - radius),
                          int(radius * 2), int(radius * 2))

    def draw_corona(self, painter, cx, cy, pulse):
        """Draw corona spikes emanating from core."""
        for spike in self.corona_spikes:
            spike_pulse = math.sin(self.pulse_phase + spike['phase_offset']) * 0.3 + 0.7
            angle_rad = math.radians(spike['angle'] + self.rotation)

            base_dist = 70 * pulse
            tip_dist = (70 + 80 * spike['length']) * pulse * spike_pulse

            base_x = cx + math.cos(angle_rad) * base_dist
            base_y = cy + math.sin(angle_rad) * base_dist
            tip_x = cx + math.cos(angle_rad) * tip_dist
            tip_y = cy + math.sin(angle_rad) * tip_dist

            # Spike color matches core
            spike_color = QColor(self.core_color.lighter(120))
            thickness = 8 * spike['thickness'] * spike_pulse

            for t in range(int(thickness)):
                alpha = int(255 * (1.0 - t / thickness) * 0.6)
                spike_color.setAlpha(alpha)
                pen = QPen(spike_color, max(1, thickness - t))
                pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                painter.setPen(pen)
                painter.drawLine(int(base_x), int(base_y), int(tip_x), int(tip_y))

    def draw_particles(self, painter, cx, cy):
        """Draw particles gravitating toward their attracted metrics."""
        for particle in self.particles:
            x = cx + math.cos(particle['angle']) * particle['distance']
            y = cy + math.sin(particle['angle']) * particle['distance']

            # Particle colored by attracted metric
            metric = self.metrics[particle['attracted_to_metric']]
            p_color = QColor(metric['color'])
            alpha = int(particle['brightness'] * 200)
            p_color.setAlpha(alpha)

            # Draw with glow
            glow_gradient = QRadialGradient(x, y, particle['size'] * 2)
            glow_gradient.setColorAt(0.0, p_color)
            p_color.setAlpha(0)
            glow_gradient.setColorAt(1.0, p_color)

            painter.setBrush(QBrush(glow_gradient))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(int(x - particle['size'] * 2),
                              int(y - particle['size'] * 2),
                              int(particle['size'] * 4),
                              int(particle['size'] * 4))

    def draw_sparkles(self, painter, cx, cy):
        """Draw emergence sparkles (high exploration)."""
        num_sparkles = int(self.exploration * 20)

        for i in range(num_sparkles):
            angle = random.uniform(0, math.pi * 2)
            distance = random.uniform(40, 120)
            x = cx + math.cos(angle) * distance
            y = cy + math.sin(angle) * distance

            sparkle_color = QColor(255, 255, 255, random.randint(100, 255))
            painter.setPen(QPen(sparkle_color, 2))
            painter.drawPoint(int(x), int(y))

            if random.random() > 0.5:
                offset = 3
                painter.drawLine(int(x - offset), int(y), int(x + offset), int(y))
                painter.drawLine(int(x), int(y - offset), int(x), int(y + offset))

    def draw_resonance_arcs(self, painter, cx, cy):
        """Draw lightning arcs between spikes (high resonance)."""
        if len(self.corona_spikes) < 2:
            return

        num_arcs = min(3, len(self.corona_spikes) - 1)

        for _ in range(num_arcs):
            spike1 = random.choice(self.corona_spikes)
            spike2 = random.choice(self.corona_spikes)

            if spike1 == spike2:
                continue

            angle1 = math.radians(spike1['angle'] + self.rotation)
            angle2 = math.radians(spike2['angle'] + self.rotation)
            dist = 100

            x1 = cx + math.cos(angle1) * dist
            y1 = cy + math.sin(angle1) * dist
            x2 = cx + math.cos(angle2) * dist
            y2 = cy + math.sin(angle2) * dist

            arc_color = self.metrics[0]['color']  # Resonance color
            arc_color.setAlpha(int(self.resonance * 150))

            pen = QPen(arc_color, 2)
            painter.setPen(pen)

            # Jagged lightning effect
            steps = 5
            for i in range(steps):
                t1 = i / steps
                t2 = (i + 1) / steps

                offset = random.uniform(-10, 10)
                perp_x = -(y2 - y1) / steps * 0.3
                perp_y = (x2 - x1) / steps * 0.3

                x1_seg = x1 + (x2 - x1) * t1 + perp_x * offset
                y1_seg = y1 + (y2 - y1) * t1 + perp_y * offset
                x2_seg = x1 + (x2 - x1) * t2 + perp_x * offset
                y2_seg = y1 + (y2 - y1) * t2 + perp_y * offset

                painter.drawLine(int(x1_seg), int(y1_seg), int(x2_seg), int(y2_seg))

    def draw_celebration_burst(self, painter, cx, cy):
        """Celebration burst - rings of joy!"""
        num_rings = 3
        for i in range(num_rings):
            ring_phase = (self.celebration_phase + i * 0.5) % 3.0
            radius = 80 + (ring_phase * 60)
            alpha = int(255 * (1.0 - ring_phase / 3.0))

            hue = int((self.celebration_phase * 50 + i * 120) % 360)
            ring_color = QColor.fromHsv(hue, 220, 255, alpha)

            pen = QPen(ring_color, 3)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(int(cx - radius), int(cy - radius),
                              int(radius * 2), int(radius * 2))

        # Extra sparkles!
        for i in range(50):
            angle = (self.celebration_phase * 2 + i) % (math.pi * 2)
            distance = 50 + (i * 3)
            x = cx + math.cos(angle) * distance
            y = cy + math.sin(angle) * distance

            hue = int((self.celebration_phase * 100 + i * 7) % 360)
            sparkle_color = QColor.fromHsv(hue, 255, 255, random.randint(150, 255))
            painter.setPen(QPen(sparkle_color, 3))
            painter.drawPoint(int(x), int(y))

    def mousePressEvent(self, event):
        """Triple-click activates celebration mode!"""
        current_time = time.time()

        if current_time - self.last_click_time < 0.5:
            self.click_count += 1
        else:
            self.click_count = 1

        self.last_click_time = current_time

        if self.click_count >= 3:
            self.celebration_mode = True
            self.celebration_phase = 0.0
            self.celebration_timer = 0
            self.click_count = 0

            for particle in self.particles:
                particle['distance'] = random.uniform(80, 120)
