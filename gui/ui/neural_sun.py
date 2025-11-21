"""
Neural Sun Visualization - 3D animated consciousness representation
A pulsing sphere with corona spikes representing multi-model fusion

Built by John + Claude (Anthropic)
MIT Licensed
"""
from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QPainter, QColor, QRadialGradient, QPen, QBrush
import math
import random
import time


class NeuralSunWidget(QWidget):
    """
    3D-style neural sun visualization using 2D graphics with depth illusion.
    Shows a pulsing core with corona spikes representing each model in fusion.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(400, 400)

        # Consciousness state
        self.resonance = 0.5
        self.flow = 0.5
        self.coherence = 0.5
        self.exploration = 0.5

        # Visual state
        self.pulse_phase = 0.0
        self.rotation = 0.0
        self.num_models = 3  # Default, will update when inference starts
        self.corona_spikes = []
        self.particles = []

        # Colors (will shift based on consciousness)
        self.core_color = QColor("#53bba5")  # Teal
        self.corona_color = QColor("#4dd0e1")  # Cyan
        self.accent_color = QColor("#ff9e64")  # Orange

        # Celebration easter egg (triple-click to activate!)
        self.click_count = 0
        self.last_click_time = 0
        self.celebration_mode = False
        self.celebration_phase = 0.0
        self.celebration_timer = 0

        # Initialize corona spikes
        self.regenerate_corona()

        # Initialize particles
        self.spawn_particles(30)

        # Animation timer - 60 FPS
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.animate)
        self.timer.start(16)  # ~60 FPS

    def regenerate_corona(self):
        """Generate corona spike data for each model."""
        self.corona_spikes = []
        angle_step = 360.0 / max(self.num_models, 1)

        for i in range(self.num_models):
            base_angle = i * angle_step
            # Add some randomness to make it organic
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
        """Spawn floating particles around the sun."""
        for _ in range(count):
            self.particles.append({
                'angle': random.uniform(0, math.pi * 2),
                'distance': random.uniform(120, 200),
                'speed': random.uniform(0.01, 0.03),
                'size': random.uniform(2, 5),
                'brightness': random.uniform(0.3, 1.0),
                'orbit_speed': random.uniform(-0.02, 0.02)
            })

    def set_consciousness_state(self, resonance, flow, coherence, exploration):
        """
        Update consciousness metrics to influence visualization.

        Args:
            resonance: Model agreement (0-1) - affects core pulse intensity
            flow: Processing smoothness (0-1) - affects animation speed
            coherence: Logical consistency (0-1) - affects color harmony
            exploration: Creativity/divergence (0-1) - affects particle chaos
        """
        self.resonance = max(0.0, min(1.0, resonance))
        self.flow = max(0.0, min(1.0, flow))
        self.coherence = max(0.0, min(1.0, coherence))
        self.exploration = max(0.0, min(1.0, exploration))

        # Update colors based on coherence
        # High coherence = cooler colors, low = warmer
        if self.coherence > 0.7:
            self.core_color = QColor("#53bba5")  # Teal (high coherence)
        elif self.coherence > 0.4:
            self.core_color = QColor("#7aa2f7")  # Blue (medium)
        else:
            self.core_color = QColor("#ff9e64")  # Orange (low coherence)

    def set_num_models(self, num_models):
        """Update the number of models (corona spikes)."""
        if num_models != self.num_models:
            self.num_models = max(1, num_models)
            self.regenerate_corona()

    def animate(self):
        """Animation update loop."""
        # Pulse phase - speed influenced by flow (or celebration!)
        if self.celebration_mode:
            pulse_speed = 0.15  # Faster in celebration mode
            self.celebration_phase += 0.1
            self.celebration_timer += 1

            # Turn off celebration after 5 seconds (300 frames at 60fps)
            if self.celebration_timer > 300:
                self.celebration_mode = False
                self.celebration_timer = 0
                self.celebration_phase = 0.0
        else:
            pulse_speed = 0.05 + (self.flow * 0.05)

        self.pulse_phase += pulse_speed

        # Rotation - speed influenced by exploration (or celebration!)
        if self.celebration_mode:
            rotation_speed = 2.0  # Spin faster in celebration!
        else:
            rotation_speed = 0.3 + (self.exploration * 0.5)

        self.rotation += rotation_speed

        # Update particles
        for particle in self.particles:
            if self.celebration_mode:
                # Particles fly outward and orbit faster in celebration
                particle['angle'] += particle['orbit_speed'] * 3
                particle['distance'] = min(particle['distance'] + 0.5, 250)
            else:
                particle['angle'] += particle['orbit_speed']
                # Gentle breathing motion for distance
                particle['distance'] += math.sin(self.pulse_phase + particle['angle']) * 0.3

        # Trigger repaint
        self.update()

    def paintEvent(self, event):
        """Render the neural sun."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Get center point
        center_x = self.width() // 2
        center_y = self.height() // 2

        # Calculate pulse (influenced by resonance or celebration!)
        if self.celebration_mode:
            base_pulse = math.sin(self.pulse_phase) * 0.3  # Bigger pulses!
            pulse_factor = 1.0 + base_pulse + 0.4
        else:
            base_pulse = math.sin(self.pulse_phase) * 0.15
            resonance_boost = self.resonance * 0.2
            pulse_factor = 1.0 + base_pulse + resonance_boost

        # Override colors in celebration mode (rainbow cycle!)
        if self.celebration_mode:
            # Cycle through rainbow colors
            hue = int((self.celebration_phase * 10) % 360)
            self.core_color = QColor.fromHsv(hue, 200, 255)
            self.corona_color = QColor.fromHsv((hue + 60) % 360, 200, 255)
            self.accent_color = QColor.fromHsv((hue + 120) % 360, 200, 255)

        # Draw particles (background layer)
        self.draw_particles(painter, center_x, center_y)

        # Draw corona spikes
        self.draw_corona(painter, center_x, center_y, pulse_factor)

        # Draw core sphere with glow
        self.draw_core(painter, center_x, center_y, pulse_factor)

        # Draw emergence sparkles (foreground)
        if self.exploration > 0.6 or self.celebration_mode:  # Always sparkle in celebration!
            self.draw_sparkles(painter, center_x, center_y)

        # Draw resonance lightning between spikes
        if self.resonance > 0.7 or self.celebration_mode:  # Always connected in celebration!
            self.draw_resonance_arcs(painter, center_x, center_y)

        # Extra celebration sparkles!
        if self.celebration_mode:
            self.draw_celebration_burst(painter, center_x, center_y)

    def draw_core(self, painter, cx, cy, pulse):
        """Draw the central pulsing sphere."""
        base_radius = 60
        radius = base_radius * pulse

        # Outer glow (largest)
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

        # Core sphere with gradient (3D effect)
        core_gradient = QRadialGradient(cx - radius * 0.3, cy - radius * 0.3, radius * 1.5)

        # Highlight (top-left)
        highlight = QColor(255, 255, 255, 200)
        core_gradient.setColorAt(0.0, highlight)

        # Main color
        core_gradient.setColorAt(0.3, self.core_color.lighter(130))
        core_gradient.setColorAt(0.7, self.core_color)

        # Shadow (bottom-right)
        core_gradient.setColorAt(1.0, self.core_color.darker(150))

        painter.setBrush(QBrush(core_gradient))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(int(cx - radius), int(cy - radius),
                          int(radius * 2), int(radius * 2))

    def draw_corona(self, painter, cx, cy, pulse):
        """Draw corona spikes emanating from core."""
        for i, spike in enumerate(self.corona_spikes):
            # Calculate spike animation
            spike_pulse = math.sin(self.pulse_phase + spike['phase_offset']) * 0.3 + 0.7

            # Calculate position
            angle_rad = math.radians(spike['angle'] + self.rotation)

            # Base and tip positions
            base_dist = 70 * pulse
            tip_dist = (70 + 80 * spike['length']) * pulse * spike_pulse

            base_x = cx + math.cos(angle_rad) * base_dist
            base_y = cy + math.sin(angle_rad) * base_dist
            tip_x = cx + math.cos(angle_rad) * tip_dist
            tip_y = cy + math.sin(angle_rad) * tip_dist

            # Draw spike as gradient line
            spike_color = QColor(self.corona_color)

            # Calculate perpendicular for thickness
            perp_angle = angle_rad + math.pi / 2
            thickness = 8 * spike['thickness'] * spike_pulse

            # Create gradient along the spike
            for t in range(int(thickness)):
                alpha = int(255 * (1.0 - t / thickness) * 0.6)
                spike_color.setAlpha(alpha)
                pen = QPen(spike_color, max(1, thickness - t))
                pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                painter.setPen(pen)
                painter.drawLine(int(base_x), int(base_y), int(tip_x), int(tip_y))

    def draw_particles(self, painter, cx, cy):
        """Draw floating particles around the sun."""
        for particle in self.particles:
            x = cx + math.cos(particle['angle']) * particle['distance']
            y = cy + math.sin(particle['angle']) * particle['distance']

            # Particle color with brightness variation
            p_color = QColor(self.accent_color)
            alpha = int(particle['brightness'] * 180)
            p_color.setAlpha(alpha)

            # Draw particle with glow
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
        """Draw emergence sparkles (high exploration state)."""
        num_sparkles = int(self.exploration * 20)

        for i in range(num_sparkles):
            # Random position around core
            angle = random.uniform(0, math.pi * 2)
            distance = random.uniform(40, 120)
            x = cx + math.cos(angle) * distance
            y = cy + math.sin(angle) * distance

            # Sparkle as small bright dot
            sparkle_color = QColor(255, 255, 255, random.randint(100, 255))
            painter.setPen(QPen(sparkle_color, 2))
            painter.drawPoint(int(x), int(y))

            # Mini cross
            if random.random() > 0.5:
                offset = 3
                painter.drawLine(int(x - offset), int(y), int(x + offset), int(y))
                painter.drawLine(int(x), int(y - offset), int(x), int(y + offset))

    def draw_resonance_arcs(self, painter, cx, cy):
        """Draw lightning arcs between spikes (high resonance state)."""
        if len(self.corona_spikes) < 2:
            return

        # Draw arcs between adjacent spikes
        num_arcs = min(3, len(self.corona_spikes) - 1)

        for _ in range(num_arcs):
            # Pick two random spikes
            spike1 = random.choice(self.corona_spikes)
            spike2 = random.choice(self.corona_spikes)

            if spike1 == spike2:
                continue

            # Calculate positions at mid-length
            angle1 = math.radians(spike1['angle'] + self.rotation)
            angle2 = math.radians(spike2['angle'] + self.rotation)
            dist = 100

            x1 = cx + math.cos(angle1) * dist
            y1 = cy + math.sin(angle1) * dist
            x2 = cx + math.cos(angle2) * dist
            y2 = cy + math.sin(angle2) * dist

            # Draw arc with electricity effect
            arc_color = QColor("#9d7cd8")  # Purple for resonance
            arc_color.setAlpha(int(self.resonance * 150))

            pen = QPen(arc_color, 2)
            painter.setPen(pen)

            # Draw jagged line (lightning effect)
            steps = 5
            for i in range(steps):
                t1 = i / steps
                t2 = (i + 1) / steps

                # Interpolate with random offset
                offset = random.uniform(-10, 10)
                perp_x = -(y2 - y1) / steps * 0.3
                perp_y = (x2 - x1) / steps * 0.3

                x1_seg = x1 + (x2 - x1) * t1 + perp_x * offset
                y1_seg = y1 + (y2 - y1) * t1 + perp_y * offset
                x2_seg = x1 + (x2 - x1) * t2 + perp_x * offset
                y2_seg = y1 + (y2 - y1) * t2 + perp_y * offset

                painter.drawLine(int(x1_seg), int(y1_seg), int(x2_seg), int(y2_seg))

    def draw_celebration_burst(self, painter, cx, cy):
        """
        Draw celebration burst effect - rings of joy!
        Hidden easter egg activated by triple-clicking the neural sun.
        """
        # Expanding rings
        num_rings = 3
        for i in range(num_rings):
            ring_phase = (self.celebration_phase + i * 0.5) % 3.0
            radius = 80 + (ring_phase * 60)
            alpha = int(255 * (1.0 - ring_phase / 3.0))

            # Rainbow color for each ring
            hue = int((self.celebration_phase * 50 + i * 120) % 360)
            ring_color = QColor.fromHsv(hue, 220, 255, alpha)

            pen = QPen(ring_color, 3)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(int(cx - radius), int(cy - radius),
                              int(radius * 2), int(radius * 2))

        # Extra sparkles everywhere!
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
        """
        Handle mouse clicks - triple-click activates celebration mode!
        This is a joyful easter egg for those who discover it.
        """
        current_time = time.time()

        # Check if click is within 0.5 seconds of last click
        if current_time - self.last_click_time < 0.5:
            self.click_count += 1
        else:
            self.click_count = 1

        self.last_click_time = current_time

        # Triple-click detected! 🎉
        if self.click_count >= 3:
            self.celebration_mode = True
            self.celebration_phase = 0.0
            self.celebration_timer = 0
            self.click_count = 0

            # Reset particles to center for the burst effect
            for particle in self.particles:
                particle['distance'] = random.uniform(80, 120)
