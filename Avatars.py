import tkinter as tk
import time
import math
import numpy as np
from PIL import Image, ImageDraw, ImageTk

# === BASE AVATAR CLASS ===
class BaseAvatarWindow(tk.Toplevel):
    """Base class for all avatars with common functionality"""

    # Common constants
    LEVELS = 32
    BG = "#000000"
    MASK_COLOR = "#00FF00"

    # Scale constants
    SCALE_MIN = 0.3
    SCALE_MAX = 2.0
    SCALE_STEP = 0.1

    # Window constants
    BASE_DIAMETER = 480

    def __init__(self, master, title="Avatar"):
        super().__init__(master)
        self.title(title)

        # === WINDOW SETUP ===
        self._scale_factor = 1.0
        self._base_diameter = self.BASE_DIAMETER

        # Make window circular/transparent
        try:
            self.overrideredirect(True)
            self.wm_attributes("-transparentcolor", self.MASK_COLOR)
            self.configure(bg=self.MASK_COLOR)
        except Exception:
            pass

        self._update_window_size()
        self.center_on_screen()

        # === DRAG & SCALE SETUP ===
        self._drag_data = {"x": 0, "y": 0}
        self.bind("<Button-1>", self._start_drag)
        self.bind("<B1-Motion>", self._do_drag)
        self.bind("<MouseWheel>", self._on_mouse_wheel)
        self.bind("<Button-4>", self._on_mouse_wheel)
        self.bind("<Button-5>", self._on_mouse_wheel)

        # Canvas setup
        self.canvas = tk.Canvas(self, bg=self.MASK_COLOR, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", self._on_configure)

        # Bind mouse events to canvas too
        self.canvas.bind("<MouseWheel>", self._on_mouse_wheel)
        self.canvas.bind("<Button-4>", self._on_mouse_wheel)
        self.canvas.bind("<Button-5>", self._on_mouse_wheel)

        # Common state
        self.level = 0
        self._running = True
        self.pad = 8

        # Animation timer
        self._animation_timer = None

    def _on_configure(self, e):
        """Handle canvas resize"""
        self.redraw()

    def _update_window_size(self):
        """Update window size based on current scale factor"""
        current_diameter = int(self._base_diameter * self._scale_factor)
        try:
            current_geometry = self.geometry()
            if '+' in current_geometry:
                parts = current_geometry.split('+')
                if len(parts) == 3:
                    x_pos, y_pos = int(parts[1]), int(parts[2])
                    self.geometry(f"{current_diameter}x{current_diameter}+{x_pos}+{y_pos}")
                else:
                    self.geometry(f"{current_diameter}x{current_diameter}")
            else:
                self.geometry(f"{current_diameter}x{current_diameter}")
        except Exception:
            self.geometry(f"{current_diameter}x{current_diameter}")

    def center_on_screen(self):
        """Center the window on screen"""
        self.update_idletasks()
        width = self.winfo_width()
        height = self.winfo_height()
        x = (self.winfo_screenwidth() // 2) - (width // 2)
        y = (self.winfo_screenheight() // 2) - (height // 2)
        self.geometry(f"{width}x{height}+{x}+{y}")

    def _start_drag(self, e):
        self._drag_data["x"] = e.x_root - self.winfo_x()
        self._drag_data["y"] = e.y_root - self.winfo_y()

    def _do_drag(self, e):
        self.geometry(f"+{e.x_root - self._drag_data['x']}+{e.y_root - self._drag_data['y']}")

    def _on_mouse_wheel(self, event):
        """Handle mouse wheel scaling"""
        if event.delta > 0 or event.num == 4:  # Scroll up
            new_scale = min(self.SCALE_MAX, self._scale_factor + self.SCALE_STEP)
        else:  # Scroll down
            new_scale = max(self.SCALE_MIN, self._scale_factor - self.SCALE_STEP)

        if new_scale != self._scale_factor:
            self._scale_factor = new_scale
            self._update_window_size()
            self.log_scale_change()

    def log_scale_change(self):
        """Log scale change - override if needed"""
        class_name = self.__class__.__name__
        try:
            if hasattr(self.master, 'logln'):
                self.master.logln(f"[avatar] {class_name} scale: {self._scale_factor:.1f}x")
            else:
                print(f"[avatar] {class_name} scale: {self._scale_factor:.1f}x")
        except:
            print(f"[avatar] {class_name} scale: {self._scale_factor:.1f}x")

    def _hsv_to_hex(self, h, s, v):
        """Convert HSV to hex color"""
        i = int(h * 6)
        f = h * 6 - i
        p = v * (1 - s)
        q = v * (1 - f * s)
        t = v * (1 - (1 - f) * s)
        i %= 6
        r, g, b = [(v, t, p), (q, v, p), (p, v, t),
                   (p, q, v), (t, p, v), (v, p, q)][i]
        return "#%02x%02x%02x" % (int(r * 255), int(g * 255), int(b * 255))

    def _circle_geom(self):
        """Get circle geometry"""
        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())
        cx, cy = cw // 2, ch // 2
        r = max(1, min(cw, ch) // 2 - self.pad)
        return cx, cy, r

    # === PUBLIC INTERFACE ===
    def show(self):
        self.deiconify()
        self.lift()

    def hide(self):
        self.withdraw()

    def destroy(self):
        self._running = False
        try:
            if self._animation_timer is not None:
                self.after_cancel(self._animation_timer)
        except Exception:
            pass
        super().destroy()

    def set_level(self, level: int):
        """Set the audio level (0 to LEVELS-1)"""
        self.level = max(0, min(self.LEVELS - 1, int(level)))

    def redraw(self):
        """Override this method in subclasses"""
        pass

    def set_scale(self, scale_factor: float):
        """Programmatically set scale factor"""
        self._scale_factor = max(self.SCALE_MIN, min(self.SCALE_MAX, scale_factor))
        self._update_window_size()
        self.log_scale_change()

    def get_scale(self) -> float:
        """Get current scale factor"""
        return self._scale_factor

    def reset_scale(self):
        """Reset to default scale"""
        self._scale_factor = 1.0
        self._update_window_size()
        self.log_scale_change()


# === CONCRETE AVATAR CLASSES ===

class CircleAvatarWindow(BaseAvatarWindow):
    """Circular rings avatar"""

    MAX_RINGS = 32

    def __init__(self, master):
        super().__init__(master, "Avatar - Rings")
        self._t0 = time.perf_counter()
        self.start_animation()

    def start_animation(self):
        """Start the animation timer"""
        if not self._running:
            return
        self.redraw()
        self._animation_timer = self.after(50, self.start_animation)

    def _ring_color(self, k, rings):
        t = (time.perf_counter() - self._t0) * 0.05
        x = ((k / max(1, rings - 1)) + t) % 1.0
        return self._hsv_to_hex(x, 0.9, 1.0)

    def redraw(self):
        cx, cy, r = self._circle_geom()

        # Draw circular background
        self.canvas.delete("all")
        self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r,
                                fill=self.BG, outline=self.BG)

        # Draw rings based on level
        rings = max(1, int((self.level / float(self.LEVELS - 1)) * self.MAX_RINGS + 0.5))
        r_min = max(1, int(r * 0.05))
        r_max = int(r * 0.95)
        r_outer = int(r_min + (r_max - r_min) * (self.level / float(self.LEVELS - 1)))
        stroke = max(2, int(r * 0.02))

        if rings == 1:
            col = self._ring_color(0, 1)
            self.canvas.create_oval(cx - r_min, cy - r_min, cx + r_min, cy + r_min,
                                    fill=col, outline=col)
            return

        for k in range(rings):
            rk = int(r_outer * (1.0 - k / float(rings)))
            if rk <= 1:
                continue
            col = self._ring_color(k, rings)
            self.canvas.create_oval(cx - rk, cy - rk, cx + rk, cy + rk,
                                    outline=col, width=stroke)


class RectAvatarWindow(BaseAvatarWindow):
    """Horizontal rectangles avatar"""

    # Visual parameters
    MAX_PARTICLES = 450
    SPAWN_AT_MAX_LVL = 60
    RECT_MIN_LEN_F = 0.03
    RECT_MAX_LEN_F = 0.22
    RECT_THICK_F = 0.012
    RECT_LIFETIME = 0.9
    DRIFT_PIX_F = 0.01
    LEVEL_DEADZONE = 2
    SPAWN_GAMMA = 1.8
    MIN_SPAWN = 0

    def __init__(self, master):
        super().__init__(master, "Avatar - Horizontal Rectangles")
        self._last_time = time.perf_counter()
        self._particles = []
        self.start_animation()

    def start_animation(self):
        """Start the animation loop"""
        if not self._running:
            return
        self.redraw()
        self._animation_timer = self.after(16, self.start_animation)

    def _spawn_count(self):
        if self.level <= self.LEVEL_DEADZONE:
            return 0
        usable = self.LEVELS - 1 - self.LEVEL_DEADZONE
        if usable <= 0:
            return 0
        x = (self.level - self.LEVEL_DEADZONE) / float(usable)
        x = max(0.0, min(1.0, x))
        return int(0.5 + self.MIN_SPAWN +
                   (self.SPAWN_AT_MAX_LVL - self.MIN_SPAWN) * (x ** self.SPAWN_GAMMA))

    def _uniform_point_in_disc(self, cx, cy, r):
        inner_r = max(2, r * 0.85)
        u = np.random.random()
        theta = 2 * np.pi * np.random.random()
        rho = inner_r * np.sqrt(u)
        return int(cx + rho * np.cos(theta)), int(cy + rho * np.sin(theta))

    def _spawn(self, n):
        """Spawn horizontal rectangles only"""
        cx, cy, r = self._circle_geom()
        if r <= 4:
            return

        d = 2 * r
        min_len = max(6, int(d * self.RECT_MIN_LEN_F))
        max_len = max(min_len + 2, int(d * self.RECT_MAX_LEN_F))
        thick = max(3, int(r * self.RECT_THICK_F))
        drift_p = max(1, int(r * self.DRIFT_PIX_F))
        now = time.perf_counter()

        for _ in range(n):
            x0, y0 = self._uniform_point_in_disc(cx, cy, r)

            # Horizontal rectangles only
            L = np.random.randint(min_len, max_len)
            dy = abs(y0 - cy)
            chord_half = int(math.sqrt(max(0, r * r - dy * dy)) * 0.95)
            halfL = min(L // 2, chord_half)
            x1, x2 = x0 - halfL, x0 + halfL
            y1, y2 = y0 - thick // 2, y0 + thick // 2

            vx = np.random.randint(-drift_p, drift_p)
            vy = np.random.randint(-drift_p, drift_p)
            col = self._hsv_to_hex(np.random.random(), 0.95, 1.0)

            self._particles.append({
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "vx": vx, "vy": vy, "birth": now, "life": self.RECT_LIFETIME,
                "color": col, "vertical": False
            })

        if len(self._particles) > self.MAX_PARTICLES:
            self._particles = self._particles[-self.MAX_PARTICLES:]

    def redraw(self):
        now = time.perf_counter()
        dt = max(0.0, now - self._last_time)
        self._last_time = now

        cx, cy, r = self._circle_geom()

        # Clear and draw circular background
        self.canvas.delete("all")
        self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r,
                                fill=self.BG, outline=self.BG)

        if self.level <= 0:
            self._particles.clear()
            return

        spawn = self._spawn_count()
        if spawn > 0:
            self._spawn(spawn)

        alive = []
        for p in self._particles:
            age = now - p["birth"]
            if age > p["life"]:
                continue

            # Update position
            p["x1"] += p["vx"] * dt
            p["x2"] += p["vx"] * dt
            p["y1"] += p["vy"] * dt
            p["y2"] += p["vy"] * dt

            # Keep within circle bounds
            mx = 0.5 * (p["x1"] + p["x2"])
            my = 0.5 * (p["y1"] + p["y2"])

            # Horizontal rectangle clipping
            dy = my - cy
            max_half_thick = max(0, int(math.sqrt(max(0, r * r - dy * dy))))
            half_thick = max(1, int((p["y2"] - p["y1"]) * 0.5))
            half_thick = min(half_thick, max_half_thick)
            p["y1"], p["y2"] = my - half_thick, my + half_thick

            chord_half = max(0, int(math.sqrt(max(0, r * r - dy * dy)) * 0.95))
            halfL = min(int((p["x2"] - p["x1"]) * 0.5), chord_half)
            p["x1"], p["x2"] = mx - halfL, mx + halfL

            # Fade effect
            t = age / p["life"]
            stipples = ("", "gray12", "gray25", "gray50", "gray75")
            idx = min(len(stipples) - 1, int(t * len(stipples)))
            stipple = stipples[idx]

            self.canvas.create_rectangle(
                int(p["x1"]), int(p["y1"]), int(p["x2"]), int(p["y2"]),
                fill=p["color"], outline=p["color"],
                stipple=stipple if stipple else None
            )
            alive.append(p)

        self._particles = alive


class RectAvatarWindow2(RectAvatarWindow):
    """Enhanced rectangles avatar with both horizontal and vertical rectangles"""

    # Enhanced parameters
    BASE_DIAMETER = 900
    VERTICAL_PROPORTION = 0.40
    CENTER_PULL = 0.07
    EDGE_INSET_F = 1.00
    SPAWN_RADIUS_F = 0.90

    def __init__(self, master):
        super().__init__(master)
        self.title("Avatar — Rectangles 2")

    def _spawn(self, n):
        """Spawn both horizontal and vertical rectangles"""
        cx, cy, r = self._circle_geom()
        if r <= 4:
            return

        d = 2 * r
        min_len_h = max(6, int(d * self.RECT_MIN_LEN_F))
        max_len_h = max(min_len_h + 2, int(d * self.RECT_MAX_LEN_F))
        min_len_v = min_len_h
        max_len_v = max_len_h

        thick = max(3, int(r * self.RECT_THICK_F))
        thick_h = thick
        thick_v = thick

        drift_p = max(1, int(r * self.DRIFT_PIX_F))
        now = time.perf_counter()

        for _ in range(n):
            x0, y0 = self._uniform_point_in_disc(cx, cy, r)
            vertical = (np.random.random() < self.VERTICAL_PROPORTION)

            if vertical:
                L = np.random.randint(min_len_v, max_len_v)
                dx = abs(x0 - cx)
                chord_half = int(math.sqrt(max(0, r * r - dx * dx)) * self.EDGE_INSET_F)
                halfL = min(L // 2, chord_half)
                x1, x2 = x0 - thick_v // 2, x0 + thick_v // 2
                y1, y2 = y0 - halfL, y0 + halfL
            else:
                L = np.random.randint(min_len_h, max_len_h)
                dy = abs(y0 - cy)
                chord_half = int(math.sqrt(max(0, r * r - dy * dy)) * self.EDGE_INSET_F)
                halfL = min(L // 2, chord_half)
                x1, x2 = x0 - halfL, x0 + halfL
                y1, y2 = y0 - thick_h // 2, y0 + thick_h // 2

            vx = np.random.randint(-drift_p, drift_p)
            vy = np.random.randint(-drift_p, drift_p)
            col = self._hsv_to_hex(np.random.random(), 0.95, 1.0)

            self._particles.append({
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "vx": vx, "vy": vy, "birth": now, "life": self.RECT_LIFETIME,
                "color": col, "vertical": bool(vertical)
            })

        if len(self._particles) > self.MAX_PARTICLES:
            self._particles = self._particles[-self.MAX_PARTICLES:]

    def redraw(self):
        now = time.perf_counter()
        dt = max(0.0, now - self._last_time)
        self._last_time = now

        cx, cy, r = self._circle_geom()

        # Clear and draw circular background
        self.canvas.delete("all")
        self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r,
                                fill=self.BG, outline=self.BG)

        if self.level <= 0:
            self._particles.clear()
            return

        spawn = self._spawn_count()
        if spawn > 0:
            self._spawn(spawn)

        alive = []
        for p in self._particles:
            age = now - p["birth"]
            if age > p["life"]:
                continue

            # Integrate motion
            p["x1"] += p["vx"] * dt
            p["x2"] += p["vx"] * dt
            p["y1"] += p["vy"] * dt
            p["y2"] += p["vy"] * dt

            # Gentle center pull
            if self.CENTER_PULL > 0.0:
                mx = 0.5 * (p["x1"] + p["x2"])
                my = 0.5 * (p["y1"] + p["y2"])
                pull = self.CENTER_PULL * dt
                dx = (cx - mx) * pull
                dy = (cy - my) * pull
                p["x1"] += dx
                p["x2"] += dx
                p["y1"] += dy
                p["y2"] += dy
                mx += dx
                my += dy
            else:
                mx = 0.5 * (p["x1"] + p["x2"])
                my = 0.5 * (p["y1"] + p["y2"])

            # Circle-aware clipping
            if p.get("vertical", False):
                dx = mx - cx
                max_half_thick = max(0, int(math.sqrt(max(0, r * r - dx * dx))))
                half_thick = max(1, int((p["x2"] - p["x1"]) * 0.5))
                half_thick = min(half_thick, max_half_thick)
                p["x1"], p["x2"] = mx - half_thick, mx + half_thick

                chord_half = max(0, int(math.sqrt(max(0, r * r - dx * dx)) * self.EDGE_INSET_F))
                halfL = min(int((p["y2"] - p["y1"]) * 0.5), chord_half)
                p["y1"], p["y2"] = my - halfL, my + halfL
            else:
                dy = my - cy
                max_half_thick = max(0, int(math.sqrt(max(0, r * r - dy * dy))))
                half_thick = max(1, int((p["y2"] - p["y1"]) * 0.5))
                half_thick = min(half_thick, max_half_thick)
                p["y1"], p["y2"] = my - half_thick, my + half_thick

                chord_half = max(0, int(math.sqrt(max(0, r * r - dy * dy)) * self.EDGE_INSET_F))
                halfL = min(int((p["x2"] - p["x1"]) * 0.5), chord_half)
                p["x1"], p["x2"] = mx - halfL, mx + halfL

            # Fade via stipple
            t = age / p["life"]
            stipples = ("", "gray12", "gray25", "gray50", "gray75")
            idx = min(len(stipples) - 1, int(t * len(stipples)))
            stipple = stipples[idx]

            self.canvas.create_rectangle(
                int(p["x1"]), int(p["y1"]), int(p["x2"]), int(p["y2"]),
                fill=p["color"], outline=p["color"],
                stipple=stipple if stipple else None
            )
            alive.append(p)

        self._particles = alive


class RadialPulseAvatar(BaseAvatarWindow):
    """Minimalist radial pulse avatar"""

    # Visual parameters
    DOT_COLOR = "#FF0000"
    LINE_COLOR = "#FF3333"
    MAX_LINES = 24
    MAX_LINE_LENGTH = 0.8
    PULSE_SMOOTHING = 0.3

    def __init__(self, master):
        super().__init__(master, "Avatar - Radial Pulse")
        self._smoothed_level = 0.0
        self._last_time = time.perf_counter()
        self.start_animation()

    def start_animation(self):
        """Start the animation loop"""
        if not self._running:
            return
        self.redraw()
        self._animation_timer = self.after(16, self.start_animation)

    def redraw(self):
        """Draw the radial pulse avatar"""
        now = time.perf_counter()
        dt = max(0.001, now - self._last_time)
        self._last_time = now

        # Smooth level changes
        target_level = self.level / float(self.LEVELS - 1)
        self._smoothed_level += (target_level - self._smoothed_level) * self.PULSE_SMOOTHING

        self.canvas.delete("all")
        cx, cy, r = self._circle_geom()

        # Draw circular background
        self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r,
                                fill=self.BG, outline=self.BG)

        # Calculate pulse intensity
        base_intensity = self._smoothed_level
        pulse_intensity = base_intensity * (0.8 + 0.4 * math.sin(time.perf_counter() * 8))

        # Draw central red dot
        dot_radius = max(2, int(4 + pulse_intensity * 6))
        self.canvas.create_oval(
            cx - dot_radius, cy - dot_radius,
            cx + dot_radius, cy + dot_radius,
            fill=self.DOT_COLOR, outline=self.DOT_COLOR
        )

        # Draw radial lines
        if pulse_intensity > 0.05:
            num_lines = self.MAX_LINES
            max_line_length = min(self.canvas.winfo_width(),
                                  self.canvas.winfo_height()) * self.MAX_LINE_LENGTH * pulse_intensity

            for i in range(num_lines):
                angle = (2 * math.pi * i) / num_lines

                # Add slight randomness to line lengths
                line_variation = 0.7 + 0.6 * math.sin(angle * 3 + time.perf_counter() * 6)
                line_length = max_line_length * line_variation

                # Calculate line end point
                end_x = cx + line_length * math.cos(angle)
                end_y = cy + line_length * math.sin(angle)

                # Line width varies with intensity
                line_width = max(1, int(1 + pulse_intensity * 3))

                # Draw the radial line
                self.canvas.create_line(
                    cx, cy, end_x, end_y,
                    fill=self.LINE_COLOR,
                    width=line_width,
                    capstyle=tk.ROUND
                )

class FaceRadialAvatar(tk.Toplevel):
    """Face avatar with radial lines that respond to audio levels - Egg shaped"""

    LEVELS = 32
    BG = "#000000"
    MASK_COLOR = "#00FF00"

    # Scale constants
    SCALE_MIN = 0.3
    SCALE_MAX = 2.0
    SCALE_STEP = 0.1

    # Window constants - ELLIPTICAL
    BASE_HEIGHT = 600  # Vertical is longer for egg shape
    BASE_WIDTH = 480  # Horizontal is shorter

    # Radial line parameters (from RadialPulseAvatar template)
    DOT_COLOR = "#FF0000"
    LINE_COLOR = "#FF3333"
    MAX_LINES = 24
    MAX_LINE_LENGTH = 0.95
    PULSE_SMOOTHING = 0.3

    def __init__(self, master):
        super().__init__(master)
        self.title("Avatar - Face (Egg)")

        # === WINDOW SETUP ===
        self._scale_factor = 1.0
        self._base_height = self.BASE_HEIGHT
        self._base_width = self.BASE_WIDTH

        # Make window elliptical/transparent
        try:
            self.overrideredirect(True)
            self.wm_attributes("-transparentcolor", self.MASK_COLOR)
            self.configure(bg=self.MASK_COLOR)
        except Exception:
            pass

        self._update_window_size()
        self.center_on_screen()

        # === DRAG & SCALE SETUP ===
        self._drag_data = {"x": 0, "y": 0}
        self.bind("<Button-1>", self._start_drag)
        self.bind("<B1-Motion>", self._do_drag)
        self.bind("<MouseWheel>", self._on_mouse_wheel)
        self.bind("<Button-4>", self._on_mouse_wheel)
        self.bind("<Button-5>", self._on_mouse_wheel)

        # Canvas setup
        self.canvas = tk.Canvas(self, bg=self.MASK_COLOR, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", self._on_configure)

        # Bind mouse events to canvas too
        self.canvas.bind("<MouseWheel>", self._on_mouse_wheel)
        self.canvas.bind("<Button-4>", self._on_mouse_wheel)
        self.canvas.bind("<Button-5>", self._on_mouse_wheel)

        # Face image setup
        self.level = 0
        self._running = True
        self.pad = 8

        # Load face image
        self.face_image = None
        self.face_photo = None
        self._load_face_image()

        # Animation state (from RadialPulseAvatar)
        self._smoothed_level = 0.0
        self._last_time = time.perf_counter()
        self._animation_timer = None

        self.start_animation()

    def _load_face_image(self):
        """Load the face image from file"""
        try:
            self.face_image = Image.open("face.png")
            self._update_face_display()
        except Exception as e:
            print(f"Error loading face.png: {e}")
            # Create a placeholder if image fails to load
            self.face_image = Image.new('RGB', (100, 150), color='red')  # Taller placeholder

    def _on_configure(self, e):
        """Handle canvas resize"""
        self._update_face_display()
        self.redraw()

    def _update_window_size(self):
        """Update window size based on current scale factor - ELLIPTICAL"""
        current_height = int(self._base_height * self._scale_factor)
        current_width = int(self._base_width * self._scale_factor)

        try:
            current_geometry = self.geometry()
            if '+' in current_geometry:
                parts = current_geometry.split('+')
                if len(parts) == 3:
                    x_pos, y_pos = int(parts[1]), int(parts[2])
                    self.geometry(f"{current_width}x{current_height}+{x_pos}+{y_pos}")
                else:
                    self.geometry(f"{current_width}x{current_height}")
            else:
                self.geometry(f"{current_width}x{current_height}")
        except Exception:
            self.geometry(f"{current_width}x{current_height}")

    def center_on_screen(self):
        """Center the elliptical window on screen"""
        self.update_idletasks()
        width = self.winfo_width()
        height = self.winfo_height()
        x = (self.winfo_screenwidth() // 2) - (width // 2)
        y = (self.winfo_screenheight() // 2) - (height // 2)
        self.geometry(f"{width}x{height}+{x}+{y}")

    def _start_drag(self, e):
        self._drag_data["x"] = e.x_root - self.winfo_x()
        self._drag_data["y"] = e.y_root - self.winfo_y()

    def _do_drag(self, e):
        self.geometry(f"+{e.x_root - self._drag_data['x']}+{e.y_root - self._drag_data['y']}")

    def _on_mouse_wheel(self, event):
        """Handle mouse wheel scaling"""
        if event.delta > 0 or event.num == 4:  # Scroll up
            new_scale = min(self.SCALE_MAX, self._scale_factor + self.SCALE_STEP)
        else:  # Scroll down
            new_scale = max(self.SCALE_MIN, self._scale_factor - self.SCALE_STEP)

        if new_scale != self._scale_factor:
            self._scale_factor = new_scale
            self._update_window_size()
            self.log_scale_change()
            self._update_face_display()

    def log_scale_change(self):
        """Log scale change"""
        class_name = self.__class__.__name__
        try:
            if hasattr(self.master, 'logln'):
                self.master.logln(f"[avatar] {class_name} scale: {self._scale_factor:.1f}x")
            else:
                print(f"[avatar] {class_name} scale: {self._scale_factor:.1f}x")
        except:
            print(f"[avatar] {class_name} scale: {self._scale_factor:.1f}x")

    def _update_face_display(self):
        """Update the face image display based on current window size - FIT TO ELLIPSE"""
        if self.face_image is None:
            return

        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())

        # Calculate size for elliptical display - fit within the egg shape
        display_width = cw - (self.pad * 2)
        display_height = ch - (self.pad * 2)

        if display_width > 0 and display_height > 0:
            # Resize image to fit the elliptical bounds (taller than wide)
            resized_image = self.face_image.resize(
                (display_width, display_height),
                Image.Resampling.LANCZOS
            )
            self.face_photo = ImageTk.PhotoImage(resized_image)

    def start_animation(self):
        """Start the animation loop"""
        if not self._running:
            return
        self.redraw()
        self._animation_timer = self.after(16, self.start_animation)

    def redraw(self):
        """Draw the face avatar with radial lines - EGG SHAPED"""
        now = time.perf_counter()
        dt = max(0.001, now - self._last_time)
        self._last_time = now

        # Smooth level changes (from RadialPulseAvatar)
        target_level = self.level / float(self.LEVELS - 1)
        self._smoothed_level += (target_level - self._smoothed_level) * self.PULSE_SMOOTHING

        self.canvas.delete("all")

        # Get elliptical geometry instead of circular
        cx, cy, rx, ry = self._ellipse_geom()

        # Draw elliptical background (egg shape)
        self.canvas.create_oval(
            cx - rx, cy - ry,
            cx + rx, cy + ry,
            fill=self.BG, outline=self.BG
        )

        # Draw face image if loaded (fits the elliptical bounds)
        if self.face_photo:
            img_x = cx - self.face_photo.width() // 2
            img_y = cy - self.face_photo.height() // 2
            self.canvas.create_image(img_x, img_y, anchor="nw", image=self.face_photo)

        # Calculate pulse intensity (from RadialPulseAvatar)
        base_intensity = self._smoothed_level
        pulse_intensity = base_intensity * (0.8 + 0.4 * math.sin(time.perf_counter() * 8))

        # Draw central dot (optional - can remove if you don't want it)
        dot_radius = max(2, int(4 + pulse_intensity * 6))
        self.canvas.create_oval(
            cx - dot_radius, cy - dot_radius,
            cx + dot_radius, cy + dot_radius,
            fill=self.DOT_COLOR, outline=self.DOT_COLOR
        )

        # Draw radial lines (from RadialPulseAvatar template) - ADAPTED FOR ELLIPSE
        if pulse_intensity > 0.05:  # Only draw lines when there's significant audio
            num_lines = self.MAX_LINES

            # Calculate max line length based on elliptical bounds
            # Use the average of rx and ry for balanced line lengths, or adjust as needed
            avg_radius = (rx + ry) / 2
            max_line_length = avg_radius * self.MAX_LINE_LENGTH * pulse_intensity

            for i in range(num_lines):
                angle = (2 * math.pi * i) / num_lines

                # Add slight randomness to line lengths for organic feel
                line_variation = 0.7 + 0.6 * math.sin(angle * 3 + time.perf_counter() * 6)
                line_length = max_line_length * line_variation

                # Calculate line end point
                end_x = cx + line_length * math.cos(angle)
                end_y = cy + line_length * math.sin(angle)

                # Line width varies with intensity
                line_width = max(1, int(1 + pulse_intensity * 3))

                # Draw the radial line
                self.canvas.create_line(
                    cx, cy, end_x, end_y,
                    fill=self.LINE_COLOR,
                    width=line_width,
                    capstyle='round'
                )

    def _ellipse_geom(self):
        """Get ellipse geometry (egg shape - taller than wide)"""
        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())
        cx, cy = cw // 2, ch // 2

        # Elliptical radii - vertical is longer (egg shape)
        rx = max(1, (cw // 2) - self.pad)  # Horizontal radius
        ry = max(1, (ch // 2) - self.pad)  # Vertical radius (longer)

        return cx, cy, rx, ry

    # === PUBLIC INTERFACE ===
    def show(self):
        self.deiconify()
        self.lift()

    def hide(self):
        self.withdraw()

    def destroy(self):
        self._running = False
        try:
            if self._animation_timer is not None:
                self.after_cancel(self._animation_timer)
        except Exception:
            pass
        super().destroy()

    def set_level(self, level: int):
        """Set the audio level (0 to LEVELS-1)"""
        self.level = max(0, min(self.LEVELS - 1, int(level)))

    def set_scale(self, scale_factor: float):
        """Programmatically set scale factor"""
        self._scale_factor = max(self.SCALE_MIN, min(self.SCALE_MAX, scale_factor))
        self._update_window_size()
        self.log_scale_change()
        self._update_face_display()

    def get_scale(self) -> float:
        """Get current scale factor"""
        return self._scale_factor

    def reset_scale(self):
        """Reset to default scale"""
        self._scale_factor = 1.0
        self._update_window_size()
        self.log_scale_change()
        self._update_face_display()
