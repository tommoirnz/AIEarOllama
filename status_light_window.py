import tkinter as tk
from tkinter import ttk


class StatusLightWindow(tk.Toplevel):
    """Circular status light window with 3D effect"""

    def __init__(self, master):
        super().__init__(master)
        self.title("Status Light")

        # Initial size - your preferred smaller size
        self._size = 15  # Much smaller as requested
        self.geometry(f"{self._size}x{self._size}")

        # === TRANSPARENT BACKGROUND SETUP ===
        self.BG = "#000000"
        self.MASK_COLOR = "#00FF00"  # Same as avatars

        try:
            self.overrideredirect(True)
            self.wm_attributes("-transparentcolor", self.MASK_COLOR)
            self.configure(bg=self.MASK_COLOR)
        except Exception:
            pass

        self.center_on_screen()

        # === DRAG SETUP ===
        self._drag_data = {"x": 0, "y": 0}
        self.bind("<Button-1>", self._start_drag)
        self.bind("<B1-Motion>", self._do_drag)

        # === MOUSE WHEEL SCALING ===
        self.bind("<MouseWheel>", self._on_mouse_wheel)
        self.bind("<Button-4>", self._on_mouse_wheel)  # Linux scroll up
        self.bind("<Button-5>", self._on_mouse_wheel)  # Linux scroll down

        # Canvas setup with transparent background
        self.canvas = tk.Canvas(self, bg=self.MASK_COLOR, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=0, pady=0)

        # Bind mouse events to canvas too
        self.canvas.bind("<MouseWheel>", self._on_mouse_wheel)
        self.canvas.bind("<Button-4>", self._on_mouse_wheel)
        self.canvas.bind("<Button-5>", self._on_mouse_wheel)

        # Draw initial light
        self._current_color = "#f1c40f"  # Default idle color
        self._draw_light()

        # === RIGHT-CLICK TO CLOSE ===
        self.bind("<Button-3>", self._on_right_click)  # Right-click to close
        self.canvas.bind("<Button-3>", self._on_right_click)

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
        MIN_SIZE = 10  # Even smaller minimum
        MAX_SIZE = 200
        SIZE_STEP = 2  # Smaller steps for fine control at small sizes

        if event.delta > 0 or event.num == 4:  # Scroll up = larger
            new_size = min(MAX_SIZE, self._size + SIZE_STEP)
        else:  # Scroll down = smaller
            new_size = max(MIN_SIZE, self._size - SIZE_STEP)

        if new_size != self._size:
            # Get current position
            x, y = self.winfo_x(), self.winfo_y()

            # Update size
            self._size = new_size
            self.geometry(f"{new_size}x{new_size}+{x}+{y}")

            # Redraw light
            self._draw_light()

            # Log the change
            self._log_size_change()

    def _on_right_click(self, event):
        """Right-click to close the window"""
        self.hide()

    def _log_size_change(self):
        """Log size change"""
        try:
            if hasattr(self.master, 'logln'):
                self.master.logln(f"[status-light] Size: {self._size}x{self._size}")
        except:
            print(f"[status-light] Size: {self._size}x{self._size}")

    def _get_3d_colors(self, base_color):
        """Generate 3D effect colors from base color"""
        import colorsys

        # Convert hex to RGB
        hex_color = base_color.lstrip('#')
        rgb = tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))

        # Convert RGB to HSL
        h, l, s = colorsys.rgb_to_hls(rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0)

        # Create lighter version for highlight
        light_h, light_l, light_s = h, min(1.0, l + 0.3), s
        light_rgb = colorsys.hls_to_rgb(light_h, light_l, light_s)
        light_color = '#%02x%02x%02x' % (
            int(light_rgb[0] * 255),
            int(light_rgb[1] * 255),
            int(light_rgb[2] * 255)
        )

        # Create darker version for shadow
        dark_h, dark_l, dark_s = h, max(0.0, l - 0.3), s
        dark_rgb = colorsys.hls_to_rgb(dark_h, dark_l, dark_s)
        dark_color = '#%02x%02x%02x' % (
            int(dark_rgb[0] * 255),
            int(dark_rgb[1] * 255),
            int(dark_rgb[2] * 255)
        )

        return base_color, light_color, dark_color

    def _draw_light(self):
        """Draw the status light circle with 3D effect"""
        self.canvas.delete("all")

        # Get canvas dimensions
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()

        # Use current size if canvas not yet drawn
        if cw <= 1:
            cw, ch = self._size, self._size

        cx, cy = cw // 2, ch // 2

        # Dynamic sizing based on window size
        if self._size <= 20:
            radius = self._size // 2 - 1
            bevel_size = 1
        elif self._size <= 40:
            radius = self._size // 2 - 2
            bevel_size = 1
        else:
            radius = self._size // 2 - 3
            bevel_size = 2

        # Ensure minimum radius
        radius = max(2, radius)

        # Get 3D colors
        base_color, light_color, dark_color = self._get_3d_colors(self._current_color)

        # Draw circular background (transparent area)
        self.canvas.create_oval(cx - radius, cy - radius, cx + radius, cy + radius,
                                fill=self.MASK_COLOR, outline=self.MASK_COLOR)

        # For very small sizes, use simple 3D effect
        if self._size <= 20:
            # Simple inner glow effect for tiny lights
            inner_radius = radius - 1
            self.canvas.create_oval(cx - radius, cy - radius, cx + radius, cy + radius,
                                    fill=base_color, outline=base_color)
            if inner_radius > 0:
                self.canvas.create_oval(cx - inner_radius, cy - inner_radius,
                                        cx + inner_radius, cy + inner_radius,
                                        fill=light_color, outline=light_color)
        else:
            # Full 3D effect for larger lights
            # Main light body
            self.canvas.create_oval(cx - radius, cy - radius, cx + radius, cy + radius,
                                    fill=base_color, outline=base_color)

            # Top-left highlight (lighter color)
            highlight_radius = radius - bevel_size
            if highlight_radius > 0:
                self.canvas.create_oval(cx - highlight_radius, cy - highlight_radius,
                                        cx + highlight_radius, cy + highlight_radius,
                                        fill=light_color, outline=light_color)

            # Bottom-right shadow (darker color) - offset slightly
            shadow_offset = max(1, bevel_size // 2)
            shadow_radius = radius - bevel_size
            if shadow_radius > 0:
                self.canvas.create_oval(cx - shadow_radius + shadow_offset,
                                        cy - shadow_radius + shadow_offset,
                                        cx + shadow_radius + shadow_offset,
                                        cy + shadow_radius + shadow_offset,
                                        fill=dark_color, outline=dark_color)

    def set_light(self, color):
        """Set the light color and redraw with 3D effect"""
        self._current_color = color
        self._draw_light()

    def show(self):
        self.deiconify()
        self.lift()
        self.focus_force()

    def hide(self):
        self.withdraw()
