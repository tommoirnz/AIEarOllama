# avatar_wav_rects_modulated_blackbg.py
import os, time, random
import tkinter as tk
import numpy as np
import sounddevice as sd
import soundfile as sf

# === CONFIG ===
WAV_PATH    = r"out\last_reply.wav"
LEVELS      = 32
FPS_MS      = 16
BG_ACTIVE   = "#000000"   # black during speech
BG_SILENT   = "#000000"   # also black when silent

# Rectangle visual tuning
MAX_PARTICLES     = 450
SPAWN_AT_MAX_LVL  = 60
RECT_MIN_LEN_F    = 0.03
RECT_MAX_LEN_F    = 0.22
RECT_THICK_F      = 0.012
RECT_LIFETIME     = 0.9
DRIFT_PIX_F       = 0.01

# Spawn shaping
LEVEL_DEADZONE   = 2     # ignore very quiet input
SPAWN_GAMMA      = 1.8   # controls growth curve
MIN_SPAWN        = 0     # min number of rectangles at low volume

# NEW: proportion of rectangles that will be vertical (0..1)
VERTICAL_PROPORTION = 0.35

# ------------------------------------------------------------
def list_output_devices():
    devs = sd.query_devices()
    return [(i, d["name"], d.get("default_samplerate")) for i, d in enumerate(devs)
            if d.get("max_output_channels", 0) > 0]

def load_wav_and_envelope(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    stereo = data[:, :2] if data.shape[1] > 1 else np.repeat(data, 2, 1)
    mono = stereo.mean(1)

    # Normalize + attack/release smooth + compand
    pk = np.max(np.abs(mono)) if mono.size else 1.0
    if pk > 1e-9:
        mono /= pk
    e = 0.0
    sm = np.empty_like(mono)
    for n, v in enumerate(np.abs(mono)):
        a = 0.6 if v > e else 0.08
        e += a * (v - e)
        sm[n] = e
    if sm.max() > 0:
        sm = (sm / sm.max()) ** 0.6
    return stereo, int(sr), sm.astype(np.float32)

# ------------------------------------------------------------
class RectAvatar(tk.Frame):
    def __init__(self, master, pad=8):
        super().__init__(master, bg=BG_SILENT)
        self.canvas = tk.Canvas(self, bg=BG_SILENT, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.pad = pad
        self.level = 0
        self._last_time = time.perf_counter()
        self._particles = []
        self.canvas.bind("<Configure>", lambda e: self.redraw())

    def set_level(self, level):
        level = max(0, min(LEVELS-1, int(level)))
        if level != self.level:
            self.level = level

    def _hsv_to_hex(self, h, s, v):
        i = int(h * 6)
        f = h * 6 - i
        p = v * (1 - s)
        q = v * (1 - f * s)
        t = v * (1 - (1 - f) * s)
        i %= 6
        r, g, b = [(v, t, p), (q, v, p), (p, v, t),
                   (p, q, v), (t, p, v), (v, p, q)][i]
        return "#%02x%02x%02x" % (int(r*255), int(g*255), int(b*255))

    def _spawn_count(self):
        """Map current level -> rectangles per frame, with deadzone + gamma curve."""
        if self.level <= LEVEL_DEADZONE:
            return 0
        usable = LEVELS - 1 - LEVEL_DEADZONE
        if usable <= 0:
            return 0
        x = (self.level - LEVEL_DEADZONE) / float(usable)
        x = max(0.0, min(1.0, x))
        return int(0.5 + MIN_SPAWN + (SPAWN_AT_MAX_LVL - MIN_SPAWN) * (x ** SPAWN_GAMMA))

    def _spawn(self, n):
        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())
        if cw <= 2*self.pad or ch <= 2*self.pad:
            return

        # Horizontal sizing (length along X, thickness along Y)
        min_len_h = max(6, int(cw * RECT_MIN_LEN_F))
        max_len_h = max(min_len_h+4, int(cw * RECT_MAX_LEN_F))
        thick_h   = max(3, int(ch * RECT_THICK_F))

        # Vertical sizing (length along Y, thickness along X)
        min_len_v = max(6, int(ch * RECT_MIN_LEN_F))
        max_len_v = max(min_len_v+4, int(ch * RECT_MAX_LEN_F))
        thick_v   = max(3, int(cw * RECT_THICK_F))

        drift_p = max(1, int(min(cw, ch) * DRIFT_PIX_F))
        now = time.perf_counter()

        for _ in range(n):
            cx = random.randint(self.pad, cw - self.pad)
            cy = random.randint(self.pad, ch - self.pad)

            vertical = (random.random() < VERTICAL_PROPORTION)
            if vertical:
                L = random.randint(min_len_v, max_len_v)
                x1 = max(self.pad, cx - thick_v//2)
                x2 = min(cw - self.pad, cx + thick_v//2)
                y1 = max(self.pad, cy - L//2)
                y2 = min(ch - self.pad, cy + L//2)
            else:
                L = random.randint(min_len_h, max_len_h)
                x1 = max(self.pad, cx - L//2)
                x2 = min(cw - self.pad, cx + L//2)
                y1 = max(self.pad, cy - thick_h//2)
                y2 = min(ch - self.pad, cy + thick_h//2)

            vx = random.randint(-drift_p, drift_p)
            vy = random.randint(-drift_p, drift_p)
            hue = random.random()
            col = self._hsv_to_hex(hue, 0.95, 1.0)
            self._particles.append({
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "vx": vx, "vy": vy,
                "birth": now, "life": RECT_LIFETIME,
                "color": col
            })
        if len(self._particles) > MAX_PARTICLES:
            self._particles = self._particles[-MAX_PARTICLES:]

    def redraw(self):
        now = time.perf_counter()
        dt = max(0.0, now - self._last_time)
        self._last_time = now

        if self.level <= 0:
            self._particles.clear()
            self.canvas.delete("all")
            return

        spawn = self._spawn_count()
        if spawn > 0:
            self._spawn(spawn)

        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())
        alive = []
        self.canvas.delete("all")
        for p in self._particles:
            age = now - p["birth"]
            if age > p["life"]:
                continue
            p["x1"] += p["vx"] * dt
            p["x2"] += p["vx"] * dt
            p["y1"] += p["vy"] * dt
            p["y2"] += p["vy"] * dt

            pad = self.pad
            if p["x1"] < pad:
                s = pad - p["x1"]; p["x1"] += s; p["x2"] += s
            if p["x2"] > cw - pad:
                s = (cw - pad) - p["x2"]; p["x1"] += s; p["x2"] += s
            if p["y1"] < pad:
                s = pad - p["y1"]; p["y1"] += s; p["y2"] += s
            if p["y2"] > ch - pad:
                s = (ch - pad) - p["y2"]; p["y1"] += s; p["y2"] += s

            t = age / p["life"]
            stipples = ("", "gray12", "gray25", "gray50", "gray75")
            idx = min(len(stipples)-1, int(t * (len(stipples))))
            stipple = stipples[idx]
            self.canvas.create_rectangle(
                int(p["x1"]), int(p["y1"]), int(p["x2"]), int(p["y2"]),
                fill=p["color"], outline=p["color"],
                stipple=stipple if stipple else None
            )
            alive.append(p)
        self._particles = alive

# ------------------------------------------------------------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Audio-Driven Rect Avatar (Black BG)")
        self.configure(bg=BG_SILENT)
        self.geometry("900x900")
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        self.avatar = RectAvatar(self, pad=8)
        self.avatar.grid(row=0, column=0, padx=12, pady=12, sticky="nsew")

        bar = tk.Frame(self, bg=self["bg"])
        bar.grid(row=1, column=0, sticky="ew", padx=12, pady=(0,12))
        bar.columnconfigure(1, weight=1)

        tk.Label(bar, text="Output Device:", bg=self["bg"], fg="white").grid(row=0, column=0, sticky="w")
        outs = list_output_devices()
        self.dev_var = tk.StringVar()
        options = [f"{i}: {name} (sr={int(sr) if sr else '?'}Hz)" for i, name, sr in outs] or ["Default"]
        menu = tk.OptionMenu(bar, self.dev_var, *options)
        menu.configure(bg="white")
        menu.grid(row=0, column=1, sticky="ew")
        self.dev_var.set(options[0])

        tk.Button(bar, text="Play WAV", command=self.play_wav, bg="white").grid(row=1, column=0, sticky="w", pady=(8,0))
        tk.Button(bar, text="Stop",     command=self.stop_audio, bg="white").grid(row=1, column=1, sticky="e", pady=(8,0))

        self._timer = None
        self._env = None
        self._sr = 44100
        self._start = None
        self._playing = False
        self._tick_draw = self.after(FPS_MS, self._frame_tick)

    def _frame_tick(self):
        self.avatar.redraw()
        self._tick_draw = self.after(FPS_MS, self._frame_tick)

    def _pick_device(self):
        sel = self.dev_var.get()
        if not sel or sel == "Default":
            return None, 44100
        idx = int(sel.split(":")[0])
        try:
            sr = int(sd.query_devices(idx).get("default_samplerate", 44100))
        except Exception:
            sr = 44100
        return idx, sr

    def _start_playback(self, audio, sr):
        self.stop_audio()
        idx, _ = self._pick_device()
        sd.default.device = (None, idx)
        try:
            sd.play(audio, samplerate=sr, blocking=False)
            print(f"[audio] playing on device {idx} @ {sr}Hz")
        except Exception as e:
            print("[audio] play failed:", e)
        self._sr = sr
        self._start = time.perf_counter()
        self._playing = True
        self._timer = self.after(FPS_MS, self._tick_env)

    def play_wav(self):
        try:
            audio, sr, env = load_wav_and_envelope(WAV_PATH)
        except Exception as e:
            print(f"[wav] load failed: {e}")
            return
        self._env = env
        self._start_playback(audio, sr)

    def _tick_env(self):
        if not self._playing or self._env is None:
            return
        t = time.perf_counter() - self._start
        i = int(t * self._sr)
        if i >= len(self._env):
            self.stop_audio()
            return
        e = min(max(float(self._env[i]), 0.0), 1.0)
        level = int(e * (LEVELS - 1) + 1e-6)
        self.avatar.set_level(level)
        self._timer = self.after(FPS_MS, self._tick_env)

    def stop_audio(self):
        if self._timer:
            try: self.after_cancel(self._timer)
            except Exception: pass
            self._timer = None
        try: sd.stop()
        except Exception: pass
        self._playing = False
        self.avatar.set_level(0)

# ------------------------------------------------------------
if __name__ == "__main__":
    print("\n=== Output devices ===")
    for i, name, sr in list_output_devices():
        print(f"{i:>3}: {name} (sr={int(sr) if sr else '?'}Hz)")
    print("======================\n")
    App().mainloop()
