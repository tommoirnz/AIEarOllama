# === Part 1: Imports, Config, Cleaners, LaTeX Window =========================
import os, json, threading, time, re, math
from collections import deque
import numpy as np
import soundfile as sf
import sounddevice as sd
import tkinter as tk
from tkinter import ttk, messagebox
from io import BytesIO
from PIL import Image, ImageTk
import matplotlib
matplotlib.rcParams["figure.max_open_warning"] = 0
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tkinter.font as tkfont
from tkinter.scrolledtext import ScrolledText

# This is probably the best version to date with a choice of 3 avatars
# No Vision in this program though, only text based AI

try:
    import torch
    print("[GPU torch]", torch.cuda.is_available(),
          torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
except Exception:
    pass

try:
    import ctranslate2
    print("[CT2 CUDA types]", ctranslate2.get_supported_compute_types("cuda"))
except Exception:
    pass

# External modules you provide
from audio_io import list_input_devices, VADListener
from asr_whisper import ASR
from qwen_llm3 import QwenLLM
from pydub import AudioSegment


# ---------- Config ----------
def load_cfg():
    import os, json
    env_path = os.environ.get("APP_CONFIG")
    if env_path and os.path.exists(env_path):
        path = env_path
    else:
        base = os.path.dirname(os.path.abspath(__file__))
        c1 = os.path.join(base, "config.json")
        c2 = os.path.join(base, "config.example.json")
        c3 = "config.json" if os.path.exists("config.json") else None
        c4 = "config.example.json" if os.path.exists("config.example.json") else None
        path = next((p for p in (c1, c2, c3, c4) if p and os.path.exists(p)), None)

    if not path:
        raise FileNotFoundError("No config.json or config.example.json found")

    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    print(f"[cfg] loaded: {os.path.abspath(path)}")
    sp = cfg.get("system_prompt", cfg.get("qwen_system_prompt", ""))
    print(f"[cfg] system_prompt present: {bool(sp)} (len={len(sp) if isinstance(sp,str) else 'n/a'})")
    return cfg


# ---------- TTS cleaner ----------
def clean_for_tts(text: str) -> str:
    """
    Collapse ANY LaTeX/math to the word 'equation' and do light cleanup.
    Supports: ```math```, \[...\], $$...$$, \(...\), $...$
    """
    math_pat = re.compile(
        r"```(?:math|latex)\s+.+?```"
        r"|\\\[(.+?)\\\]"
        r"|\$\$(.+?)\$\$"
        r"|\\\((.+?)\\\)"
        r"|\$(.+?)\$",
        flags=re.DOTALL | re.IGNORECASE,
    )
    text = math_pat.sub(" equation ", text)
    text = re.sub(r"[#*_`~>\[\]\(\)]", "", text)
    text = re.sub(r":[a-z_]+:", "", text)
    text = re.sub(r"^[QAqa]:\s*", "", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


# ---------- LaTeX delimiter normalization ----------

# === LaTeX Window (tight inline layout, default 12 / 8 pt) ===================
# === LaTeX Window (tight inline layout, default 12 / 8 pt) ===================
class LatexWindow(tk.Toplevel):
    def __init__(self, master, log_fn=None, text_family="Segoe UI", text_size=12, math_pt=8):
        super().__init__(master)
        self.title("LaTeX Preview")
        self.protocol("WM_DELETE_WINDOW", self.hide)
        self.geometry("560x400")
        self._log = log_fn or (lambda msg: None)

        # ✅ Initialize defaults before creating IntVars
        self.text_family = text_family
        self.text_size = int(text_size)
        self.math_pt = int(math_pt)

        self._last_text = ""
        self._img_refs = []
        self._text_font = tkfont.Font(family=self.text_family, size=self.text_size)
        self._usetex_checked = False
        self._usetex_available = False
        self.show_raw = tk.BooleanVar(value=False)

        # --- Container ---
        container = ttk.Frame(self)
        container.pack(fill="both", expand=True)

        # --- Top bar ---
        topbar = ttk.Frame(container)
        topbar.pack(fill="x", padx=6, pady=(6, 2))

        ttk.Checkbutton(
            topbar, text="Show raw LaTeX",
            variable=self.show_raw,
            command=lambda: self.show_document(self._last_text or "")
        ).pack(side="left")

        ttk.Button(topbar, text="Copy Raw LaTeX", command=self.copy_raw_latex).pack(side="left", padx=(8, 6))
        ttk.Button(topbar, text="LaTeX Diagnostics", command=self._run_latex_diagnostics).pack(side="left")

        ttk.Label(topbar, text="Text pt").pack(side="left", padx=(12, 2))
        self.text_pt_var = tk.IntVar(value=self.text_size)
        txt_spin = ttk.Spinbox(
            topbar, from_=8, to=48, width=4,
            textvariable=self.text_pt_var,
            command=lambda: self.set_text_font(size=self.text_pt_var.get())
        )
        txt_spin.pack(side="left")

        ttk.Label(topbar, text="Math pt").pack(side="left", padx=(12, 2))
        self.math_pt_var = tk.IntVar(value=self.math_pt)
        math_spin = ttk.Spinbox(
            topbar, from_=6, to=64, width=4,
            textvariable=self.math_pt_var,
            command=lambda: self.set_math_pt(self.math_pt_var.get())
        )
        math_spin.pack(side="left")

        txt_spin.bind("<Return>", lambda _e: self.set_text_font(size=self.text_pt_var.get()))
        math_spin.bind("<Return>", lambda _e: self.set_math_pt(self.math_pt_var.get()))

        # --- Text + Scrollbar ---
        textwrap = ttk.Frame(container)
        textwrap.pack(fill="both", expand=True)
        self.textview = tk.Text(textwrap, bg="#ffffff", fg="#111", wrap="word", undo=False)
        vbar = ttk.Scrollbar(textwrap, orient="vertical", command=self.textview.yview)
        self.textview.configure(yscrollcommand=vbar.set)
        vbar.pack(side="right", fill="y")
        self.textview.pack(side="left", fill="both", expand=True)

        self.textview.configure(font=self._text_font, state="normal")
        self.textview.bind("<Key>", self._block_keys)
        self.textview.bind("<<Paste>>", lambda e: "break")
        self.textview.bind("<Control-v>", lambda e: "break")
        self.textview.bind("<Control-x>", lambda e: "break")
        self.textview.bind("<Control-c>", lambda e: None)
        self.textview.bind("<Control-a>", self._select_all)

        # --- Context menu ---
        self._menu = tk.Menu(self, tearoff=0)
        self._menu.add_command(label="Copy", command=lambda: self.textview.event_generate("<<Copy>>"))
        self._menu.add_command(label="Select All", command=lambda: self._select_all(None))
        self._menu.add_separator()
        self._menu.add_command(label="Copy Raw LaTeX", command=self.copy_raw_latex)
        self.textview.bind("<Button-3>", self._popup_menu)
        self.textview.bind("<Button-2>", self._popup_menu)

        # --- Highlight tags + tighter spacing ---
        self.textview.tag_configure("speak", background="#fff3a3")
        self.textview.tag_configure("normal", background="")
        self.textview.tag_configure("tight", spacing1=1, spacing3=1)

        self.withdraw()

    # ---------- Helpers ----------
    def _block_keys(self, e):
        if (e.state & 0x4) and e.keysym.lower() in ("c", "a"):
            return None
        return "break"

    def _select_all(self, _):
        self.textview.tag_add("sel", "1.0", "end-1c")
        return "break"

    def _popup_menu(self, event):
        try:
            self._menu.tk_popup(event.x_root, event.y_root)
        finally:
            self._menu.grab_release()

    def copy_raw_latex(self):
        try:
            self.clipboard_clear()
            self.clipboard_append(self._last_text or "")
            self._log("[latex] raw source copied to clipboard")
        except Exception as e:
            self._log(f"[latex] copy raw failed: {e}")

    def show(self): self.deiconify(); self.lift()
    def hide(self): self.withdraw()
    def clear(self): self.textview.delete("1.0", "end"); self._img_refs.clear()

    def set_text_font(self, family=None, size=None):
        if family is not None: self.text_family = family
        if size is not None: self.text_size = int(size)
        try:
            self._text_font.config(family=self.text_family, size=self.text_size)
            self.textview.configure(font=self._text_font)
        except Exception as e:
            self._log(f"[latex] set_text_font error: {e}")

    def set_math_pt(self, pt: int):
        try: self.math_pt = int(pt)
        except Exception as e: self._log(f"[latex] set_math_pt error: {e}")

    def _is_inline_math(self, expr: str) -> bool:
        s = expr.strip()
        if "\n" in s: return False
        if re.search(r"\\begin\{.*?\}", s): return False
        if re.search(r"\\(pmatrix|bmatrix|Bmatrix|vmatrix|Vmatrix|matrix|cases)\b", s): return False
        if len(s) > 80: return False
        return True

    def _needs_latex_engine(self, s: str) -> bool:
        return bool(re.search(
            r"(\\begin\{(?:bmatrix|pmatrix|Bmatrix|vmatrix|Vmatrix|matrix|cases|smallmatrix)\})"
            r"|\\boxed\s*\(" r"|\\boxed\s*\{"
            r"|\\text\s*\{"  r"|\\overset\s*\{" r"|\\underset\s*\{",
            s, flags=re.IGNORECASE
        ))

    def _render_with_engine(self, latex: str, fontsize: int, dpi: int, use_usetex: bool):
        preamble = r"\usepackage{amsmath,amssymb,bm}"
        rc = {'text.usetex': bool(use_usetex)}
        if use_usetex: rc['text.latex.preamble'] = preamble
        fig = plt.figure(figsize=(1, 1), dpi=dpi)
        try:
            with matplotlib.rc_context(rc):
                ax = fig.add_axes([0, 0, 1, 1])
                ax.axis("off")
                ax.text(0.5, 0.5, f"${latex}$", ha="center", va="center", fontsize=fontsize)
                buf = BytesIO()
                fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.02, transparent=True)
                return buf.getvalue()
        finally:
            plt.close(fig)
            plt.close('all')

    def _probe_usetex(self):
        if self._usetex_checked: return
        self._usetex_checked = True
        try:
            _ = self._render_with_engine(r"\begin{pmatrix}1&2\\3&4\end{pmatrix}", 10, 100, use_usetex=True)
            self._usetex_available = True
            self._log("[latex] usetex available")
        except Exception as e:
            self._usetex_available = False
            self._log(f"[latex] usetex not available ({e}); fallback to MathText")

    def render_png_bytes(self, latex, fontsize=None, dpi=200):
        fontsize = fontsize or self.math_pt
        expr = latex.strip()
        needs_tex = self._needs_latex_engine(expr)
        if needs_tex and not self._usetex_checked:
            self._probe_usetex()
        prefer_usetex = self._usetex_available and (needs_tex or "\\begin{pmatrix" in expr or "\\frac" in expr or "\\sqrt" in expr)
        expr = expr.replace("\n", " ")
        try:
            return self._render_with_engine(expr, fontsize, dpi, use_usetex=prefer_usetex)
        except Exception:
            return self._render_with_engine(expr, fontsize, dpi, use_usetex=False)

    def split_text_math(self, text):
        if not text: return []
        pattern = re.compile(
            r"""
            ```(?:math|latex)\s+(.+?)```   |   # fenced code block
            \\\[(.+?)\\\]                  |   # \[ ... \]
            \$\$(.+?)\$\$                  |   # $$ ... $$
            \\\((.+?)\\\)                  |   # \( ... \)
            \$(.+?)\$                          # $ ... $
            """,
            flags=re.DOTALL | re.IGNORECASE | re.VERBOSE
        )
        out, idx = [], 0
        for m in pattern.finditer(text):
            s, e = m.span()
            if s > idx: out.append(("text", text[idx:s]))
            latex_expr = next(g for g in m.groups() if g is not None)
            out.append(("math", latex_expr.strip()))
            idx = e
        if idx < len(text): out.append(("text", text[idx:]))
        return out

    def show_document(self, text, wrap=900):
        self._last_text = text or ""
        self.clear()
        if not text: return
        try:
            blocks = self.split_text_math(text)
        except Exception as e:
            self._log(f"[latex] split error: {e} — plain text")
            self.textview.insert("end", text, ("normal", "tight"))
            return

        raw_mode = bool(self.show_raw.get())
        for kind, content in blocks:
            if kind == "text":
                self.textview.insert("end", content, ("normal", "tight"))
                continue
            if raw_mode:
                self.textview.insert("end", f" \\[{content}\\] ", ("normal", "tight"))
                continue
            try:
                inline = self._is_inline_math(content)
                fsz = max(6, self.math_pt - 2) if inline else self.math_pt
                png = self.render_png_bytes(content, fontsize=fsz)
                img = Image.open(BytesIO(png)).convert("RGBA")
                bbox = img.getbbox()
                if bbox:
                    img = img.crop(bbox)
                max_w = max(450, int(self.winfo_width() * 0.85))
                if img.width > max_w:
                    scale = max_w / img.width
                    img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                self._img_refs.append(photo)
                if inline:
                    self.textview.image_create("end", image=photo, align="baseline")
                else:
                    self.textview.insert("end", "\n", ("tight",))
                    self.textview.image_create("end", image=photo, align="center")
                    self.textview.insert("end", "\n", ("tight",))
            except Exception as e:
                self._log(f"[latex] render error (block): {e} — raw fallback")
                self.textview.insert("end", f" \\[{content}\\] ", ("normal", "tight"))
        self.textview.insert("end", "\n")
        self._prepare_word_spans()

    # ---------- Highlight helpers ----------
    def _word_spans(self):
        content = self.textview.get("1.0", "end-1c")
        spans = []
        for m in re.finditer(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?", content):
            s, e = m.span()
            spans.append((f"1.0+{s}c", f"1.0+{e}c"))
        return spans

    def _prepare_word_spans(self):
        try:
            self._hi_spans = self._word_spans()
            self._hi_n = len(self._hi_spans)
        except Exception:
            self._hi_spans, self._hi_n = [], 0

    def set_highlight_index(self, i: int):
        if not getattr(self, "_hi_spans", None):
            return
        i = max(0, min(i, self._hi_n - 1))
        s, e = self._hi_spans[i]
        self.textview.tag_remove("speak", "1.0", "end")
        self.textview.tag_add("speak", s, e)
        self.textview.see(s)

    def set_highlight_ratio(self, r: float):
        if not getattr(self, "_hi_spans", None):
            return
        if r <= 0:
            idx = 0
        elif r >= 1:
            idx = self._hi_n - 1
        else:
            idx = int(r * self._hi_n)
        self.set_highlight_index(idx)

    def clear_highlight(self):
        self.textview.tag_remove("speak", "1.0", "end")

    def _run_latex_diagnostics(self):
        try:
            import shutil, platform, subprocess
            from matplotlib import __version__ as mpl_ver
            self._log(f"[diag] Matplotlib {mpl_ver} on {platform.system()} {platform.release()}")
            for tool in ("latex", "pdflatex", "dvipng"):
                path = shutil.which(tool)
                self._log(f"[diag] which {tool}: {path or '(not found)'}")
            gs_path = (
                shutil.which("gswin64c") or shutil.which("gswin32c") or
                shutil.which("gs") or shutil.which("ghostscript")
            )
            if gs_path:
                self._log(f"[diag] Ghostscript found: {gs_path}")
                try:
                    out = subprocess.check_output([gs_path, "--version"], text=True, stderr=subprocess.STDOUT)
                    self._log(f"[diag] Ghostscript version: {out.strip()}")
                except Exception as e:
                    self._log(f"[diag] (warning) Ghostscript version query failed: {e}")
            else:
                self._log("[diag] Ghostscript not found")
            self._usetex_checked = False
            self._probe_usetex()
        except Exception as e:
            self._log(f"[diag] diagnostics failed: {e}")

# === Part 2: Lightweight Echo + Avatar ======================================

# --- Ultra-light Echo (single feedback delay) ---
def _read_wav_mono(path):
    x, sr = sf.read(path)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x.astype(np.float32), sr

def _write_wav(path, y, sr):
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 1.0:
        y = y / peak
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sf.write(path, y.astype(np.float32), sr)

class EchoEngine:
    """
    Ultra-light 'sci-fi echo': y[n] = dry*x[n] + wet*(x[n] + fb*y[n-D])
    Controls:
      - delay_ms (echo spacing)
      - intensity (maps to feedback & wet)
    """
    def __init__(self):
        self.enabled = False
        self.delay_ms = 180.0
        self.intensity = 0.55  # 0..1
        self.dry = 0.70

    def _coeffs(self):
        # Map intensity -> (feedback, wet) smoothly but safely stable
        fb = min(0.90, max(0.05, 0.15 + 0.8 * self.intensity))
        wet = min(0.95, max(0.05, 0.2 + 0.7 * self.intensity))
        return fb, wet

    def process_array(self, x, sr):
        if not self.enabled:
            return x.astype(np.float32)

        D = max(1, int(sr * self.delay_ms / 1000.0))
        fb, wet = self._coeffs()
        dry = float(self.dry)

        y = np.zeros_like(x, dtype=np.float32)
        # simple IIR echo; use integer delay for speed
        for n in range(len(x)):
            x_n = float(x[n])
            y_n = x_n
            if n - D >= 0:
                y_n += fb * y[n - D]
            y[n] = dry * x_n + wet * y_n

        peak = float(np.max(np.abs(y))) if y.size else 0.0
        if peak > 1.0:
            y = y / peak
        return y.astype(np.float32)

    def process_file(self, in_wav, out_wav):
        x, sr = _read_wav_mono(in_wav)
        y = self.process_array(x, sr)
        _write_wav(out_wav, y, sr)
        return out_wav, sr


class EchoWindow(tk.Toplevel):
    """Tiny control panel: Enable, Delay (ms), Intensity."""
    def __init__(self, master, engine: EchoEngine):
        super().__init__(master)
        self.engine = engine
        self.title("Sci-Fi Echo")
        self.geometry("360x180")
        self.protocol("WM_DELETE_WINDOW", self.withdraw)
        self.build_ui()
        self.withdraw()

    def _slider(self, parent, text, vmin, vmax, var, fmt="{:.1f}"):
        row = ttk.Frame(parent); row.pack(fill="x", padx=8, pady=4)
        ttk.Label(row, text=text, width=16).pack(side="left")
        s = ttk.Scale(row, from_=vmin, to=vmax, orient="horizontal", variable=var)
        s.pack(side="left", fill="x", expand=True, padx=8)
        lab = ttk.Label(row, width=8); lab.pack(side="right")
        def update(*_): lab.config(text=fmt.format(var.get()))
        var.trace_add("write", lambda *_: update()); update()
        return s

    def build_ui(self):
        e = self.engine
        wrap = ttk.Frame(self); wrap.pack(fill="both", expand=True, padx=8, pady=8)

        self.v_enabled = tk.BooleanVar(value=e.enabled)
        ttk.Checkbutton(wrap, text="Enable Echo", variable=self.v_enabled, command=self._apply).pack(anchor="w")

        self.v_delay = tk.DoubleVar(value=e.delay_ms)
        self._slider(wrap, "Delay (ms)", 60.0, 480.0, self.v_delay, "{:.0f}")

        self.v_inten = tk.DoubleVar(value=e.intensity)
        self._slider(wrap, "Intensity", 0.0, 1.0, self.v_inten, "{:.2f}")

        btns = ttk.Frame(wrap); btns.pack(fill="x", pady=(4, 0))
        ttk.Button(btns, text="Apply", command=self._apply).pack(side="left")
        ttk.Button(btns, text="Hide",  command=self.withdraw).pack(side="right")

        for v in (self.v_delay, self.v_inten):
            v.trace_add("write", lambda *_: self._apply())

    def _apply(self):
        e = self.engine
        e.enabled   = bool(self.v_enabled.get())
        e.delay_ms  = float(self.v_delay.get())
        e.intensity = float(self.v_inten.get())


# === Avatar Window ===========================================================
class CircleAvatarWindow(tk.Toplevel):
    LEVELS = 32
    MAX_RINGS = 32
    BG = "#000000"

    def __init__(self, master):
        super().__init__(master)
        self.title("Avatar")
        self.configure(bg=self.BG)
        self.geometry("480x480")
        self.protocol("WM_DELETE_WINDOW", self.hide)
        self.canvas = tk.Canvas(self, bg=self.BG, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda e: self.redraw())
        self.pad = 8
        self.level = 0
        self._t0 = time.perf_counter()
        self._color_timer = None
        self._running = True
        self.start_color_timer()

    def show(self):
        self.deiconify(); self.lift()

    def hide(self):
        self.withdraw()

    def destroy(self):
        self._running = False
        try:
            if self._color_timer is not None:
                self.after_cancel(self._color_timer)
        except Exception:
            pass
        super().destroy()

    def start_color_timer(self):
        if not self._running:
            return
        self.redraw()
        self._color_timer = self.after(50, self.start_color_timer)

    def set_level(self, level: int):
        level = max(0, min(self.LEVELS - 1, int(level)))
        if level != self.level:
            self.level = level
            self.redraw()

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

    def _ring_color(self, k, rings):
        t = (time.perf_counter() - self._t0) * 0.05
        x = ((k / max(1, rings - 1)) + t) % 1.0
        return self._hsv_to_hex(x, 0.9, 1.0)

    def redraw(self):
        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())
        cx, cy = cw // 2, ch // 2
        avail = max(1, min(cw, ch) - 2 * self.pad)
        rings = max(1, int((self.level / float(self.LEVELS - 1)) * self.MAX_RINGS + 0.5))
        r_min = max(1, int(avail * 0.012))
        r_max = int(avail * 0.49)
        r_outer = int(r_min + (r_max - r_min) * (self.level / float(self.LEVELS - 1)))
        stroke = max(2, int(avail * 0.009))
        self.canvas.delete("all")
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
class RectAvatarWindow(tk.Toplevel):
    LEVELS = 32
    BG = "#000000"

    # --- visual tuning ---
    MAX_PARTICLES     = 450
    SPAWN_AT_MAX_LVL  = 60
    RECT_MIN_LEN_F    = 0.03
    RECT_MAX_LEN_F    = 0.22
    RECT_THICK_F      = 0.012
    RECT_LIFETIME     = 0.9
    DRIFT_PIX_F       = 0.01

    # spawn shaping (few at low volume, many at high)
    LEVEL_DEADZONE    = 2
    SPAWN_GAMMA       = 1.8
    MIN_SPAWN         = 0

    def __init__(self, master):
        super().__init__(master)
        self.title("Avatar — Rectangles")
        self.configure(bg=self.BG)
        self.geometry("480x480")
        self.protocol("WM_DELETE_WINDOW", self.hide)

        self.canvas = tk.Canvas(self, bg=self.BG, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda e: self.redraw())

        self.pad = 8
        self.level = 0
        self._last_time = time.perf_counter()
        self._particles = []
        self._run = True

        # repaint loop
        self._tick = self.after(16, self._loop)

    def destroy(self):
        self._run = False
        try:
            if self._tick: self.after_cancel(self._tick)
        except Exception:
            pass
        super().destroy()

    def show(self):
        self.deiconify(); self.lift()

    def hide(self):
        self.withdraw()

    def set_level(self, level: int):
        self.level = max(0, min(self.LEVELS - 1, int(level)))

    # color util
    def _hsv_to_hex(self, h, s, v):
        i = int(h * 6); f = h * 6 - i
        p = v * (1 - s); q = v * (1 - f * s); t = v * (1 - (1 - f) * s)
        i %= 6
        r, g, b = [(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)][i]
        return "#%02x%02x%02x" % (int(r*255), int(g*255), int(b*255))

    def _spawn_count(self):
        if self.level <= self.LEVEL_DEADZONE:
            return 0
        usable = self.LEVELS - 1 - self.LEVEL_DEADZONE
        if usable <= 0: return 0
        x = (self.level - self.LEVEL_DEADZONE) / float(usable)
        x = max(0.0, min(1.0, x))
        return int(0.5 + self.MIN_SPAWN + (self.SPAWN_AT_MAX_LVL - self.MIN_SPAWN) * (x ** self.SPAWN_GAMMA))

    def _spawn(self, n):
        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())
        if cw <= 2*self.pad or ch <= 2*self.pad: return

        min_len = max(6, int(cw * self.RECT_MIN_LEN_F))
        max_len = max(min_len+4, int(cw * self.RECT_MAX_LEN_F))
        thick   = max(3, int(ch * self.RECT_THICK_F))
        drift_p = max(1, int(min(cw, ch) * self.DRIFT_PIX_F))

        now = time.perf_counter()
        for _ in range(n):
            cx = np.random.randint(self.pad, cw - self.pad)
            cy = np.random.randint(self.pad, ch - self.pad)
            L  = np.random.randint(min_len, max_len)
            x1 = max(self.pad, cx - L//2); x2 = min(cw - self.pad, cx + L//2)
            y1 = max(self.pad, cy - thick//2); y2 = min(ch - self.pad, cy + thick//2)
            vx = np.random.randint(-drift_p, drift_p); vy = np.random.randint(-drift_p, drift_p)
            col = self._hsv_to_hex(np.random.random(), 0.95, 1.0)
            self._particles.append({"x1":x1,"y1":y1,"x2":x2,"y2":y2,"vx":vx,"vy":vy,"birth":now,"life":self.RECT_LIFETIME,"color":col})

        if len(self._particles) > self.MAX_PARTICLES:
            self._particles = self._particles[-self.MAX_PARTICLES:]

    def _loop(self):
        if not self._run: return
        self.redraw()
        self._tick = self.after(16, self._loop)

    def redraw(self):
        now = time.perf_counter()
        dt = max(0.0, now - self._last_time)
        self._last_time = now

        if self.level <= 0:
            self._particles.clear()
            self.canvas.delete("all")
            return

        spawn = self._spawn_count()
        if spawn > 0: self._spawn(spawn)

        cw = max(1, self.canvas.winfo_width()); ch = max(1, self.canvas.winfo_height())
        self.canvas.delete("all")
        alive = []
        for p in self._particles:
            age = now - p["birth"]
            if age > p["life"]: continue
            p["x1"] += p["vx"] * dt; p["x2"] += p["vx"] * dt
            p["y1"] += p["vy"] * dt; p["y2"] += p["vy"] * dt
            pad = self.pad
            if p["x1"] < pad: s = pad - p["x1"]; p["x1"] += s; p["x2"] += s
            if p["x2"] > cw - pad: s = (cw - pad) - p["x2"]; p["x1"] += s; p["x2"] += s
            if p["y1"] < pad: s = pad - p["y1"]; p["y1"] += s; p["y2"] += s
            if p["y2"] > ch - pad: s = (ch - pad) - p["y2"]; p["y1"] += s; p["y2"] += s

            t = age / p["life"]
            stipples = ("", "gray12", "gray25", "gray50", "gray75")
            idx = min(len(stipples)-1, int(t * (len(stipples))))
            stipple = stipples[idx]
            self.canvas.create_rectangle(int(p["x1"]), int(p["y1"]), int(p["x2"]), int(p["y2"]),
                                         fill=p["color"], outline=p["color"],
                                         stipple=stipple if stipple else None)
            alive.append(p)
        self._particles = alive
class RectAvatarWindow2(RectAvatarWindow):
    """
    Rectangles 2: same particle system as RectAvatarWindow,
    but each spawned rectangle is randomly vertical with probability VERTICAL_PROPORTION.
    """
    VERTICAL_PROPORTION = 0.35  # 35% vertical (long in Y, thin in X)

    def __init__(self, master):
        super().__init__(master)
        self.title("Avatar — Rectangles 2 (vertical mix)")

    def _spawn(self, n):
        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())
        if cw <= 2*self.pad or ch <= 2*self.pad:
            return

        # Horizontal sizing (length along X, thickness along Y)
        min_len_h = max(6, int(cw * self.RECT_MIN_LEN_F))
        max_len_h = max(min_len_h + 4, int(cw * self.RECT_MAX_LEN_F))
        thick_h   = max(3, int(ch * self.RECT_THICK_F))

        # Vertical sizing (length along Y, thickness along X)
        min_len_v = max(6, int(ch * self.RECT_MIN_LEN_F))
        max_len_v = max(min_len_v + 4, int(ch * self.RECT_MAX_LEN_F))
        thick_v   = max(3, int(cw * self.RECT_THICK_F))

        drift_p = max(1, int(min(cw, ch) * self.DRIFT_PIX_F))
        now = time.perf_counter()

        for _ in range(n):
            cx = np.random.randint(self.pad, cw - self.pad)
            cy = np.random.randint(self.pad, ch - self.pad)

            vertical = (np.random.random() < self.VERTICAL_PROPORTION)
            if vertical:
                L  = np.random.randint(min_len_v, max_len_v)
                x1 = max(self.pad, cx - thick_v // 2); x2 = min(cw - self.pad, cx + thick_v // 2)
                y1 = max(self.pad, cy - L // 2);       y2 = min(ch - self.pad, cy + L // 2)
            else:
                L  = np.random.randint(min_len_h, max_len_h)
                x1 = max(self.pad, cx - L // 2);       x2 = min(cw - self.pad, cx + L // 2)
                y1 = max(self.pad, cy - thick_h // 2); y2 = min(ch - self.pad, cy + thick_h // 2)

            vx = np.random.randint(-drift_p, drift_p)
            vy = np.random.randint(-drift_p, drift_p)
            col = self._hsv_to_hex(np.random.random(), 0.95, 1.0)

            self._particles.append({
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "vx": vx, "vy": vy, "birth": now, "life": self.RECT_LIFETIME,
                "color": col
            })

        if len(self._particles) > self.MAX_PARTICLES:
            self._particles = self._particles[-self.MAX_PARTICLES:]

#last part

# === Part 3: Main App + Voice Loop + Run ====================================

class App:
    def __init__(self, master):
        self.cfg = load_cfg()
        self.logln(f"[cfg] qwen_model_path -> {self.cfg.get('qwen_model_path')!r}")

        self.master = master
        master.title("Always Listening — Qwen (local)")
        master.geometry("1080x600")

        # Ensure output dir always exists
        os.makedirs("out", exist_ok=True)

        # Minimal required keys; default missing ones to safe values
        required = ["whisper_model","whisper_device","whisper_compute_type","whisper_beam_size",
                    "qwen_model_path","qwen_temperature","qwen_max_tokens","out_wav",
                    "sample_rate","frame_ms","vad_aggressiveness","min_utt_ms",
                    "max_utt_ms","silence_hang_ms"]
        missing = [k for k in required if k not in self.cfg]
        if missing:
            self.cfg.setdefault("out_wav", "out/last_reply.wav")
            self.cfg.setdefault("sample_rate", 16000)
            self.cfg.setdefault("frame_ms", 30)
            self.cfg.setdefault("vad_aggressiveness", 2)
            self.cfg.setdefault("min_utt_ms", 350)
            self.cfg.setdefault("max_utt_ms", 12000)
            self.cfg.setdefault("silence_hang_ms", 250)
            self.cfg.setdefault("qwen_temperature", 0.6)
            self.cfg.setdefault("qwen_max_tokens", 512)
            self.cfg.setdefault("barge_cooldown_s", 0.7)
            self.cfg.setdefault("barge_min_utt_chars", 3)
            self.cfg.setdefault("bargein_enable", True)
            self.cfg.setdefault("barge_trips_needed", 3)
            self.cfg.setdefault("highlight_gamma", 1.12)

        # --- UI State ---
        self.state = tk.StringVar(value="idle")
        self.running = False
        self.device_idx = tk.StringVar()
        self.out_device_idx = tk.StringVar()
        self.duplex_mode = tk.StringVar(value="Half-duplex")

        # Echo engine (replaces heavy reverb)
        self.echo_engine = EchoEngine()
        self.echo_enabled_var = tk.BooleanVar(value=False)
        self._echo_win = None

        # Ducking controls
        self.ducking_enable = tk.BooleanVar(value=True)
        self.duck_db = tk.DoubleVar(value=12.0)
        self.duck_thresh = tk.DoubleVar(value=1400.0)
        self.duck_attack = tk.IntVar(value=50)
        self.duck_release = tk.IntVar(value=250)
        self._duck_gain = 1.0
        self._duck_active = False
        self._duck_log = bool(self.cfg.get("duck_log", False))
        self._chime_played = False
        self._last_chime_ts = 0.0
        self._beep_once_guard = False

        # Top controls
        top = ttk.Frame(master); top.grid(row=0, column=0, columnspan=12, sticky="we")
        self.light = tk.Canvas(top, width=48, height=48, highlightthickness=0)
        self.circle = self.light.create_oval(4, 4, 44, 44, fill="#f1c40f", outline="")
        self.light.grid(row=0, column=0, padx=10, pady=10)

        self.start_btn = ttk.Button(top, text="Start", command=self.start)
        self.stop_btn = ttk.Button(top, text="Stop", command=self.stop, state=tk.DISABLED)
        self.reset_btn = ttk.Button(top, text="Reset Chat", command=self.reset_chat)
        self.start_btn.grid(row=0, column=1, padx=6)
        self.stop_btn.grid(row=0, column=2, padx=6)
        self.reset_btn.grid(row=0, column=3, padx=6)

        # --- Echo controls on main panel ---
        ttk.Checkbutton(top, text="Echo ON", variable=self.echo_enabled_var,
                        command=lambda: setattr(self.echo_engine, "enabled",
                                                bool(self.echo_enabled_var.get()))
                        ).grid(row=0, column=4, padx=(10,4))
        ttk.Button(top, text="Show Echo", command=self._toggle_echo_window)\
            .grid(row=0, column=5, padx=(4,10))

        # LaTeX controls
        self.latex_auto = tk.BooleanVar(value=True)
        ttk.Checkbutton(top, text="Auto LaTeX preview", variable=self.latex_auto).grid(row=0, column=6, padx=6)
        ttk.Button(top, text="Show/Hide LaTeX", command=self.toggle_latex).grid(row=0, column=7, padx=6)
        ttk.Button(top, text="Copy Raw LaTeX", command=lambda: self.latex_win.copy_raw_latex()
                   if hasattr(self, "latex_win") else None).grid(row=0, column=8, padx=(0,6))

        # Avatar toggle + selector (keeps same grid column)
        self.avatar_win = None
        self.avatar_kind = tk.StringVar(value="Rings")  # "Rings", "Rectangles", or "Rectangles 2"

        _avatar_bar = ttk.Frame(top)
        _avatar_bar.grid(row=0, column=11, padx=6, sticky="n")

        ttk.Label(_avatar_bar, text="Avatar").pack(anchor="n")
        self.avatar_combo = ttk.Combobox(
            _avatar_bar, textvariable=self.avatar_kind, state="readonly",
            width=14, values=["Rings", "Rectangles", "Rectangles 2"]
        )
        self.avatar_combo.current(0)
        self.avatar_combo.pack(pady=(2, 4), anchor="n")

        ttk.Button(_avatar_bar, text="Open/Close", command=self.toggle_avatar).pack(anchor="n")

        # (optional) hot-switch: if a window is open, recreate it when selection changes
        def _on_avatar_kind_change(_e=None):
            if self.avatar_win and self.avatar_win.winfo_exists():
                try: self.avatar_win.destroy()
                except Exception: pass
                self.avatar_win = None
                self.open_avatar()
        self.avatar_combo.bind("<<ComboboxSelected>>", _on_avatar_kind_change)

        # Mode combobox
        ttk.Label(top, text="Mode:").grid(row=0, column=9, sticky="e", padx=(12,2))
        self.mode_combo = ttk.Combobox(top, textvariable=self.duplex_mode, state="readonly", width=18,
                                       values=["Half-duplex", "Full-duplex (barge-in)"])
        self.mode_combo.current(0)
        self.mode_combo.grid(row=0, column=10, sticky="w", padx=(0,8))

        # Ducking UI
        duck = ttk.Frame(top); duck.grid(row=1, column=0, columnspan=12, padx=10, sticky="we")
        ttk.Checkbutton(duck, text="Ducking", variable=self.ducking_enable).pack(side="left", padx=(0,8))
        ttk.Label(duck, text="↓dB").pack(side="left"); ttk.Spinbox(duck, from_=0, to=36, width=3, textvariable=self.duck_db).pack(side="left", padx=(2,8))
        ttk.Label(duck, text="Thr").pack(side="left"); ttk.Spinbox(duck, from_=200, to=5000, width=5, textvariable=self.duck_thresh).pack(side="left", padx=(2,8))
        ttk.Label(duck, text="Atk/Rel ms").pack(side="left")
        ttk.Spinbox(duck, from_=5, to=300, width=4, textvariable=self.duck_attack).pack(side="left", padx=(2,2))
        ttk.Spinbox(duck, from_=20, to=1000, width=5, textvariable=self.duck_release).pack(side="left", padx=(2,8))
        ttk.Label(duck, text="Gain").pack(side="left", padx=(8,2))
        self.duck_var = tk.DoubleVar(value=100.0)
        ttk.Progressbar(duck, orient="horizontal", length=120, mode="determinate",
                        variable=self.duck_var, maximum=100.0).pack(side="left", padx=(0,8))
        self.rms_var = tk.StringVar(value="RMS: 0")
        ttk.Label(duck, textvariable=self.rms_var).pack(side="left")

        # Mic device combo
        ttk.Label(master, text="Mic device:").grid(row=2, column=0, sticky="e")
        self.dev_combo = ttk.Combobox(master, textvariable=self.device_idx, state="readonly", width=58)
        devs = list_input_devices()
        vals = [f"{i}: {n}" for i, n in devs] if devs else ["No input devices found"]
        self.dev_combo["values"] = vals; self.dev_combo.current(0)
        self.dev_combo.grid(row=2, column=1, columnspan=9, sticky="we", padx=6, pady=6)

        # Output device combo
        ttk.Label(master, text="Speaker device:").grid(row=3, column=0, sticky="e")
        out_vals = self._list_output_devices()
        self.out_combo = ttk.Combobox(master, textvariable=self.out_device_idx, state="readonly", width=58, values=out_vals)
        if out_vals: self.out_combo.current(0)
        self.out_combo.grid(row=3, column=1, columnspan=9, sticky="we", padx=6, pady=6)

        # TTS selection
        self.tts_engine = tk.StringVar(value="edge")
        ttk.Label(master, text="TTS Engine:").grid(row=4, column=0, sticky="e")
        ttk.Radiobutton(master, text="Edge", variable=self.tts_engine, value="edge").grid(row=4, column=1, sticky="w")
        ttk.Radiobutton(master, text="SAPI5", variable=self.tts_engine, value="sapi5").grid(row=4, column=2, sticky="w")

        # Edge voices
        self.edge_voice_var = tk.StringVar()
        edge_voices_fallback = [
            "en-GB-RyanNeural", "en-GB-SoniaNeural", "en-US-GuyNeural", "en-US-JennyNeural",
            "ja-JP-NanamiNeural", "ja-JP-KeitaNeural",
            "zh-CN-XiaoxiaoNeural", "zh-CN-YunxiNeural", "zh-CN-XiaoyiNeural",
            "zh-TW-HsiaoYuNeural", "zh-TW-YunJheNeural",
            "zh-HK-HiuMaanNeural", "zh-HK-WanLungNeural",
        ]
        try:
            import asyncio, edge_tts  # pip install edge-tts
            async def _list_voices():
                vm = await edge_tts.VoicesManager.create()
                names = sorted({v["Name"] for v in vm.voices if "Neural" in v.get("Name", "")})
                return names or edge_voices_fallback
            edge_voices = asyncio.run(_list_voices())
        except Exception:
            edge_voices = edge_voices_fallback

        ttk.Label(master, text="Edge Voice:").grid(row=5, column=0, sticky="e")
        self.edge_combo = ttk.Combobox(master, textvariable=self.edge_voice_var, values=edge_voices, width=60)
        self.edge_combo.set(self.cfg.get("voice", edge_voices[0] if edge_voices else ""))
        self.edge_combo.grid(row=5, column=1, columnspan=3, sticky="we", padx=6, pady=4)

        # SAPI voices
        self.sapi_voice_var = tk.StringVar()
        try:
            import pyttsx3
            eng = pyttsx3.init()
            voices = [f"{v.id} | {v.name}" for v in eng.getProperty("voices")]
        except Exception:
            voices = ["(no SAPI5 voices)"]
        ttk.Label(master, text="SAPI Voice:").grid(row=5, column=4, sticky="e")
        self.sapi_combo = ttk.Combobox(master, textvariable=self.sapi_voice_var, values=voices, width=60)
        self.sapi_combo.current(0)
        self.sapi_combo.grid(row=5, column=5, columnspan=5, sticky="we", padx=6, pady=4)

        # Text input
        from tkinter.scrolledtext import ScrolledText  # <-- put this with your imports at the top of the file

        # Text input (multiline box, about 10 lines tall)
        ttk.Label(master, text="Text input:").grid(row=6, column=0, sticky="ne", padx=(6, 0), pady=(0, 6))

        self.text_box = ScrolledText(master, width=70, height=10, wrap="word")
        self.text_box.grid(row=6, column=1, columnspan=8, sticky="we", padx=6, pady=(0, 6))

        # Send button stays on the right
        ttk.Button(master, text="Send", command=self.send_text).grid(row=6, column=9, sticky="nw", padx=6, pady=(0, 6))

        # Optional shortcut: Ctrl+Enter to send
        self.text_box.bind("<Control-Return>", lambda e: (self.send_text(), "break"))

        # Log
        ttk.Label(master, text="Log:").grid(row=7, column=0, sticky="nw", padx=6)
        self.log = tk.Text(master, height=12, width=80)
        self.log.grid(row=7, column=1, columnspan=9, sticky="nsew", padx=6)
        master.grid_rowconfigure(7, weight=1)
        master.grid_columnconfigure(9, weight=1)
        master.grid_columnconfigure(1, weight=1)
        master.grid_columnconfigure(5, weight=1)

        # === Engines ===========================================================
        self.asr = ASR(
            self.cfg["whisper_model"],
            self.cfg["whisper_device"],
            self.cfg["whisper_compute_type"],
            self.cfg["whisper_beam_size"]
        )

        import importlib, inspect, sys

        self.qwen = QwenLLM(
            model_path=self.cfg["qwen_model_path"],  # name some versions expect
            model=self.cfg["qwen_model_path"],  # name other versions expect
            temperature=self.cfg["qwen_temperature"],
            max_tokens=self.cfg["qwen_max_tokens"]
        )

        if hasattr(self.qwen, "model_path") and self.qwen.model_path:
            self.logln(f"[qwen] ✅ Using local GGUF model: {self.qwen.model_path}")
        elif hasattr(self.qwen, "model") and self.qwen.model:
            self.logln(f"[qwen] ✅ Using Ollama model: {self.qwen.model}")
        else:
            self.logln("[qwen] ⚠️ Could not detect model name.")

        mod = importlib.import_module(self.qwen.__class__.__module__)
        self.logln(f"[qwen] backend module: {mod.__name__}")
        self.logln(f"[qwen] backend file: {getattr(mod, '__file__', '?')}")
        self.logln(f"[cwd] {os.getcwd()}")
        self.logln("[sys.path 1st 5] " + " | ".join(sys.path[:5]))

        sys_prompt = self.cfg.get("system_prompt") or self.cfg.get("qwen_system_prompt") or ""
        self.qwen.system_prompt = sys_prompt
        self.logln(f"[qwen] system prompt (first 80): {sys_prompt[:80]!r}")

        # LaTeX window
        # LaTeX window (defaults now 12 / 8, overridable via config)
        DEFAULT_TEXT_PT = int(self.cfg.get("latex_text_pt", 12))
        DEFAULT_MATH_PT = int(self.cfg.get("latex_math_pt", 8))
        DEFAULT_TEXT_FAMILY = self.cfg.get("latex_text_family", "Segoe UI")

        self.latex_win = LatexWindow(
            master,
            log_fn=self.logln,
            text_family=DEFAULT_TEXT_FAMILY,
            text_size=DEFAULT_TEXT_PT,
            math_pt=DEFAULT_MATH_PT
        )

        if self.latex_auto.get():
            self.latex_win.show()

        # Apply cfg defaults to UI
        try:
            if self.cfg.get("voice"):
                self.edge_voice_var.set(self.cfg.get("voice"))
            if bool(self.cfg.get("duplex", False)):
                self.duplex_mode.set("Full-duplex (barge-in)")
            else:
                self.duplex_mode.set("Half-duplex")

            if "duck_enable" in self.cfg:
                self.ducking_enable.set(bool(self.cfg.get("duck_enable", True)))
            self.duck_db.set(float(self.cfg.get("duck_db", self.duck_db.get())))
            self.duck_attack.set(int(self.cfg.get("duck_attack_ms", self.duck_attack.get())))
            self.duck_release.set(int(self.cfg.get("duck_release_ms", self.duck_release.get())))
            self.duck_thresh.set(float(self.cfg.get("duck_thresh", self.duck_thresh.get())))

            self._bargein_enabled = bool(self.cfg.get("bargein_enable", True))
        except Exception as e:
            self.logln(f"[cfg] apply defaults error: {e}")

        try:
            if hasattr(self.qwen, "configure"):
                self.qwen.configure(
                    n_ctx=self.cfg.get("qwen_n_ctx"),
                    n_gpu_layers=self.cfg.get("qwen_n_gpu_layers"),
                    top_p=self.cfg.get("qwen_top_p"),
                    system_prompt=self.cfg.get("qwen_system_prompt")
                )
        except Exception as e:
            self.logln(f"[qwen] optional params not applied: {e}")

        # Barge-in control
        self._bargein_enabled = bool(self.cfg.get("bargein_enable", True))
        self._barge_latched = False
        self._barge_until = 0.0
        self._barge_cooldown_s = float(self.cfg.get("barge_cooldown_s", 0.7))
        self._barge_min_utt_chars = int(self.cfg.get("barge_min_utt_chars", 3))

        # State
        self.speaking_flag = False
        self.interrupt_flag = False
        self.barge_buffer = None  # set in start_bargein_mic
        self.barge_stream = None
        self.monitor_thread = None
        self._mode_last = None
        self._dev_idx = None

        # Highlight progress fields
        self._tts_total_samples = 0
        self._tts_cursor_samples = 0
        self._hi_stop = True
        self._tts_silent = False
        self._ui_last_ratio = 0.0
        self._ui_eased_ratio = 0.0
        self._ui_gamma = float(self.cfg.get("highlight_gamma", 1.12))

    # --- Thread-safe logger ---
    def logln(self, msg):
        def _append():
            try:
                self.log.insert("end", msg + "\n")
                self.log.see("end")
            except Exception:
                pass
        try:
            self.master.after(0, _append)
        except Exception:
            pass

    def _toggle_echo_window(self):
        try:
            if self._echo_win is None or not self._echo_win.winfo_exists():
                self._echo_win = EchoWindow(self.master, self.echo_engine)
            if self._echo_win.state() == "withdrawn":
                self._echo_win.deiconify(); self._echo_win.lift()
            else:
                self._echo_win.withdraw()
        except Exception as e:
            self.logln(f"[echo] window error: {e}")

    def set_light(self, mode):
        color = {"idle": "#f1c40f","listening":"#2ecc71","speaking":"#e74c3c"}.get(mode,"#f1c40f")
        self.light.itemconfig(self.circle, fill=color)
        self.state.set(mode)

    def toggle_latex(self):
        try:
            if self.latex_win.state() == "withdrawn": self.latex_win.show()
            else: self.latex_win.hide()
        except Exception: self.latex_win.show()

    # Avatar window controls
    def open_avatar(self):
        try:
            kind = self.avatar_kind.get()
            # Map selection to window class
            klass_map = {
                "Rings": CircleAvatarWindow,
                "Rectangles": RectAvatarWindow,
                "Rectangles 2": RectAvatarWindow2,
            }
            klass = klass_map.get(kind, CircleAvatarWindow)

            need_new = (
                    self.avatar_win is None
                    or not self.avatar_win.winfo_exists()
                    or not isinstance(self.avatar_win, klass)
            )
            if need_new:
                if self.avatar_win and self.avatar_win.winfo_exists():
                    try:
                        self.avatar_win.destroy()
                    except Exception:
                        pass
                self.avatar_win = klass(self.master)
            self.avatar_win.show()
        except Exception as e:
            self.logln(f"[avatar] open error: {e}")

    def close_avatar(self):
        try:
            if self.avatar_win and self.avatar_win.winfo_exists():
                self.avatar_win.hide()
        except Exception as e:
            self.logln(f"[avatar] close error: {e}")

    def toggle_avatar(self):
        try:
            if self.avatar_win is None or not self.avatar_win.winfo_exists() or self.avatar_win.state() == "withdrawn":
                self.open_avatar()
            else:
                self.close_avatar()
        except Exception as e:
            self.logln(f"[avatar] toggle error: {e}")

    def _avatar_set_level_async(self, lvl: int):
        try:
            if self.avatar_win and self.avatar_win.winfo_exists():
                self.avatar_win.set_level(lvl)
        except Exception:
            pass

    # --- Text query handler ---
    def send_text(self):
        if hasattr(self, "text_box"):
            # Read all text from the multiline box
            text = self.text_box.get("1.0", "end-1c").strip()
            self.text_box.delete("1.0", "end")
        else:
            # Fallback if you ever revert to Entry
            text = self.text_entry.get().strip()
            self.text_entry.delete(0, "end")

        if not text:
            return

        threading.Thread(target=self.handle_text_query, args=(text,), daemon=True).start()




    def handle_text_query(self, text):
        self.logln(f"[user] {text}")
        self.preview_latex(text)
        reply = self.qwen.generate(text)
        self.logln(f"[qwen] {reply}")
        self.preview_latex(reply)
        clean = clean_for_tts(reply)
        if self.synthesize_to_wav(clean, self.cfg["out_wav"]):
            self.master.after(0, self.latex_win._prepare_word_spans)
            play_path = self.cfg["out_wav"]
            if bool(self.echo_enabled_var.get()):
                try:
                    play_path, _ = self.echo_engine.process_file(self.cfg["out_wav"], "out/last_reply_echo.wav")
                    self.logln("[echo] processed -> out/last_reply_echo.wav")
                except Exception as e:
                    self.logln(f"[echo] processing failed: {e} (playing dry)")
            self.play_wav_with_interrupt(play_path)
        self.set_light("idle")

    # --- MP3 Beep (cached) ---
    def play_chime2(self, path="beep.mp3", gain_db=0.0):
        try:
            if not hasattr(self, "_beep_cache") or self._beep_cache.get("path") != path:
                seg = AudioSegment.from_file(path)
                self._beep_cache = {"path": path, "seg": seg}
            else:
                seg = self._beep_cache["seg"]
            if gain_db:
                seg = seg.apply_gain(gain_db)
            samples = np.array(seg.get_array_of_samples())
            if seg.channels > 1:
                samples = samples.reshape((-1, seg.channels))
            else:
                samples = samples.reshape((-1, 1))
            samples = samples.astype(np.float32) / (2 ** (8 * seg.sample_width - 1))
            fade = min(int(0.008 * seg.frame_rate), max(1, samples.shape[0] // 6))
            if fade > 0:
                ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32).reshape(-1, 1)
                samples[:fade] *= ramp
                samples[-fade:] *= ramp[::-1]
            sd.play(samples, seg.frame_rate, blocking=True,
                    device=self._selected_out_device_index())
        except Exception as e:
            self.logln(f"[beep mp3] {e} — fallback tone")
            self.play_chime()

    def play_chime(self, freq=880, ms=140, vol=0.20):
        try:
            fs = 16000
            n = int(fs * (ms / 1000.0))
            t = np.linspace(0, ms / 1000.0, n, endpoint=False)
            s = np.sin(2 * np.pi * freq * t).astype(np.float32)
            fade = np.linspace(0.0, 1.0, min(16, n), dtype=np.float32)
            s[:fade.size] *= fade; s[-fade.size:] *= fade[::-1]
            sd.play((vol * s).reshape(-1, 1), fs, blocking=False, device=self._selected_out_device_index())
        except Exception as e:
            self.logln(f"[beep] {e}")

    def brief_listen_prompt(self):
        if not self.cfg.get("announce_listening", True):
            return
        prev = self.speaking_flag
        try:
            self.speaking_flag = True
            self.play_chime2("beep.mp3")
        finally:
            self.speaking_flag = prev

    # --- Playback with progress-driven highlight ---
    def play_wav_with_interrupt(self, path):
        import platform as _plat
        start_time = time.monotonic()
        try:
            data, fs = sf.read(path, dtype="float32")
            if data.ndim == 1:
                data = data.reshape(-1, 1)
            if data.size == 0:
                return
            total_samples = int(data.shape[0])
            self._tts_total_samples = total_samples
            self._tts_cursor_samples = 0
            self._hi_stop = False

            target_fs = int(self.cfg.get("force_out_samplerate", fs))
            if target_fs != fs:
                try:
                    n = data.shape[0]
                    t = np.linspace(0.0, 1.0, n, endpoint=False)
                    m = int(np.ceil(n * (target_fs / fs)))
                    ti = np.linspace(0.0, 1.0, m, endpoint=False)
                    chans = data.shape[1]
                    out = []
                    for c in range(chans):
                        out.append(np.interp(ti, t, data[:, c]))
                    data = np.stack(out, axis=1).astype(np.float32)
                    fs = target_fs
                    self.logln(f"[audio] resample -> {fs} Hz for output")
                except Exception as e:
                    self.logln(f"[audio] resample failed, using original fs ({fs}): {e}")

            out_dev = self._selected_out_device_index()
            blocksize = self.cfg.get("out_blocksize", 4096)
            latency_hint = self.cfg.get("out_latency", "high")
            extra = None
            try:
                if _plat.system() == "Windows":
                    extra = sd.WasapiSettings(exclusive=False)
            except Exception:
                extra = None

            SILENCE_THRESH = 1e-4
            SILENCE_MAX_BLOCKS = 20
            cursor = 0

            def _ui_progress_tick():
                if self._hi_stop:
                    return
                try:
                    if self._tts_total_samples > 0:
                        r_raw = float(self._tts_cursor_samples) / float(self._tts_total_samples)
                        if self._tts_silent:
                            r_target = self._ui_last_ratio
                        else:
                            r_curved = r_raw ** self._ui_gamma
                            r_target = r_curved
                            self._ui_last_ratio = r_target
                        # Clamp to stop micro-oscillation
                        self._ui_last_ratio = 0.0 if self._ui_last_ratio < 0 else (1.0 if self._ui_last_ratio > 1 else self._ui_last_ratio)
                        alpha = 0.25
                        self._ui_eased_ratio += alpha * (r_target - self._ui_eased_ratio)
                        r_show = 0.0 if self._ui_eased_ratio < 0 else (1.0 if self._ui_eased_ratio > 1 else self._ui_eased_ratio)
                        try:
                            self.latex_win.set_highlight_ratio(r_show)
                        except Exception:
                            pass
                    self.master.after(33, _ui_progress_tick)
                except Exception:
                    pass

            def run_stream():
                nonlocal cursor, fs, data, blocksize, latency_hint, out_dev, extra
                silent_blocks = 0
                last_cursor_check = -1
                stall_ticks = 0
                STALL_TICKS_MAX = int(self.cfg.get("stall_ticks_max", 120))
                RESUME_FADE_SAMPLES = int(0.01 * fs)
                did_fade = False

                def cb(outdata, frames, *_):
                    nonlocal cursor, silent_blocks, did_fade
                    if self.interrupt_flag or not self.running:
                        outdata[:] = 0
                        raise sd.CallbackStop()

                    end = min(cursor + frames, data.shape[0])
                    out_frames = end - cursor
                    block = data[cursor:end]
                    avg_abs = 0.0
                    gain = float(np.clip(getattr(self, "_duck_gain", 1.0), 0.0, 1.5))

                    if out_frames > 0:
                        out = block.copy()
                        if not did_fade and cursor == 0:
                            n = min(out.shape[0], RESUME_FADE_SAMPLES)
                            if n > 0:
                                ramp = np.linspace(0.0, 1.0, n, dtype=np.float32).reshape(-1, 1)
                                out[:n] *= ramp
                            did_fade = True
                        outdata[:out_frames] = out * gain
                        avg_abs = float(np.mean(np.abs(block))) if block.size else 0.0
                        if avg_abs < SILENCE_THRESH:
                            silent_blocks += 1
                            if silent_blocks >= SILENCE_MAX_BLOCKS:
                                outdata[out_frames:] = 0
                                raise sd.CallbackStop()
                        else:
                            silent_blocks = 0

                    self._tts_silent = bool(avg_abs < SILENCE_THRESH)
                    env = min(max(avg_abs * 4.0, 0.0), 1.0) ** 0.6
                    AVATAR_LEVELS = 32  # place once near the top of the file with your other constants
                    level = int(env * (AVATAR_LEVELS - 1) + 1e-6)

                    try:
                        self.master.after(0, self._avatar_set_level_async, level)
                    except Exception:
                        pass

                    if end - cursor < frames:
                        outdata[end - cursor:] = 0
                        raise sd.CallbackStop()

                    cursor = end
                    self._tts_cursor_samples = int(cursor)

                def open_stream(extra_settings, device_idx):
                    return sd.OutputStream(
                        samplerate=fs,
                        channels=data.shape[1],
                        dtype="float32",
                        blocksize=blocksize,
                        latency=latency_hint,
                        callback=cb,
                        device=device_idx,
                        extra_settings=extra_settings,
                    )

                chosen_dev = out_dev
                hostapi_name = self._device_hostapi_name(chosen_dev)
                use_extra = bool(hostapi_name and "WASAPI" in hostapi_name)
                self.logln(
                    f"[audio] open stream dev={chosen_dev} hostapi={hostapi_name or 'default'} "
                    f"extra={'wasapi' if use_extra else 'none'}"
                )

                try:
                    ctx = open_stream(extra if use_extra else None, chosen_dev)
                except Exception:
                    self.logln("[audio] stream open failed with WASAPI; retrying without extras")
                    chosen_dev = None
                    ctx = open_stream(None, chosen_dev)

                with ctx:
                    self.master.after(0, _ui_progress_tick)
                    while self.running and not self.interrupt_flag and cursor < data.shape[0]:
                        if cursor == last_cursor_check:
                            stall_ticks += 1
                            if stall_ticks >= STALL_TICKS_MAX:
                                return False
                        else:
                            last_cursor_check = cursor
                            stall_ticks = 0
                        time.sleep(0.01)
                return True

            ok = run_stream()
            if not ok:
                remaining = data.shape[0] - cursor
                if remaining < int(0.25 * fs):
                    self.logln("[audio] stalled near end — skipping retry")
                else:
                    self.logln("[audio] output stalled — retrying with larger buffers (resume)")
                    blocksize = max(int(blocksize) if blocksize else 0, 8192)
                    latency_hint = "high"
                    run_stream()
        except Exception as e:
            self.logln(f"[warn] playback error: {e}")
        finally:
            self.speaking_flag = False
            self.interrupt_flag = False
            self.set_light("idle")
            try:
                if self.avatar_win and self.avatar_win.winfo_exists():
                    self.avatar_win.set_level(0)
            except Exception:
                pass
            self._hi_stop = True
            self._tts_silent = False
            self._ui_last_ratio = 0.0
            self._ui_eased_ratio = 0.0
            try:
                self.latex_win.set_highlight_ratio(1.0)
                self.master.after(250, self.latex_win.clear_highlight)
            except Exception:
                pass
            self._beep_once_guard = False  # ensure next listen chime is single
            dur = time.monotonic() - start_time
            self.logln(f"[audio] playback done ({dur:.2f}s)")

    def preview_latex(self, content: str):
        if not self.latex_auto.get():
            return
        def _go():
            try:
                self.latex_win.show()
                self.latex_win.show_document(content)
            except Exception as e:
                self.logln(f"[latex] preview error: {e}")
        self.master.after(0, _go)

    # --- TTS ---
    def synthesize_to_wav(self, text, out_wav):
        engine = self.tts_engine.get()
        try:
            if engine == "sapi5":
                import pyttsx3
                voice_id = self.sapi_voice_var.get().split(" | ")[0]
                tmp = out_wav + ".tmp.wav"
                eng = pyttsx3.init()
                eng.setProperty("voice", voice_id)
                eng.save_to_file(text, tmp)
                eng.runAndWait()
                os.replace(tmp, out_wav)
                self.logln(f"[tts] sapi5: {voice_id}")
                return True
            else:
                from tts_edge import say as edge_say
                v = self.edge_voice_var.get()
                edge_say(text, voice=v, rate=self.cfg.get("rate", "-10%"), out_wav=out_wav)
                self.logln(f"[tts] edge: {v}")
                return True
        except Exception as e:
            self.logln(f"[tts] error: {e}")
            try: messagebox.showerror("TTS error", str(e))
            except: pass
            return False

    def _update_duck_ui(self):
        try:
            g = float(getattr(self, '_duck_gain', 1.0))
            self.duck_var.set(100.0 * g)
            self.rms_var.set(f"RMS: {int(getattr(self, '_last_rms', 0))}")
        except Exception:
            pass

    # --- Voice loop ---
    def start(self):
        if self.running: return
        self.running = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.set_light("idle")
        threading.Thread(target=self.loop, daemon=True).start()

    def stop(self):
        self.running = False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.set_light("idle")
        try:
            if self.barge_stream and self.barge_stream.active:
                self.barge_stream.stop()
        except Exception:
            pass
        try:
            if self.barge_stream:
                self.barge_stream.close()
        except Exception:
            pass
        self.barge_stream = None

    def reset_chat(self):
        self.qwen.clear_history()
        self.qwen.system_prompt = self.cfg.get("system_prompt", "")
        self.logln("[info] Chat reset.")

    def loop(self):
        dev_choice = self.dev_combo.get()
        dev_idx = int(dev_choice.split(":")[0]) if ":" in dev_choice else None
        self._dev_idx = dev_idx
        self.logln(f"[audio] mic device={dev_idx}")
        if getattr(self, "_barge_latched", False):
            time.sleep(0.06)
        self.barge_stream, self.barge_buffer = self.start_bargein_mic(dev_idx)

        guard_half = (lambda: self.speaking_flag)
        guard_full = (lambda: False)
        echo_guard = guard_half if self.duplex_mode.get().startswith("Half") else guard_full

        use_frame_ms = 20 if self.duplex_mode.get().startswith("Full") else self.cfg["frame_ms"]
        use_vad_thresh = self.cfg.get("vad_threshold_full", 0.005) if self.duplex_mode.get().startswith("Full") else self.cfg.get("vad_threshold", 0.01)

        listener = VADListener(
            self.cfg["sample_rate"], use_frame_ms,
            self.cfg["vad_aggressiveness"], self.cfg["min_utt_ms"],
            self.cfg["max_utt_ms"], self.cfg["silence_hang_ms"],
            dev_idx, use_vad_thresh
        )
        it = listener.listen(echo_guard=echo_guard)
        self._mode_last = self.duplex_mode.get()
        self.logln(f"[mode] start as {self._mode_last}")

        self.monitor_thread = threading.Thread(target=self.monitor_interrupt, daemon=True)
        self.monitor_thread.start()
        self.logln("[info] Listening…")

        while self.running:
            cur_mode = self.duplex_mode.get()
            if cur_mode != self._mode_last:
                try:
                    guard_half = (lambda: self.speaking_flag)
                    guard_full = (lambda: False)
                    echo_guard = guard_half if cur_mode.startswith("Half") else guard_full
                    use_frame_ms = 20 if cur_mode.startswith("Full") else self.cfg["frame_ms"]
                    use_vad_thresh = (
                        self.cfg.get("vad_threshold_full", 0.005)
                        if cur_mode.startswith("Full")
                        else self.cfg.get("vad_threshold", 0.01)
                    )
                    listener = VADListener(
                        self.cfg["sample_rate"], use_frame_ms,
                        self.cfg["vad_aggressiveness"], self.cfg["min_utt_ms"],
                        self.cfg["max_utt_ms"], self.cfg["silence_hang_ms"],
                        self._dev_idx, use_vad_thresh
                    )
                    it = listener.listen(echo_guard=echo_guard)
                    self._mode_last = cur_mode
                    self.logln(f"[mode] switched to {cur_mode} (frame_ms={use_frame_ms}, vad_thresh={use_vad_thresh})")
                    self._beep_once_guard = False
                except Exception as e:
                    self.logln(f"[mode] switch error: {e}")

            if self.speaking_flag and cur_mode.startswith("Half"):
                time.sleep(0.02)
                continue

            self.set_light("listening")
            if cur_mode.startswith("Half") and not self._beep_once_guard:
                self.brief_listen_prompt()
                self._beep_once_guard = True

            try:
                utt = next(it)
            except StopIteration:
                break
            if not self.running: break
            if self.speaking_flag and self.duplex_mode.get().startswith("Half"):
                continue

            text = self.asr.transcribe(utt, self.cfg["sample_rate"])
            if not text:
                continue
            self.logln(f"[asr] {text}")

            if getattr(self, "_barge_latched", False):
                if time.monotonic() < getattr(self, "_barge_until", 0.0):
                    if len(text.strip()) < int(self.cfg.get("barge_min_utt_chars", 3)):
                        self.logln("[barge-in] suppressing tiny fragment")
                        continue
                    self.logln("[barge-in] listen-only window: suppressing LLM/TTS")
                    continue
                else:
                    self._barge_latched = False

            try:
                reply = self.qwen.generate(text)
            except Exception as e:
                self.logln(f"[llm] {e}\n[hint] Is Ollama running? Open a terminal and run:  ollama serve")
                self.set_light("idle")
                continue

            self.logln(f"[qwen] {reply}")
            self.preview_latex(reply)

            clean = clean_for_tts(reply)
            self.speaking_flag = True
            self.interrupt_flag = False
            self.set_light("speaking")
            try:
                if self.synthesize_to_wav(clean, self.cfg["out_wav"]):
                    self.master.after(0, self.latex_win._prepare_word_spans)
                    play_path = self.cfg["out_wav"]
                    if bool(self.echo_enabled_var.get()):
                        try:
                            play_path, _ = self.echo_engine.process_file(self.cfg["out_wav"], "out/last_reply_echo.wav")
                            self.logln("[echo] processed -> out/last_reply_echo.wav")
                        except Exception as e:
                            self.logln(f"[echo] processing failed: {e} (playing dry)")
                    self.play_wav_with_interrupt(play_path)
            finally:
                self.speaking_flag = False
                self.interrupt_flag = False
                self.set_light("idle")
                self._chime_played = False
                self._beep_once_guard = False

        self.stop()

    # --- Barge-in mic + monitor ---
    def start_bargein_mic(self, device_idx):
        q = deque(maxlen=64)  # cap growth
        def callback(indata, frames, time_info, status):
            if self.speaking_flag:
                q.append(np.copy(indata))
        stream = sd.InputStream(
            device=device_idx, samplerate=self.cfg["sample_rate"],
            channels=1, dtype="float32", blocksize=1024, callback=callback
        )
        stream.start()
        return stream, q

    def monitor_interrupt(self):
        import numpy as _np, time as _time
        threshold_interrupt = self.cfg.get("bargein_threshold", 1500)
        trips_needed = int(self.cfg.get("barge_trips_needed", 3))
        trips = 0
        dt = 0.05
        while self.running:
            if self.speaking_flag and self.barge_buffer and len(self.barge_buffer) > 0:
                audio = _np.concatenate(list(self.barge_buffer))
                self.barge_buffer.clear()
                if audio.size == 0:
                    _time.sleep(dt); continue
                rms = _np.sqrt(_np.mean(audio ** 2)) * 32768
                self._last_rms = float(rms)

                if self._bargein_enabled:
                    if rms > threshold_interrupt:
                        trips += 1
                        if trips >= trips_needed:
                            self.logln(f"[barge-in] RMS={rms:.0f} interrupt -> latch listen-only")
                            self.interrupt_flag = True
                            import time as _t
                            self._barge_latched = True
                            self._barge_until = _t.monotonic() + self._barge_cooldown_s
                            try:
                                self.speaking_flag = False
                                self.set_light("listening")
                            except Exception:
                                pass
                            trips = 0
                    else:
                        trips = max(trips - 1, 0)

                if self.ducking_enable.get():
                    target = 1.0
                    if rms > float(self.duck_thresh.get()):
                        target = 10 ** (-float(self.duck_db.get()) / 20.0)
                    atk = max(5, int(self.duck_attack.get())) / 1000.0
                    rel = max(20, int(self.duck_release.get())) / 1000.0
                    alpha_atk = min(1.0, dt / atk) if atk > 0 else 1.0
                    alpha_rel = min(1.0, dt / rel) if rel > 0 else 1.0
                    cur = getattr(self, "_duck_gain", 1.0)
                    if target < cur:
                        cur += (target - cur) * alpha_atk
                    else:
                        cur += (target - cur) * alpha_rel
                    self._duck_gain = float(_np.clip(cur, 0.0, 1.0))
                    active_now = self._duck_gain < 0.98
                    if self._duck_log:
                        if active_now and not self._duck_active:
                            self.logln(f"[duck] engage gain={self._duck_gain:.2f} (rms={rms:.0f})")
                        elif not active_now and self._duck_active:
                            self.logln(f"[duck] release (rms={rms:.0f})")
                    self._duck_active = bool(active_now)
                else:
                    self._duck_gain = 1.0
                    self._duck_active = False

                self.master.after(0, self._update_duck_ui)
            else:
                cur = getattr(self, "_duck_gain", 1.0)
                rel = max(20, int(self.duck_release.get())) / 1000.0 if hasattr(self, "duck_release") else 0.25
                alpha_rel = min(1.0, dt / rel) if rel > 0 else 1.0
                cur += (1.0 - cur) * alpha_rel
                self._duck_gain = float(_np.clip(cur, 0.0, 1.0))
                self._duck_active = False
                self.master.after(0, self._update_duck_ui)
                _time.sleep(dt)

    # --- Devices helpers ---
    def _device_hostapi_name(self, index):
        try:
            if index is None:
                return None
            info = sd.query_devices(index)
            hostapi_idx = info.get('hostapi', None)
            if hostapi_idx is None:
                return None
            hai = sd.query_hostapis(hostapi_idx)
            return hai.get('name')
        except Exception:
            return None

    def _list_output_devices(self):
        try:
            info = sd.query_devices()
            outs = []
            for i, d in enumerate(info):
                if d.get('max_output_channels', 0) > 0:
                    name = d.get('name', f'Device {i}')
                    outs.append(f"{i}: {name}")
            return outs if outs else ["(default output)"]
        except Exception as e:
            self.logln(f"[audio] output device query failed: {e}")
            return ["(default output)"]

    def _selected_out_device_index(self):
        try:
            choice = self.out_combo.get()
            return int(choice.split(":")[0]) if ":" in choice else None
        except Exception:
            return None

    # -------- Run --------


if __name__ == "__main__":
    root = tk.Tk()
    try:
        root.call("tk", "scaling", 1.25)
    except Exception:
        pass
    try:
        style = ttk.Style()
        style.theme_use("clam")
    except Exception:
        pass
    app = App(root)
    root.mainloop()
