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
# This uses two models and a different Json file and handles images as well
#Look for the Json file  called Json2. You will need two models as per the Json file loaded into ollama
#can read equations off paper and handwriting

matplotlib.rcParams["figure.max_open_warning"] = 0
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tkinter.font as tkfont
from tkinter.scrolledtext import ScrolledText
import base64, tempfile, requests

try:
    import cv2  # pip install opencv-python
except Exception:
    cv2 = None

# Optional drag & drop support
try:
    from tkinterdnd2 import DND_FILES, TkinterDnD  # pip install tkinterdnd2
except Exception:
    DND_FILES = None
    TkinterDnD = None

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
    print(f"[cfg] system_prompt present: {bool(sp)} (len={len(sp) if isinstance(sp, str) else 'n/a'})")
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


def purge_temp_images(folder="out"):
    """
    Remove temporary/snapshot images we create. Keeps non-image files (e.g., last_reply.wav).
    """
    try:
        if not os.path.isdir(folder):
            return
        for name in os.listdir(folder):
            low = name.lower()
            # keep audio; delete our snapshots and temp frames
            if low.startswith("snapshot_") and low.endswith(".png"):
                try:
                    os.remove(os.path.join(folder, name))
                except Exception:
                    pass
            if low in ("live_frame.png", "tmp_frame.png"):
                try:
                    os.remove(os.path.join(folder, name))
                except Exception:
                    pass
    except Exception as e:
        print(f"[startup] purge_temp_images: {e}")


# === LaTeX Window ===
class LatexWindow(tk.Toplevel):
    def __init__(self, master, log_fn=None, text_family="Segoe UI", text_size=12, math_pt=8):
        super().__init__(master)
        self.title("LaTeX Preview")
        self.protocol("WM_DELETE_WINDOW", self.hide)
        self.geometry("560x400")
        self._log = log_fn or (lambda msg: None)

        # Initialize defaults before creating IntVars
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

        # Theme defaults (must be AFTER self.textview exists)
        self._default_bg = "#ffffff"
        self._default_fg = "#111"

        self.withdraw()

    # --------- Theme toggle (used to blue-tint in vision mode) ----------
    def set_scheme(self, scheme: str):
        """Switch between default and vision mode (blue background)."""
        try:
            if scheme == "vision":
                self.textview.configure(bg="#e8f1ff", fg="#0b2545")  # soft blue scheme
            else:
                self.textview.configure(bg=self._default_bg, fg=self._default_fg)
        except Exception:
            pass

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

    def show(self):
        self.deiconify()
        self.lift()

    def hide(self):
        self.withdraw()

    def clear(self):
        self.textview.delete("1.0", "end")
        self._img_refs.clear()

    def set_text_font(self, family=None, size=None):
        if family is not None:
            self.text_family = family
        if size is not None:
            self.text_size = int(size)
        try:
            self._text_font.config(family=self.text_family, size=self.text_size)
            self.textview.configure(font=self._text_font)
        except Exception as e:
            self._log(f"[latex] set_text_font error: {e}")

    def set_math_pt(self, pt: int):
        try:
            self.math_pt = int(pt)
        except Exception as e:
            self._log(f"[latex] set_math_pt error: {e}")

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
        if use_usetex:
            rc['text.latex.preamble'] = preamble
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
        if self._usetex_checked:
            return
        self._usetex_checked = True
        try:
            _ = self._render_with_engine(r"\begin{pmatrix}1&2\\3&4\end{pmatrix}", 10, 100, use_usetex=True)
            self._usetex_available = True
            self._log("[latex] usetex available")
        except Exception as e:
            self._usetex_available = False
            self._log(f"[latex] usetex not available ({e}); fallback to MathText)")

    def render_png_bytes(self, latex, fontsize=None, dpi=200):
        fontsize = fontsize or self.math_pt
        expr = latex.strip()
        needs_tex = self._needs_latex_engine(expr)
        if needs_tex and not self._usetex_checked:
            self._probe_usetex()
        prefer_usetex = self._usetex_available and (
                    needs_tex or "\\begin{pmatrix" in expr or "\\frac" in expr or "\\sqrt" in expr)
        expr = expr.replace("\n", " ")
        try:
            return self._render_with_engine(expr, fontsize, dpi, use_usetex=prefer_usetex)
        except Exception:
            return self._render_with_engine(expr, fontsize, dpi, use_usetex=False)

    def split_text_math(self, text):
        if not text:
            return []
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
            if s > idx:
                out.append(("text", text[idx:s]))
            latex_expr = next(g for g in m.groups() if g is not None)
            out.append(("math", latex_expr.strip()))
            idx = e
        if idx < len(text):
            out.append(("text", text[idx:]))
        return out

    def show_document(self, text, wrap=900):
        self._last_text = text or ""
        self.clear()
        if not text:
            return
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


# === Echo Engine ===
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
        row = ttk.Frame(parent)
        row.pack(fill="x", padx=8, pady=4)
        ttk.Label(row, text=text, width=16).pack(side="left")
        s = ttk.Scale(row, from_=vmin, to=vmax, orient="horizontal", variable=var)
        s.pack(side="left", fill="x", expand=True, padx=8)
        lab = ttk.Label(row, width=8)
        lab.pack(side="right")

        def update(*_):
            lab.config(text=fmt.format(var.get()))

        var.trace_add("write", lambda *_: update())
        update()
        return s

    def build_ui(self):
        e = self.engine
        wrap = ttk.Frame(self)
        wrap.pack(fill="both", expand=True, padx=8, pady=8)

        self.v_enabled = tk.BooleanVar(value=e.enabled)
        ttk.Checkbutton(wrap, text="Enable Echo", variable=self.v_enabled, command=self._apply).pack(anchor="w")

        self.v_delay = tk.DoubleVar(value=e.delay_ms)
        self._slider(wrap, "Delay (ms)", 60.0, 480.0, self.v_delay, "{:.0f}")

        self.v_inten = tk.DoubleVar(value=e.intensity)
        self._slider(wrap, "Intensity", 0.0, 1.0, self.v_inten, "{:.2f}")

        btns = ttk.Frame(wrap)
        btns.pack(fill="x", pady=(4, 0))
        ttk.Button(btns, text="Apply", command=self._apply).pack(side="left")
        ttk.Button(btns, text="Hide", command=self.withdraw).pack(side="right")

        for v in (self.v_delay, self.v_inten):
            v.trace_add("write", lambda *_: self._apply())

    def _apply(self):
        e = self.engine
        e.enabled = bool(self.v_enabled.get())
        e.delay_ms = float(self.v_delay.get())
        e.intensity = float(self.v_inten.get())

# === Avatar Windows ===
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


# === Main App Class ===

# === Main App Class ===
# === Main App Class ===
class App:
    def __init__(self, master):
        self.cfg = load_cfg()

        # === Initialize logln method FIRST ===
        def logln(msg):
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

        self.logln = logln

        # NOW we can use logln
        self.logln(f"[cfg] qwen_model_path -> {self.cfg.get('qwen_model_path')!r}")

        # Vision system prompt
        self.vl_system_prompt = self.cfg.get(
            "vl_system_prompt",
            "You are a multimodal assistant. You CAN see the provided image(s). "
            "Answer using only what is visible in the image plus the user question. "
            "If asked to count, return a number. If asked to identify, name the most likely class. "
            "Do not say you are text-based."
        )
        self.vl_model = (
                self.cfg.get("vl_model")
                or self.cfg.get("vl_model_path")
                or "qwen2.5-vl:7b"
        )

        self.master = master
        master.title("Always Listening — Qwen (local)")
        master.geometry("1080x600")

        # Ensure output dir exists
        os.makedirs("out", exist_ok=True)
        purge_temp_images("out")

        # === NEW: Unified playback fencing ===
        self._play_lock = threading.Lock()
        self._play_token = 0

        # --- UI State ---
        self.state = tk.StringVar(value="idle")
        self.running = False
        self.device_idx = tk.StringVar()
        self.out_device_idx = tk.StringVar()
        self.duplex_mode = tk.StringVar(value="Half-duplex")

        # Echo engine
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

        # === NEW: Vision state initialization ===
        self._last_image_path = None
        self._last_vision_reply = ""
        self._last_was_vision = False
        self._vision_context_until = 0.0
        self._vision_turns_left = 0

        # === Initialize UI components FIRST ===
        self._setup_ui()

        # === Initialize AI engines AFTER UI ===
        self._setup_ai_engines()

        # Apply config defaults
        self._apply_config_defaults()

    def _setup_ui(self):
        """Initialize all UI components"""
        # Top controls
        top = ttk.Frame(self.master)
        top.grid(row=0, column=0, columnspan=12, sticky="we")

        self.light = tk.Canvas(top, width=48, height=48, highlightthickness=0)
        self.circle = self.light.create_oval(4, 4, 44, 44, fill="#f1c40f", outline="")
        self.light.grid(row=0, column=0, padx=10, pady=10)

        self.start_btn = ttk.Button(top, text="Start", command=self.start)
        self.stop_btn = ttk.Button(top, text="Stop", command=self.stop, state=tk.DISABLED)

        self.reset_btn = ttk.Button(top, text="Reset Chat", command=self.reset_chat)
        self.reset_btn.grid(row=0, column=12, padx=6, sticky="w")

        self.start_btn.grid(row=0, column=1, padx=6)
        self.stop_btn.grid(row=0, column=2, padx=6)

        # STOP SPEAKING button
        self.stop_speech_btn = ttk.Button(top, text="Stop Speaking", command=self.stop_speaking)
        self.stop_speech_btn.grid(row=0, column=3, padx=6)

        # Echo controls
        ttk.Checkbutton(
            top, text="Echo ON", variable=self.echo_enabled_var,
            command=lambda: setattr(self.echo_engine, "enabled", bool(self.echo_enabled_var.get()))
        ).grid(row=0, column=4, padx=(10, 4))
        ttk.Button(top, text="Show Echo", command=self._toggle_echo_window).grid(row=0, column=5, padx=(4, 10))

        # Images + Pass-to-Text buttons
        _imgbar = ttk.Frame(top)
        _imgbar.grid(row=0, column=6, padx=(6, 4))
        ttk.Button(_imgbar, text="Images…", command=self._toggle_image_window).pack(side="left")
        ttk.Button(_imgbar, text="Pass to Text AI", command=self.pass_vision_to_text).pack(side="left", padx=(6, 0))
        # Manual Refresh
        ttk.Button(_imgbar, text="Refresh Last Reply", command=self._refresh_last_reply).pack(side="left", padx=(6, 0))
        # LaTeX controls
        self.latex_auto = tk.BooleanVar(value=True)
        ttk.Checkbutton(top, text="Auto LaTeX preview", variable=self.latex_auto).grid(row=0, column=7, padx=6)
        ttk.Button(top, text="Show/Hide LaTeX", command=self.toggle_latex).grid(row=0, column=8, padx=6)
        ttk.Button(
            top, text="Copy Raw LaTeX",
            command=lambda: self.latex_win.copy_raw_latex() if hasattr(self, "latex_win") else None
        ).grid(row=0, column=9, padx=(0, 6))

        # Avatar
        self.avatar_win = None
        self.avatar_kind = tk.StringVar(value="Rings")
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

        def _on_avatar_kind_change(_e=None):
            if self.avatar_win and self.avatar_win.winfo_exists():
                try:
                    self.avatar_win.destroy()
                except Exception:
                    pass
                self.avatar_win = None
                self.open_avatar()

        self.avatar_combo.bind("<<ComboboxSelected>>", _on_avatar_kind_change)

        # Mode selection
        mode_bar = ttk.Frame(top)
        mode_bar.grid(row=0, column=10, padx=6, sticky="n")
        ttk.Label(mode_bar, text="Mode").pack(anchor="n")
        self.mode_combo = ttk.Combobox(
            mode_bar, textvariable=self.duplex_mode, state="readonly", width=18,
            values=["Half-duplex", "Full-duplex (barge-in)"]
        )
        self.mode_combo.current(0)
        self.mode_combo.pack(pady=(2, 0), anchor="n")

        # Ducking UI
        duck = ttk.Frame(top)
        duck.grid(row=1, column=0, columnspan=12, padx=10, sticky="we")
        ttk.Checkbutton(duck, text="Ducking", variable=self.ducking_enable).pack(side="left", padx=(0, 8))
        ttk.Label(duck, text="↓dB").pack(side="left")
        ttk.Spinbox(duck, from_=0, to=36, width=3, textvariable=self.duck_db).pack(side="left", padx=(2, 8))
        ttk.Label(duck, text="Thr").pack(side="left")
        ttk.Spinbox(duck, from_=200, to=5000, width=5, textvariable=self.duck_thresh).pack(side="left", padx=(2, 8))
        ttk.Label(duck, text="Atk/Rel ms").pack(side="left")
        ttk.Spinbox(duck, from_=5, to=300, width=4, textvariable=self.duck_attack).pack(side="left", padx=(2, 2))
        ttk.Spinbox(duck, from_=20, to=1000, width=5, textvariable=self.duck_release).pack(side="left", padx=(2, 8))
        ttk.Label(duck, text="Gain").pack(side="left", padx=(8, 2))
        self.duck_var = tk.DoubleVar(value=100.0)
        ttk.Progressbar(duck, orient="horizontal", length=120, mode="determinate",
                        variable=self.duck_var, maximum=100.0).pack(side="left", padx=(0, 8))
        self.rms_var = tk.StringVar(value="RMS: 0")
        ttk.Label(duck, textvariable=self.rms_var).pack(side="left")

        # Mic device combo
        ttk.Label(self.master, text="Mic device:").grid(row=2, column=0, sticky="e")
        self.dev_combo = ttk.Combobox(self.master, textvariable=self.device_idx, state="readonly", width=58)
        devs = list_input_devices()
        vals = [f"{i}: {n}" for i, n in devs] if devs else ["No input devices found"]
        self.dev_combo["values"] = vals
        self.dev_combo.current(0)
        self.dev_combo.grid(row=2, column=1, columnspan=9, sticky="we", padx=6, pady=6)

        # Output device combo
        ttk.Label(self.master, text="Speaker device:").grid(row=3, column=0, sticky="e")
        out_vals = self._list_output_devices()
        self.out_combo = ttk.Combobox(self.master, textvariable=self.out_device_idx, state="readonly", width=58,
                                      values=out_vals)
        if out_vals:
            self.out_combo.current(0)
        self.out_combo.grid(row=3, column=1, columnspan=9, sticky="we", padx=6, pady=6)

        # TTS selection
        self.tts_engine = tk.StringVar(value="edge")
        ttk.Label(self.master, text="TTS Engine:").grid(row=4, column=0, sticky="e")
        ttk.Radiobutton(self.master, text="Edge", variable=self.tts_engine, value="edge").grid(row=4, column=1,
                                                                                               sticky="w")
        ttk.Radiobutton(self.master, text="SAPI5", variable=self.tts_engine, value="sapi5").grid(row=4, column=2,
                                                                                                 sticky="w")

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
            import asyncio, edge_tts
            async def _list_voices():
                vm = await edge_tts.VoicesManager.create()
                names = sorted({v["Name"] for v in vm.voices if "Neural" in v.get("Name", "")})
                return names or edge_voices_fallback

            edge_voices = asyncio.run(_list_voices())
        except Exception:
            edge_voices = edge_voices_fallback

        ttk.Label(self.master, text="Edge Voice:").grid(row=5, column=0, sticky="e")
        self.edge_combo = ttk.Combobox(self.master, textvariable=self.edge_voice_var, values=edge_voices, width=60)
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
        ttk.Label(self.master, text="SAPI Voice:").grid(row=5, column=4, sticky="e")
        self.sapi_combo = ttk.Combobox(self.master, textvariable=self.sapi_voice_var, values=voices, width=60)
        self.sapi_combo.current(0)
        self.sapi_combo.grid(row=5, column=5, columnspan=5, sticky="we", padx=6, pady=4)

        # Text input
        ttk.Label(self.master, text="Text input:").grid(row=6, column=0, sticky="ne", padx=(6, 0), pady=(0, 6))
        self.text_box = ScrolledText(self.master, width=70, height=10, wrap="word")
        self.text_box.grid(row=6, column=1, columnspan=8, sticky="we", padx=6, pady=(0, 6))
        ttk.Button(self.master, text="Send", command=self.send_text).grid(row=6, column=9, sticky="nw", padx=6,
                                                                          pady=(0, 6))
        self.text_box.bind("<Control-Return>", lambda e: (self.send_text(), "break"))

        # Log
        ttk.Label(self.master, text="Log:").grid(row=7, column=0, sticky="nw", padx=6)
        self.log = tk.Text(self.master, height=12, width=80)
        self.log.grid(row=7, column=1, columnspan=9, sticky="nsew", padx=6)
        self.master.grid_rowconfigure(7, weight=1)
        self.master.grid_columnconfigure(9, weight=1)
        self.master.grid_columnconfigure(1, weight=1)
        self.master.grid_columnconfigure(5, weight=1)

        # LaTeX window
        DEFAULT_TEXT_PT = int(self.cfg.get("latex_text_pt", 12))
        DEFAULT_MATH_PT = int(self.cfg.get("latex_math_pt", 8))
        DEFAULT_TEXT_FAMILY = self.cfg.get("latex_text_family", "Segoe UI")

        self.latex_win = LatexWindow(
            self.master,
            log_fn=self.logln,
            text_family=DEFAULT_TEXT_FAMILY,
            text_size=DEFAULT_TEXT_PT,
            math_pt=DEFAULT_MATH_PT
        )

        if self.latex_auto.get():
            self.latex_win.show()

        # Capture the FIRST SAPI voice for vision
        self._sapi_default_voice_id = None
        try:
            import pyttsx3
            _eng = pyttsx3.init()
            _v = _eng.getProperty("voices")
            if _v:
                self._sapi_default_voice_id = _v[0].id
                self.logln(f"[tts] vision default SAPI voice: {_v[0].name} ({self._sapi_default_voice_id})")
            else:
                self.logln("[tts] no SAPI voices found; vision will fall back to selected SAPI voice")
        except Exception as e:
            self.logln(f"[tts] could not enumerate SAPI voices for vision default: {e}")

    def _setup_ai_engines(self):
        """Initialize AI engines after UI is set up"""
        self.asr = ASR(
            self.cfg["whisper_model"],
            self.cfg["whisper_device"],
            self.cfg["whisper_compute_type"],
            self.cfg["whisper_beam_size"]
        )

        import importlib, inspect, sys
        self.qwen = QwenLLM(
            model_path=self.cfg["qwen_model_path"],
            model=self.cfg["qwen_model_path"],
            temperature=self.cfg["qwen_temperature"],
            max_tokens=self.cfg["qwen_max_tokens"]
        )

        if hasattr(self.qwen, "model_path") and self.qwen.model_path:
            self.logln(f"[qwen] ✅ Using local GGUF model: {self.qwen.model_path}")
        elif hasattr(self.qwen, "model") and self.qwen.model:
            self.logln(f"[qwen] ✅ Using Ollama model: {self.qwen.model}")
        else:
            self.logln("[qwen] ⚠️ Could not detect model name.")

        # System prompt
        sys_prompt = (
                self.cfg.get("system_prompt")
                or self.cfg.get("qwen_system_prompt")
                or ""
        )
        self.qwen.system_prompt = sys_prompt
        self.logln(f"[qwen] system prompt (first 80): {sys_prompt[:80]!r}")

    def _apply_config_defaults(self):
        """Apply configuration defaults to UI"""
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
            self._bargein_enabled = self.duplex_mode.get().startswith("Full")

        except Exception as e:
            self.logln(f"[cfg] apply defaults error: {e}")

        # Barge-in control
        self._bargein_enabled = bool(self.cfg.get("bargein_enable", True))
        self._barge_latched = False
        self._barge_until = 0.0
        self._barge_cooldown_s = float(self.cfg.get("barge_cooldown_s", 0.7))
        self._barge_min_utt_chars = int(self.cfg.get("barge_min_utt_chars", 3))

        # State
        self.speaking_flag = False
        self.interrupt_flag = False
        self.barge_buffer = None
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

    # === NEW: Unified Vision State Manager ===
    def _update_vision_state(self, used_turn: bool = False, reset: bool = False, new_image: bool = False):
        """Thread-safe vision state management"""
        import time as _t

        if reset:
            self._vision_turns_left = 0
            self._vision_context_until = 0.0
            self._last_was_vision = False
            self.logln("[vision] state reset to text mode")
            return

        if new_image:
            # New image = fresh context
            self._vision_turns_left = int(self.cfg.get("vision_followup_max_turns", 2))
            self._vision_context_until = _t.monotonic() + float(self.cfg.get("vision_followup_s", 180))
            self._last_was_vision = True
            self.logln(f"[vision] new image context: turns={self._vision_turns_left}")
            return

        if used_turn and self._vision_turns_left > 0:
            self._vision_turns_left -= 1
            # Extend context window when using a turn
            self._vision_context_until = _t.monotonic() + float(self.cfg.get("vision_followup_s", 180))
            self.logln(f"[vision] used one turn: {self._vision_turns_left} remaining")

    # === FIXED: _on_new_image ===
    def _on_new_image(self, path: str):
        """Called by ImageWindow whenever the current image changes (open/snapshot/drag)."""
        try:
            if not path:
                return
            abs_path = os.path.abspath(path)
            if abs_path == self._last_image_path:
                return

            self._last_image_path = abs_path
            # FIXED: Use unified state management
            self._update_vision_state(new_image=True)
            self.logln(f"[vision] context image set -> {os.path.basename(abs_path)}")

        except Exception as e:
            self.logln(f"[vision] _on_new_image error: {e}")

    def _sync_image_context_from_window(self):
        """If the image window already has a file path, sync it into App._last_image_path."""
        try:
            if hasattr(self, "_img_win") and self._img_win and self._img_win.winfo_exists():
                path = getattr(self._img_win, "_img_path", None)
                if path and os.path.isfile(path):
                    abs_path = os.path.abspath(path)
                    if abs_path != self._last_image_path:
                        self._on_new_image(abs_path)
        except Exception:
            pass

    # === FIXED: handle_text_query ===
    def handle_text_query(self, text):
        self.logln(f"[user] {text}")
        self.preview_latex(text)
        self._sync_image_context_from_window()

        # Route camera/image commands first
        if self._route_command(text):
            return

        # --- Route the user question to appropriate model ---
        try:
            use_vision = self._should_use_vision_followup(text)

            if use_vision:
                self.logln("[vision] follow-up → reuse last image")
                reply = self._ollama_generate(text, images=[self._last_image_path])
                # FIXED: Only decrement AFTER successful generation
                self._update_vision_state(used_turn=True)
            else:
                # normal text model path
                reply = self.qwen.generate(text)
                self._update_vision_state(reset=True)  # Clear vision context when using text

        except Exception as e:
            self.logln(f"[llm/vision] {e}\n[hint] Is Ollama running?  ollama serve")
            self.set_light("idle")
            return

        self.logln(f"[qwen] {reply}")
        self.preview_latex(reply)
        self._set_last_vision_reply(reply)

        clean = clean_for_tts(reply)

        # unified playback fencing
        with self._play_lock:
            self._play_token += 1
            my_token = self._play_token
            self.interrupt_flag = False
            self.speaking_flag = True

        self.set_light("speaking")
        role = "vision" if use_vision else "text"
        self._latex_theme("vision" if role == "vision" else "default")

        try:
            if self.synthesize_to_wav(clean, self.cfg["out_wav"], role=role):
                self.master.after(0, self.latex_win._prepare_word_spans)
                play_path = self.cfg["out_wav"]
                if bool(self.echo_enabled_var.get()):
                    try:
                        play_path, _ = self.echo_engine.process_file(self.cfg["out_wav"], "out/last_reply_echo.wav")
                        self.logln("[echo] processed -> out/last_reply_echo.wav")
                    except Exception as e:
                        self.logln(f"[echo] processing failed: {e} (playing dry)")
                self.play_wav_with_interrupt(play_path, token=my_token)
        finally:
            self.speaking_flag = False
            self.interrupt_flag = False
            self.set_light("idle")
            self._latex_theme("default")

    # === FIXED: ask_vision ===
    def ask_vision(self, image_path: str, prompt: str):
        """Called by ImageWindow when the user presses 'Ask model'."""
        # remember the most recent image explicitly
        self._last_image_path = image_path
        self._update_vision_state(new_image=True)

        def _worker():
            try:
                self.logln(f"[vision] {os.path.basename(image_path)} | prompt: {prompt}")
                self.preview_latex(prompt)
                reply = self._ollama_generate(prompt, images=[image_path])

                # FIXED: Use unified state management
                self._update_vision_state(used_turn=True)

                self.logln(f"[qwen] {reply}")
                self.preview_latex(reply)

                # IMPORTANT: Set the vision reply BEFORE any playback starts
                self._set_last_vision_reply(reply)

                clean = clean_for_tts(reply)

                with self._play_lock:
                    self._play_token += 1
                    my_token = self._play_token
                    self.interrupt_flag = False
                    self.speaking_flag = True

                self.set_light("speaking")
                self._latex_theme("vision")

                try:
                    if self.synthesize_to_wav(clean, self.cfg["out_wav"], role="vision"):
                        self.master.after(0, self.latex_win._prepare_word_spans)
                        play_path = self.cfg["out_wav"]
                        if bool(self.echo_enabled_var.get()):
                            try:
                                play_path, _ = self.echo_engine.process_file(self.cfg["out_wav"],
                                                                             "out/last_reply_echo.wav")
                                self.logln("[echo] processed -> out/last_reply_echo.wav")
                            except Exception as e:
                                self.logln(f"[echo] processing failed: {e} (playing dry)")
                        self.play_wav_with_interrupt(play_path, token=my_token)
                finally:
                    self.set_light("idle")
                    self._latex_theme("default")

            except Exception as e:
                self.logln(f"[vision] error: {e}")
                self.set_light("idle")

        self.set_light("listening")
        threading.Thread(target=_worker, daemon=True).start()

    # === ENHANCED: pass_vision_to_text ===
    def pass_vision_to_text(self):
        """
        If a vision reply exists, move it into the Text input box so the user can edit and send.
        """
        try:
            # Add a small delay to ensure any ongoing vision processing completes
            import time
            time.sleep(0.1)  # Brief pause to let any concurrent operations finish

            # Use a thread-safe approach to get the latest vision reply
            txt = (getattr(self, '_last_vision_reply', "") or "").strip()

            if not txt:
                self.logln("[pass] nothing to pass from vision (no recent vision reply).")
                # Debug: show what's actually in the variable
                self.logln(f"[pass] debug: _last_vision_reply = {repr(getattr(self, '_last_vision_reply', 'NOT_SET'))}")
                return

            # FIXED: Use unified state reset
            self._update_vision_state(reset=True)

            # Move reply into text box with better formatting
            self.text_box.delete("1.0", "end")
            prefix = "# Vision Reply (editable - will use text model):\n\n"
            self.text_box.insert("1.0", prefix + txt + "\n\n# Your follow-up:\n")
            self.text_box.focus_set()
            self.text_box.see("end")

            # Set cursor at the end for easy editing
            self.text_box.mark_set("insert", "end-1c")

            self.logln(f"[pass] vision reply moved into Text input ({len(txt)} chars). Edit and click Send.")

        except Exception as e:
            self.logln(f"[pass] error: {e}")
            import traceback
            self.logln(f"[pass] traceback: {traceback.format_exc()}")

    def _set_last_vision_reply(self, reply: str):
        """Store the most recent vision reply and log a concise confirmation."""
        try:
            # Use thread-safe assignment
            self._last_vision_reply = (reply or "").strip()

            # short preview for visibility in your Log panel
            preview = self._last_vision_reply.strip().replace("\n", " ")
            if len(preview) > 120:
                preview = preview[:117] + "..."

            # Log immediately to confirm it's being set
            self.logln(f"[vision][cache] ✅ saved reply ({len(self._last_vision_reply)} chars): {preview}")

            # Also log the method that called this (for debugging)
            import inspect
            caller = inspect.stack()[1].function
            self.logln(f"[vision][cache] called from: {caller}")

        except Exception as e:
            self.logln(f"[vision][cache] ❌ failed to store reply: {e}")

    def _refresh_last_reply(self):
        """Debug method to show what's currently in the last vision reply."""
        try:
            txt = (getattr(self, '_last_vision_reply', "") or "").strip()
            if txt:
                self.logln(f"[refresh] Last vision reply exists: {len(txt)} chars")
                preview = txt[:100] + "..." if len(txt) > 100 else txt
                self.logln(f"[refresh] Preview: {preview}")
            else:
                self.logln("[refresh] No last vision reply found")

            # Also show vision state
            self.logln(
                f"[refresh] Vision state: turns_left={self._vision_turns_left}, last_was_vision={self._last_was_vision}")

        except Exception as e:
            self.logln(f"[refresh] Error: {e}")
    # [CONTINUED IN NEXT MESSAGE DUE TO LENGTH LIMIT...]
    # === Core Application Methods ===
    def start(self):
        if self.running: return
        self.running = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.set_light("idle")
        threading.Thread(target=self.loop, daemon=True).start()

    def stop(self):
        self.running = False
        self.stop_speaking()
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
        use_vad_thresh = self.cfg.get("vad_threshold_full", 0.005) if self.duplex_mode.get().startswith(
            "Full") else self.cfg.get("vad_threshold", 0.01)

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
                    self._bargein_enabled = cur_mode.startswith("Full")

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
            if not self.running:
                break
            if self.speaking_flag and self.duplex_mode.get().startswith("Half"):
                continue

            text = self.asr.transcribe(utt, self.cfg["sample_rate"])
            self._sync_image_context_from_window()

            if not text:
                continue
            self.logln(f"[asr] {text}")

            # Route camera/image commands first (spoken commands)
            if self._route_command(text):
                continue
            self._sync_image_context_from_window()
            if getattr(self, "_barge_latched", False):
                if time.monotonic() < getattr(self, "_barge_until", 0.0):
                    if len(text.strip()) < int(self.cfg.get("barge_min_utt_chars", 3)):
                        self.logln("[barge-in] suppressing tiny fragment")
                        continue
                    self.logln("[barge-in] listen-only window: suppressing LLM/TTS")
                    continue
                else:
                    self._barge_latched = False

            # Handle vision/text routing
            try:
                t = (text or "").lower()
                force_visual = any(k in t for k in (
                    "color", "colour", "how many", "count",
                    "in the image", "in the picture", "in the photo",
                    "what is in", "what’s in", "whats in",
                    "on the left", "on the right", "in the middle"
                ))

                use_vision = (
                        bool(self._last_image_path) and
                        (self._should_use_vision_followup(text) or force_visual)
                )


                if use_vision:
                    self.logln("[vision][voice] follow-up → reuse last image")
                    reply = self._ollama_generate(text, images=[self._last_image_path])

                    # IMPORTANT: Set the vision reply immediately after generation
                    self._set_last_vision_reply(reply)

                    self._update_vision_state(used_turn=True)
                else:
                    self.logln("[vision][voice] not using vision (text-only)")
                    reply = self.qwen.generate(text)
                    self._update_vision_state(reset=True)

            except Exception as e:
                self.logln(f"[llm/vision] {e}\n[hint] Is Ollama running?  ollama serve")
                self.set_light("idle")
                continue

            self.logln(f"[qwen] {reply}")
            self.preview_latex(reply)

            clean = clean_for_tts(reply)
            self.speaking_flag = True
            self.interrupt_flag = False
            self.set_light("speaking")

            role = "vision" if use_vision else "text"
            self._latex_theme("vision" if role == "vision" else "default")

            try:
                if self.synthesize_to_wav(clean, self.cfg["out_wav"], role=role):
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
                self._latex_theme("default")
        self.stop()

    # === Voice/Audio Methods ===
    def start_bargein_mic(self, device_idx):
        q = deque(maxlen=64)

        def callback(indata, frames, time_info, status):
            if self.speaking_flag and self._bargein_enabled:
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
                    _time.sleep(dt)
                    continue
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

    def _update_duck_ui(self):
        try:
            g = float(getattr(self, '_duck_gain', 1.0))
            self.duck_var.set(100.0 * g)
            self.rms_var.set(f"RMS: {int(getattr(self, '_last_rms', 0))}")
        except Exception:
            pass

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
            s[:fade.size] *= fade
            s[-fade.size:] *= fade[::-1]
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

    def play_wav_with_interrupt(self, path, token=None):
        import platform as _plat
        start_time = time.monotonic()
        active_token = token

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
            blocksize = self.cfg.get("out_blocksize", 8192)
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
                        self._ui_last_ratio = 0.0 if self._ui_last_ratio < 0 else (
                            1.0 if self._ui_last_ratio > 1 else self._ui_last_ratio)
                        alpha = 0.25
                        self._ui_eased_ratio += alpha * (r_target - self._ui_eased_ratio)
                        r_show = 0.0 if self._ui_eased_ratio < 0 else (
                            1.0 if self._ui_eased_ratio > 1 else self._ui_eased_ratio)
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
                    if (active_token is not None) and (active_token != self._play_token):
                        outdata[:] = 0
                        raise sd.CallbackStop()

                    if self.interrupt_flag or not self.running:
                        outdata[:] = 0
                        raise sd.CallbackStop()

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
                    AVATAR_LEVELS = 32
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
            self._beep_once_guard = False
            dur = time.monotonic() - start_time
            self.logln(f"[audio] playback done ({dur:.2f}s)")

    # === Vision System Methods ===
    def _should_use_vision_followup(self, text: str) -> bool:
        """Decide if a text turn should reuse the last image."""
        import time as _t

        t = (text or "").lower()
        now = _t.monotonic()
        has_img = bool(self._last_image_path)
        window_ok = has_img and (now <= float(self._vision_context_until or 0))
        turns_ok = has_img and (int(self._vision_turns_left or 0) > 0)

        # Hard cues that strongly suggest you really mean the image
        cues = [
            "image", "picture", "photo", "in the image", "in the picture", "in the photo",
            "what color", "what colour", "how many", "count", "what type", "what kind",
            "on the left", "on the right", "in the middle", "do you see", "can you see",
            "people", "person", "faces",
        ]
        cue_hit = any(c in t for c in cues)

        # Light heuristic: short/pronounny follow-ups after a vision turn
        shortish = len(t) <= 60
        pronouny = any(
            p in f" {t} " for p in (" what ", " which ", " that ", " this ", " it ", " they ", " them ", " those "))
        light_hint = (shortish or pronouny) and bool(self._last_was_vision)

        # If user explicitly says to stop using the image, don't route to vision
        new_topic_cues = ["new topic", "switch to text", "ignore the image", "clear image", "text mode"]
        hard_text = any(k in t for k in new_topic_cues)

        decision = bool(window_ok and turns_ok and (cue_hit or light_hint) and not hard_text)
        self.logln(
            f"[vision-route] followup? has_img={has_img} window_ok={window_ok} turns={self._vision_turns_left} "
            f"cues={cue_hit} light={light_hint} hard_text={hard_text} -> {decision}"
        )
        return decision

    def _route_command(self, raw_text: str) -> bool:
        """
        Handle voice/typed control phrases robustly.
        Normalizes contractions & punctuation and matches many variants.
        Returns True if a command was executed.
        """
        import time as _t

        text = (raw_text or "").strip().lower()

        # Light normalization
        norm_map = {
            "what's": "what is",
            "whats": "what is",
            "i'm": "i am",
            "you're": "you are",
            "it's": "it is",
            "that's": "that is",
        }
        for k, v in norm_map.items():
            text = text.replace(k, v)

        # remove punctuation except spaces/alphanumerics
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = re.sub(r"\s{2,}", " ", text).strip()

        def matched(patterns):
            return any(p in text for p in patterns)

        # Command groups
        exit_vision = [
            "new topic", "switch to text", "text mode", "forget image", "clear image", "ignore the image",
            "stop using the image", "no vision", "text only", "go back to text", "back to text"
        ]

        start_cam = [
            "start camera", "open camera", "turn on camera", "camera on",
            "start the camera", "enable camera"
        ]
        stop_cam = [
            "stop camera", "close camera", "turn off camera", "camera off",
            "stop the camera", "disable camera"
        ]
        take_pic = [
            "take a picture", "take picture", "take a photo", "take photo",
            "snapshot", "capture photo", "capture image", "snap a picture"
        ]

        # describe/explain the latest image
        explain_roots = [
            "explain what is in", "describe what is in", "what is in",
            "explain what is about", "describe what is about", "what is about",
            "explain", "describe",
        ]
        media_words = ["image", "picture", "photo", "the image", "the picture", "the photo"]
        explain_all = [f"{root} {m}" for root in explain_roots for m in media_words]
        explain_all += [
            "describe the image", "describe the picture", "describe the photo",
            "explain the image", "explain the picture", "explain the photo",
            "describe image", "describe picture", "describe photo",
            "explain image", "explain picture", "explain photo",
            "what is in the image", "what is in the picture", "what is in the photo",
            "what is in image", "what is in picture", "what is in photo",
        ]

        # Vision takeover
        vision_takeover = [
            "vision mode", "image mode",
            "speak with the image ai", "speak to the image ai", "talk to the image ai",
            "speak with the vision ai", "speak to the vision ai", "talk to the vision ai",
            "use the image ai", "use the vision ai",
            "switch to vision", "switch to image","speak with the Guardian"
        ]

        # Dispatch
        if matched(start_cam):
            self.start_camera_ui()
            self.set_light("idle")
            return True

        if matched(stop_cam):
            self.stop_camera_ui()
            self.set_light("idle")
            return True

        if matched(take_pic):
            self.take_picture_ui()
            self.set_light("idle")
            return True

        if matched(explain_all):
            self.explain_last_image_ui()
            self.set_light("idle")
            return True

        if matched(vision_takeover):
            self._ensure_image_window()
            try:
                if self._img_win and self._img_win.winfo_exists():
                    self._img_win.deiconify()
                    self._img_win.lift()
                    has_image = bool(getattr(self._img_win, "_img_path", None))
                    if not has_image:
                        self._img_win.start_camera()
            except Exception:
                pass

            self._last_was_vision = True
            self._vision_turns_left = int(self.cfg.get("vision_followup_max_turns", 2))
            self._vision_context_until = _t.monotonic() + float(self.cfg.get("vision_followup_s", 180))

            try:
                self.latex_win.set_scheme("vision")
            except Exception:
                pass

            self.logln(f"[vision] takeover requested — turns={self._vision_turns_left}")
            self.set_light("idle")
            return True

        if matched(exit_vision):
            self._last_was_vision = False
            self._vision_context_until = 0.0
            self._vision_turns_left = 0
            try:
                self.latex_win.set_scheme("default")
            except Exception:
                pass
            self.logln("[vision] image context cleared; back to text mode")
            return True

        return False

    def _ollama_generate(self, prompt: str, images=None):
        """
        Use Ollama REST directly when images are provided.
        Falls back to self.qwen.generate for text-only.
        """
        if not images:
            return self.qwen.generate(prompt)

        b64_images = []
        for it in images:
            if isinstance(it, str) and os.path.isfile(it):
                with open(it, "rb") as f:
                    b64_images.append(base64.b64encode(f.read()).decode("ascii"))
            else:
                b64_images.append(it)

        vision_prefix = (
            "You can see the attached image. Answer directly and concisely.\n"
            "- If the question is 'how many ...', output a number.\n"
            "- If the question is 'what type ...', name the type/class.\n"
            "- Do not claim you are text-based.\n\n"
        )
        full_prompt = vision_prefix + (prompt or "")

        payload = {
            "model": self.vl_model,
            "prompt": full_prompt,
            "images": b64_images,
            "system": self.vl_system_prompt,
            "stream": False
        }

        self.logln(f"[vision] model={payload['model']} images={len(b64_images)}")
        self.logln(f"[vision] sys[:120]={payload['system'][:120]!r}")
        self.logln(f"[vision] prm[:120]={payload['prompt'][:120]!r}")

        r = requests.post("http://127.0.0.1:11434/api/generate", json=payload, timeout=180)
        try:
            r.raise_for_status()
        except Exception:
            raise RuntimeError(f"Ollama error {r.status_code}: {r.text[:500]}")

        return (r.json().get("response") or "").strip()

    # === UI Helper Methods ===
    def stop_speaking(self):
        try:
            self.interrupt_flag = True
            self.speaking_flag = False
            try:
                sd.stop()
            except Exception:
                pass
            self._hi_stop = True
            self._tts_silent = False
            self._ui_last_ratio = 0.0
            self._ui_eased_ratio = 0.0
            try:
                self.latex_win.clear_highlight()
            except Exception:
                pass
            try:
                if self.avatar_win and self.avatar_win.winfo_exists():
                    self.avatar_win.set_level(0)
            except Exception:
                pass
        finally:
            self.set_light("idle")

    def set_light(self, mode):
        color = {"idle": "#f1c40f", "listening": "#2ecc71", "speaking": "#e74c3c"}.get(mode, "#f1c40f")
        self.light.itemconfig(self.circle, fill=color)
        self.state.set(mode)

    def _toggle_echo_window(self):
        try:
            if self._echo_win is None or not self._echo_win.winfo_exists():
                self._echo_win = EchoWindow(self.master, self.echo_engine)
            if self._echo_win.state() == "withdrawn":
                self._echo_win.deiconify()
                self._echo_win.lift()
            else:
                self._echo_win.withdraw()
        except Exception as e:
            self.logln(f"[echo] window error: {e}")

    def _toggle_image_window(self):
        try:
            if not hasattr(self, "_img_win") or self._img_win is None or not self._img_win.winfo_exists():
                # Use the proper ImageWindow class
                self._img_win = self.ImageWindow(
                    self.master,
                    on_send=self.ask_vision,
                    on_image_change=self._on_new_image
                )

            if self._img_win.state() == "withdrawn":
                self._img_win.deiconify()
                self._img_win.lift()
            else:
                self._img_win.withdraw()
        except Exception as e:
            self.logln(f"[image] window error: {e}")

    # ImageWindow class definition
    class ImageWindow(tk.Toplevel):
        """
        Vision helper:
          - Open image (file dialog)
          - Drag & drop (if tkinterdnd2 is installed)
          - Camera preview + snapshot (if opencv is installed)
          - Send to model (calls parent App.ask_vision)
        """

        def __init__(self, master, on_send, on_image_change=None):
            super().__init__(master)
            self.title("Image / Camera")
            self.geometry("720x560")
            self.protocol("WM_DELETE_WINDOW", self.withdraw)

            self._on_send = on_send  # callback: on_send(image_path, prompt)
            self._on_image_change = on_image_change  # notify app when image changes
            self._img_path = None
            self._img_tk = None
            self._cam = None
            self._cam_timer = None
            self._live_mode = False

            # UI
            wrap = ttk.Frame(self)
            wrap.pack(fill="both", expand=True, padx=8, pady=8)
            top = ttk.Frame(wrap)
            top.pack(fill="x")

            ttk.Button(top, text="Open Image…", command=self.open_image).pack(side="left", padx=(0, 6))
            ttk.Button(top, text="Start Camera", command=self.start_camera).pack(side="left", padx=(0, 6))
            ttk.Button(top, text="Stop Camera", command=self.stop_camera).pack(side="left", padx=(0, 6))
            ttk.Button(top, text="Snapshot", command=self.snapshot).pack(side="left", padx=(0, 6))

            ttk.Label(top, text="Prompt:").pack(side="left", padx=(16, 4))
            self.prompt_var = tk.StringVar(value="Please solve/describe any equations in this image. Use LaTeX.")
            self.prompt_entry = ttk.Entry(top, textvariable=self.prompt_var, width=48)
            self.prompt_entry.pack(side="left", fill="x", expand=True)

            ttk.Button(top, text="Ask model", command=self.send_now).pack(side="left", padx=(6, 0))

            self.canvas = tk.Canvas(wrap, bg="#111", highlightthickness=0)
            self.canvas.pack(fill="both", expand=True, pady=(8, 0))
            self.canvas.bind("<Configure>", lambda e: self._redraw())

            # Drag & drop (optional)
            if DND_FILES:
                try:
                    self.drop_target_register(DND_FILES)
                    self.dnd_bind("<<Drop>>", self._on_drop)
                except Exception:
                    pass

        # ---- File ops ----
        def open_image(self):
            from tkinter import filedialog
            path = filedialog.askopenfilename(
                title="Open image",
                filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.webp;*.tif;*.tiff")]
            )
            if path:
                self.set_image(path)

        def _on_drop(self, event):
            # Windows sends a quoted path; handle multiple too
            paths = self._parse_drop_paths(event.data)
            if paths:
                self.set_image(paths[0])

        @staticmethod
        def _parse_drop_paths(data):
            # minimal parser for common DND formats
            items = []
            cur = ""
            in_quote = False
            for ch in data:
                if ch == '"':
                    in_quote = not in_quote
                elif ch in (" ", "\n") and not in_quote:
                    if cur.strip():
                        items.append(cur.strip('"'))
                    cur = ""
                else:
                    cur += ch
            if cur.strip():
                items.append(cur.strip('"'))
            return items

        def set_image(self, path):
            try:
                img = Image.open(path).convert("RGB")
            except Exception as e:
                messagebox.showerror("Open image", f"Could not open:\n{e}")
                return
            # Stop camera to avoid races overwriting the chosen file image
            self.stop_camera()
            self._img_path = os.path.abspath(path)
            self._img_pil = img
            self._redraw()
            # tell the app we have a new image file path
            if callable(self._on_image_change):
                try:
                    self._on_image_change(self._img_path)
                except Exception:
                    pass

        def _redraw(self):
            if not hasattr(self, "_img_pil"):
                self.canvas.delete("all")
                return
            cw, ch = max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())
            img = self._img_pil
            # fit
            scale = min((cw - 2) / img.width, (ch - 2) / img.height, 1.0)
            disp = img if scale >= 0.999 else img.resize((int(img.width * scale), int(img.height * scale)),
                                                         Image.LANCZOS)
            self._img_tk = ImageTk.PhotoImage(disp)
            self.canvas.delete("all")
            self.canvas.create_image(cw // 2, ch // 2, image=self._img_tk)

        # ---- Camera ----
        def start_camera(self):
            if cv2 is None:
                print("[camera] OpenCV not installed. pip install opencv-python")
                return

            try:
                self._cam = cv2.VideoCapture(0, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(0)
                if not self._cam or not self._cam.isOpened():
                    raise RuntimeError("No camera found")
                # entering live mode: new frames will update _img_pil; don't keep any stale _img_path
                self._live_mode = True
                self._img_path = None
                self._update_cam()
            except Exception as e:
                print(f"[camera] {e}")
                self._cam = None
                self._live_mode = False

        def _update_cam(self):
            # Keep pushing frames to _img_pil; DO NOT touch _img_path here (prevents race)
            if self._cam is None or not self._cam.isOpened():
                return
            ok, frame = self._cam.read()
            if ok:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self._img_pil = Image.fromarray(rgb)
                self._redraw()
            self._cam_timer = self.after(33, self._update_cam)

        def stop_camera(self):
            self._live_mode = False
            if self._cam_timer:
                try:
                    self.after_cancel(self._cam_timer)
                except Exception:
                    pass
                self._cam_timer = None
            if self._cam is not None:
                try:
                    self._cam.release()
                except Exception:
                    pass
                self._cam = None

        def snapshot(self):
            """
            Save current image/camera frame to ./out/snapshot_*.png.
            Returns the saved absolute path (string) on success, or None on failure.
            """
            if not hasattr(self, "_img_pil"):
                messagebox.showinfo("Snapshot", "No image/camera frame to save.")
                return None
            os.makedirs("out", exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            path = os.path.abspath(os.path.join("out", f"snapshot_{ts}.png"))
            try:
                self._img_pil.save(path)
                self._img_path = path
                # notify app that current image changed to this snapshot
                if callable(self._on_image_change):
                    try:
                        self._on_image_change(self._img_path)
                    except Exception:
                        pass
                return path
            except Exception as e:
                print(f"[snapshot] {e}")
                return None

        # ---- Send to model ----
        def send_now(self):
            if not hasattr(self, "_img_pil"):
                print("[vision] No image/camera frame yet.")
                return

            # Ensure a file path; if the image is transient (camera frame), write a temp PNG in ./out
            path = self._img_path
            if path is None:
                os.makedirs("out", exist_ok=True)
                path = os.path.abspath(os.path.join("out", "live_frame.png"))
                try:
                    self._img_pil.save(path)
                    self._img_path = path
                    # also tell the app that the current image became this file
                    if callable(self._on_image_change):
                        try:
                            self._on_image_change(self._img_path)
                        except Exception:
                            pass
                except Exception as e:
                    print(f"[vision] could not save live frame: {e}")
                    return

            prompt = self.prompt_var.get().strip() or "Describe and solve any equations in this image. Use LaTeX."
            self._on_send(path, prompt)

        def destroy(self):
            self.stop_camera()
            super().destroy()

    def toggle_latex(self):
        try:
            if self.latex_win.state() == "withdrawn":
                self.latex_win.show()
            else:
                self.latex_win.hide()
        except Exception:
            self.latex_win.show()

    def open_avatar(self):
        try:
            kind = self.avatar_kind.get()
            if kind == "Rings":
                self.avatar_win = CircleAvatarWindow(self.master)
            elif kind == "Rectangles":
                self.avatar_win = RectAvatarWindow(self.master)  # NEW NAME
            elif kind == "Rectangles 2":
                self.avatar_win = RectAvatarWindow2(self.master)  # NEW NAME
            else:
                # Fallback to Rings
                self.avatar_win = CircleAvatarWindow(self.master)
            self.avatar_win.show()
        except Exception as e:
            self.logln(f"[avatar] open error: {e}")
            import traceback
            self.logln(f"[avatar] traceback: {traceback.format_exc()}")

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
            # Try to create a new one if there's an issue
            try:
                self.avatar_win = None
                self.open_avatar()
            except Exception as e2:
                self.logln(f"[avatar] recovery failed: {e2}")

    def _avatar_set_level_async(self, lvl: int):
        try:
            if self.avatar_win and self.avatar_win.winfo_exists():
                self.avatar_win.set_level(lvl)
        except Exception:
            pass

    def send_text(self):
        if hasattr(self, "text_box"):
            text = self.text_box.get("1.0", "end-1c").strip()
            self.text_box.delete("1.0", "end")
        else:
            text = self.text_entry.get().strip()
            self.text_entry.delete(0, "end")

        if not text:
            return

        threading.Thread(target=self.handle_text_query, args=(text,), daemon=True).start()

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

    def _latex_theme(self, mode: str):
        try:
            if hasattr(self, "latex_win"):
                self.latex_win.set_scheme("vision" if mode == "vision" else "default")
        except Exception:
            pass

    def synthesize_to_wav(self, text, out_wav, role="text"):
        engine = self.tts_engine.get()
        try:
            # Vision always uses the FIRST SAPI5 voice
            if role == "vision":
                import pyttsx3
                voice_id = self._sapi_default_voice_id
                if not voice_id:
                    voice_id = self.sapi_voice_var.get().split(" | ")[0]
                tmp = out_wav + ".tmp.wav"
                eng = pyttsx3.init()
                eng.setProperty("voice", voice_id)
                eng.save_to_file(text, tmp)
                eng.runAndWait()
                os.replace(tmp, out_wav)
                self.logln(f"[tts] vision (sapi5, fixed): {voice_id}")
                return True

            # Text keeps current selection
            if engine == "sapi5":
                import pyttsx3
                voice_id = self.sapi_voice_var.get().split(" | ")[0]
                tmp = out_wav + ".tmp.wav"
                eng = pyttsx3.init()
                eng.setProperty("voice", voice_id)
                eng.save_to_file(text, tmp)
                eng.runAndWait()
                os.replace(tmp, out_wav)
                self.logln(f"[tts] text (sapi5): {voice_id}")
                return True
            else:
                from tts_edge import say as edge_say
                v = self.edge_voice_var.get()
                edge_say(text, voice=v, rate=self.cfg.get("rate", "-10%"), out_wav=out_wav)
                self.logln(f"[tts] text (edge): {v}")
                return True
        except Exception as e:
            self.logln(f"[tts] error: {e}")
            try:
                messagebox.showerror("TTS error", str(e))
            except:
                pass
            return False

    # === Device Methods ===
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

    # === Vision UI Helpers ===
    def _ensure_image_window(self):
        """Create the ImageWindow if needed (but don't show it)."""
        try:
            if not hasattr(self, "_img_win") or self._img_win is None or not self._img_win.winfo_exists():
                self._img_win = self.ImageWindow(
                    self.master,
                    on_send=self.ask_vision,
                    on_image_change=self._on_new_image
                )
        except Exception as e:
            self.logln(f"[vision] could not create image window: {e}")
            self._img_win = None

    def start_camera_ui(self):
        """Voice/typed: 'start camera' -> open window and start streaming."""
        try:
            self._ensure_image_window()
            if self._img_win is None:
                self.logln("[vision] camera UI unavailable")
                return
            self._img_win.deiconify()
            self._img_win.lift()
            self._img_win.start_camera()
            self.logln("[vision] camera started")
        except Exception as e:
            self.logln(f"[vision] start camera error: {e}")

    def stop_camera_ui(self):
        """Voice/typed: 'stop camera' -> stop streaming."""
        try:
            if hasattr(self, "_img_win") and self._img_win and self._img_win.winfo_exists():
                self._img_win.stop_camera()
                self.logln("[vision] camera stopped")
            else:
                self.logln("[vision] camera window not open")
        except Exception as e:
            self.logln(f"[vision] stop camera error: {e}")

    def take_picture_ui(self):
        """Voice/typed: 'take a picture' -> snapshot current frame."""
        try:
            self._ensure_image_window()
            if self._img_win is None:
                self.logln("[vision] camera UI unavailable")
                return
            saved = self._img_win.snapshot()
            if saved:
                self.logln(f"[vision] snapshot ready: {saved}")
            else:
                self.logln("[vision] snapshot failed")
        except Exception as e:
            self.logln(f"[vision] take picture error: {e}")

    def explain_last_image_ui(self, prompt_text=None):
        try:
            img_path = None
            if hasattr(self, "_img_win") and self._img_win and self._img_win.winfo_exists():
                img_path = getattr(self._img_win, "_img_path", None)
            if not img_path:
                img_path = self._last_image_path

            if not img_path:
                self.logln("[vision] no image available. Say 'take a picture' or open an image first.")
                return

            prompt = (prompt_text or "Describe what is in the image in clear detail.").strip()
            self.ask_vision(img_path, prompt)
        except Exception as e:
            self.logln(f"[vision] explain image error: {e}")

    def _ollama_available_models(self):
        try:
            r = requests.get("http://127.0.0.1:11434/api/tags", timeout=5)
            r.raise_for_status()
            tags = r.json().get("models", [])
            return {m.get("name", "") for m in tags}
        except Exception as e:
            self.logln(f"[ollama] tag query failed: {e}")
            return set()

    def _ensure_vl_model_installed(self):
        have = self._ollama_available_models()
        want = str(self.vl_model or "").strip()
        if not want:
            self.vl_model = "qwen2.5vl:7b"
            want = self.vl_model
        if want in have:
            self.logln(f"[vision] ✅ using VL model: {want}")
            return
        candidates = ["qwen2.5vl:7b", "qwen2.5vl:latest", "qwen2.5vl-latex:latest"]
        for c in candidates:
            if c in have:
                self.logln(f"[vision] '{want}' not found; switching to '{c}'")
                self.vl_model = c
                return
        self.logln(f"[vision] ⚠️ '{want}' not installed. Run:  ollama pull {want}")
#End of App
# Helper functions for EchoEngine
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
