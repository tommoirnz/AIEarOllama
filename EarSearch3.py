# === Part 1: Imports, Config, Cleaners, LaTeX Window =========================
# === SUPPRESS HARMLESS WARNINGS ===
# Add to your existing warning suppression
import warnings

# 24/10 2025 All working but a few duplicated procedures
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*pynvml.*")
warnings.filterwarnings("ignore", message=".*pkg_resources.*")
warnings.filterwarnings("ignore", message=".*CUDA capability.*")
warnings.filterwarnings("ignore", message=".*overflow encountered.*")

# Also set numpy to ignore specific warnings
import numpy as np

np.seterr(over='ignore', under='ignore', invalid='ignore')
from dotenv import load_dotenv

load_dotenv()

import os, json, threading, time, re, math
from collections import deque
import soundfile as sf
import sounddevice as sd
import tkinter as tk
from tkinter import ttk, messagebox
from io import BytesIO
from PIL import Image, ImageTk
import matplotlib
from status_light_window import StatusLightWindow  # Adjust the filename if needed


# This uses two models and a different Json file and handles images as well
# Look for the Json file  called Json2. You will need two models as per the Json file loaded into ollama
# can read equations off paper and handwriting. Uses qwen_llmSearch2.py
# Can use 'Sleep' and 'Awaken' to mute the microphone but text can still be used. There is also a switch to stop speech.
# So you can run entirely in text mode if you so choose.
# To search the internet you will need a key from Brave Search API https://brave.com/search/api/
# It's free for a fair number of searches but after that you need to pay
# Put that key in the file .env or the program won't work
# Try asking it "what is the latest news in New Zealand and it will find that enws and summarise it.
# Warning, Ai models make mistakes so check and answers
# Tom Moir Oct 2025
matplotlib.rcParams["figure.max_open_warning"] = 0
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tkinter.font as tkfont
from tkinter.scrolledtext import ScrolledText
import base64, tempfile, requests
import queue
import httpx
import trafilatura
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from urllib.parse import urljoin
import re
import winsound
from dataclasses import dataclass, field
from typing import List, Optional

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
from qwen_llmSearch2 import QwenLLM
from pydub import AudioSegment

...

# Import the math speech converter
from Speak_Maths import MathSpeechConverter

# Create a global instance
math_speech_converter = MathSpeechConverter()


# ===  ITEM DATACLASS ===
@dataclass
class Item:
    title: str
    url: str
    snippet: str = ""
    pubdate: Optional[str] = None
    summary: Optional[str] = None
    image_urls: List[str] = field(default_factory=list)


# === END DATACLASS ===


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
def clean_for_tts(text: str, speak_math: bool = True) -> str:
    """
    Enhanced TTS cleaner that converts LaTeX math to spoken English.

    Args:
        text: Input text containing LaTeX math
        speak_math: If True, convert math to speech; if False, say "equation"

    Returns:
        Text ready for TTS with math properly spoken
    """
    if not text:
        return ""

    # Use the math speech converter to handle LaTeX math
    cleaned_text = math_speech_converter.make_speakable_text(text, speak_math=speak_math)

    # Additional light cleanup for TTS
    cleaned_text = re.sub(r"[#*_`~>\[\]\(\)-]", "", cleaned_text)
    cleaned_text = re.sub(r":[a-z_]+:", "", cleaned_text)
    cleaned_text = re.sub(r"^[QAqa]:\s*", "", cleaned_text)
    cleaned_text = re.sub(r"\s{2,}", " ", cleaned_text)

    return cleaned_text.strip()


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
        self.geometry("800x600")
        self._log = log_fn or (lambda msg: None)

        # === PERMANENT DARK MODE COLORS ===
        self.dark_bg = "#000000"  # Black background
        self.dark_fg = "#ffffff"  # White text
        self.dark_highlight = "#4a86e8"  # Blue highlight

        # === SET WINDOW BACKGROUND (FRAME) ===
        self.configure(bg=self.dark_bg)  # Start with black frame

        # Initialize defaults
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

        # --- Top bar - SIMPLIFIED (NO DARK MODE TOGGLE) ---
        topbar = ttk.Frame(container)
        topbar.pack(fill="x", padx=6, pady=(6, 2))

        # === TOP BAR CONTROLS ===
        ttk.Checkbutton(
            topbar, text="Show raw LaTeX",
            variable=self.show_raw,
            command=lambda: self.show_document(self._last_text or "")
        ).pack(side="left")

        ttk.Button(topbar, text="Copy Raw LaTeX", command=self.copy_raw_latex).pack(side="left", padx=(8, 6))
        ttk.Button(topbar, text="LaTeX Diagnostics", command=self._run_latex_diagnostics).pack(side="left")

        # === TEXT SIZE CONTROLS ===
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

        # === WINDOW SIZE CONTROLS ===
        ttk.Button(topbar, text="📐 Size+", command=self._increase_size).pack(side="right", padx=(6, 2))
        ttk.Button(topbar, text="📐 Size-", command=self._decrease_size).pack(side="right", padx=(2, 6))

        txt_spin.bind("<Return>", lambda _e: self.set_text_font(size=self.text_pt_var.get()))
        math_spin.bind("<Return>", lambda _e: self.set_math_pt(self.math_pt_var.get()))

        # --- Text + Scrollbar with PERMANENT DARK THEME ---
        textwrap = ttk.Frame(container)
        textwrap.pack(fill="both", expand=True)

        # === MAIN TEXT WIDGET - ALWAYS DARK ===
        self.textview = tk.Text(
            textwrap,
            bg=self.dark_bg,  # Black background
            fg=self.dark_fg,  # White text
            wrap="word",
            undo=False,
            insertbackground=self.dark_fg,  # White cursor
            selectbackground=self.dark_highlight,  # Blue selection
            inactiveselectbackground=self.dark_highlight  # Blue selection when not focused
        )

        vbar = ttk.Scrollbar(textwrap, orient="vertical", command=self.textview.yview)
        self.textview.configure(yscrollcommand=vbar.set)
        vbar.pack(side="right", fill="y")
        self.textview.pack(side="left", fill="both", expand=True)

        self.textview.configure(font=self._text_font, state="normal")

        # === TEXT BINDINGS ===
        self.textview.bind("<Key>", self._block_keys)
        self.textview.bind("<<Paste>>", lambda e: "break")
        self.textview.bind("<Control-v>", lambda e: "break")
        self.textview.bind("<Control-x>", lambda e: "break")
        self.textview.bind("<Control-c>", lambda e: None)
        self.textview.bind("<Control-a>", self._select_all)

        # --- Context menu with DARK THEME ---
        self._menu = tk.Menu(self, tearoff=0, bg=self.dark_bg, fg=self.dark_fg)
        self._menu.add_command(label="Copy", command=lambda: self.textview.event_generate("<<Copy>>"))
        self._menu.add_command(label="Select All", command=lambda: self._select_all(None))
        self._menu.add_separator()
        self._menu.add_command(label="Copy Raw LaTeX", command=self.copy_raw_latex)
        # NO DARK MODE TOGGLE IN MENU
        self.textview.bind("<Button-3>", self._popup_menu)
        self.textview.bind("<Button-2>", self._popup_menu)

        # --- Highlight tags for DARK MODE ---
        self.textview.tag_configure("speak", background=self.dark_highlight, foreground=self.dark_bg)
        self.textview.tag_configure("normal", background="")
        self.textview.tag_configure("tight", spacing1=1, spacing3=1)

        self.withdraw()

    # ==================== ESSENTIAL WINDOW METHODS ====================
    def copy_raw_latex(self):
        """Copy raw LaTeX source to clipboard with proper document structure"""
        try:
            content = self._last_text or ""

            if content.strip():
                # Create a complete LaTeX document
                latex_document = f"""\\documentclass{{article}}
    \\usepackage{{amsmath}}
    \\usepackage{{amssymb}}
    \\usepackage{{graphicx}}
    \\usepackage{{hyperref}}
    \\begin{{document}}

    {content}

    \\end{{document}}"""

                self.clipboard_clear()
                self.clipboard_append(latex_document)
                self._log("[latex] Complete LaTeX document copied to clipboard")
            else:
                self.clipboard_clear()
                self.clipboard_append("")
                self._log("[latex] No content to copy")

        except Exception as e:
            self._log(f"[latex] copy raw failed: {e}")


    def show(self):
        """Show the window"""
        self.deiconify()
        self.lift()

    def hide(self):
        """Hide the window"""
        self.withdraw()

    def clear(self):
        """Clear the content"""
        self.textview.delete("1.0", "end")
        self._img_refs.clear()

    def set_text_font(self, family=None, size=None):
        """Set text font family and/or size"""
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
        """Set math point size"""
        try:
            self.math_pt = int(pt)
        except Exception as e:
            self._log(f"[latex] set_math_pt error: {e}")

    # ==================== SCHEME METHOD (VISION MODE) ====================

    def set_scheme(self, scheme: str):
        """
        Simplified scheme method - no visual changes for vision mode
        """
        try:
            # Keep everything in default dark theme regardless of mode
            self.configure(bg=self.dark_bg)
            self.textview.tag_configure("speak", background=self.dark_highlight, foreground=self.dark_bg)

            if scheme == "vision":
                self._log("[latex] Vision mode - using default dark theme")
            else:
                self._log("[latex] Default mode - dark theme")

        except Exception as e:
            self._log(f"[latex] set_scheme error: {e}")

    # ==================== WINDOW SIZE CONTROL METHODS ====================

    def _increase_size(self):
        """Increase window size by 100x80 pixels"""
        width, height = self._get_current_size()
        new_width = min(2000, width + 100)
        new_height = min(1200, height + 80)
        self.geometry(f"{new_width}x{new_height}")

    def _decrease_size(self):
        """Decrease window size by 100x80 pixels"""
        width, height = self._get_current_size()
        new_width = max(400, width - 100)
        new_height = max(300, height - 80)
        self.geometry(f"{new_width}x{new_height}")

    def _get_current_size(self):
        """Get current window size from geometry string"""
        geometry = self.geometry()
        if 'x' in geometry and '+' in geometry:
            # Format: "widthxheight+x+y"
            size_part = geometry.split('+')[0]
            width, height = map(int, size_part.split('x'))
            return width, height
        return 800, 600  # Default size

    # ==================== UI HELPER METHODS ====================
    def append_document(self, text, wrap=900, separator="\n" + "=" * 50 + "\n"):
        """Append content to the existing document instead of replacing it"""
        if not text:
            return

        # Store the combined text for raw LaTeX copying
        if self._last_text:
            self._last_text += separator + text
        else:
            self._last_text = text

        try:
            # Get current content
            current_content = self.textview.get("1.0", "end-1c")

            # Add separator if there's existing content
            if current_content.strip():
                self.textview.insert("end", separator)

            # Process and append new content
            blocks = self.split_text_math(text)
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

            # Auto-scroll to bottom
            self.textview.see("end")

        except Exception as e:
            self._log(f"[latex] append error: {e} — plain text fallback")
            self.textview.insert("end", text, ("normal", "tight"))

    def _block_keys(self, e):
        """Block most keys except copy/select all"""
        if (e.state & 0x4) and e.keysym.lower() in ("c", "a"):
            return None
        return "break"

    def _select_all(self, _):
        """Select all text"""
        self.textview.tag_add("sel", "1.0", "end-1c")
        return "break"

    def _popup_menu(self, event):
        """Show right-click context menu"""
        try:
            self._menu.tk_popup(event.x_root, event.y_root)
        finally:
            self._menu.grab_release()


    # ==================== LATEX RENDERING METHODS ====================

    def _is_inline_math(self, expr: str) -> bool:
        """Check if math expression should be rendered inline"""
        s = expr.strip()
        if "\n" in s: return False
        if re.search(r"\\begin\{.*?\}", s): return False
        if re.search(r"\\(pmatrix|bmatrix|Bmatrix|vmatrix|Vmatrix|matrix|cases)\b", s): return False
        if len(s) > 80: return False
        return True

    def _needs_latex_engine(self, s: str) -> bool:
        """Check if expression requires full LaTeX engine"""
        return bool(re.search(
            r"(\\begin\{(?:bmatrix|pmatrix|Bmatrix|vmatrix|Vmatrix|matrix|cases|smallmatrix)\})"
            r"|\\boxed\s*\(" r"|\\boxed\s*\{"
            r"|\\text\s*\{"  r"|\\overset\s*\{" r"|\\underset\s*\{",
            s, flags=re.IGNORECASE
        ))

    def _probe_usetex(self):
        """Check if LaTeX engine is available"""
        if self._usetex_checked:
            return
        self._usetex_checked = True
        try:
            _ = self._render_with_engine_dark(r"\begin{pmatrix}1&2\\3&4\end{pmatrix}", 10, 100, use_usetex=True)
            self._usetex_available = True
            self._log("[latex] usetex available")
        except Exception as e:
            self._usetex_available = False
            self._log(f"[latex] usetex not available ({e}); fallback to MathText)")

    def render_png_bytes(self, latex, fontsize=None, dpi=200):
        """Render LaTeX to PNG bytes (ALWAYS WHITE ON TRANSPARENT)"""
        fontsize = fontsize or self.math_pt
        expr = latex.strip()
        needs_tex = self._needs_latex_engine(expr)
        if needs_tex and not self._usetex_checked:
            self._probe_usetex()
        prefer_usetex = self._usetex_available and (
                needs_tex or "\\begin{pmatrix" in expr or "\\frac" in expr or "\\sqrt" in expr)
        expr = expr.replace("\n", " ")
        try:
            return self._render_with_engine_dark(expr, fontsize, dpi, use_usetex=prefer_usetex)
        except Exception:
            return self._render_with_engine_dark(expr, fontsize, dpi, use_usetex=False)

    def _render_with_engine_dark(self, latex: str, fontsize: int, dpi: int, use_usetex: bool):
        """Render LaTeX with white text on transparent background"""
        preamble = r"\usepackage{amsmath,amssymb,bm}"
        rc = {'text.usetex': bool(use_usetex)}

        # === PERMANENT DARK MODE: Always white text ===
        rc.update({
            'text.color': 'white',
            'axes.edgecolor': 'white',
            'axes.labelcolor': 'white',
            'xtick.color': 'white',
            'ytick.color': 'white'
        })

        if use_usetex:
            preamble += r"\usepackage{xcolor} \definecolor{textcolor}{RGB}{255,255,255} \color{textcolor}"
            rc['text.latex.preamble'] = preamble
            text_color = "white"
        else:
            text_color = "white"

        fig = plt.figure(figsize=(1, 1), dpi=dpi, facecolor='none')
        try:
            with matplotlib.rc_context(rc):
                ax = fig.add_axes([0, 0, 1, 1])
                ax.axis("off")
                ax.text(0.5, 0.5, f"${latex}$", ha="center", va="center",
                        fontsize=fontsize, color=text_color)
                buf = BytesIO()
                fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.02,
                            transparent=True, facecolor='none', edgecolor='none')
                return buf.getvalue()
        finally:
            plt.close(fig)
            plt.close('all')

    def split_text_math(self, text):
        """Split text into math and non-math blocks"""
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
        """Main method to display LaTeX content"""
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

    # ==================== HIGHLIGHT METHODS ====================

    def _word_spans(self):
        """Get word positions for TTS highlighting"""
        content = self.textview.get("1.0", "end-1c")
        spans = []
        for m in re.finditer(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?", content):
            s, e = m.span()
            spans.append((f"1.0+{s}c", f"1.0+{e}c"))
        return spans

    def _prepare_word_spans(self):
        """Prepare word spans for TTS highlighting"""
        try:
            self._hi_spans = self._word_spans()
            self._hi_n = len(self._hi_spans)
        except Exception:
            self._hi_spans, self._hi_n = [], 0

    def set_highlight_index(self, i: int):
        """Set highlight to specific word index"""
        if not getattr(self, "_hi_spans", None):
            return
        i = max(0, min(i, self._hi_n - 1))
        s, e = self._hi_spans[i]
        self.textview.tag_remove("speak", "1.0", "end")
        self.textview.tag_add("speak", s, e)
        self.textview.see(s)

    def set_highlight_ratio(self, r: float):
        """Set highlight based on ratio (0.0 to 1.0)"""
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
        """Clear all highlights"""
        self.textview.tag_remove("speak", "1.0", "end")

    # ==================== DIAGNOSTIC METHOD ====================

    def _run_latex_diagnostics(self):
        """Run LaTeX system diagnostics"""
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


# End Latex Window
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
        self.delay_ms = 144.0
        self.intensity = 0.47  # 0..1
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
        print(f"[ECHO DEBUG] Processing file: {in_wav} -> {out_wav}")
        print(f"[ECHO DEBUG] Echo enabled: {self.enabled}, delay: {self.delay_ms}, intensity: {self.intensity}")
        x, sr = _read_wav_mono(in_wav)
        y = self.process_array(x, sr)
        _write_wav(out_wav, y, sr)
        print(f"[ECHO DEBUG] File processing complete")
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
    MASK_COLOR = "#00FF00"  # For transparency

    def __init__(self, master):
        super().__init__(master)
        self.title("Avatar - Rings")

        # === CIRCULAR WINDOW SETUP ===
        self._scale_factor = 1.0
        self._base_diameter = 480

        # Make window circular
        try:
            self.overrideredirect(True)
            self.wm_attributes("-transparentcolor", self.MASK_COLOR)
            self.configure(bg=self.MASK_COLOR)
        except Exception:
            pass

        self._update_window_size()
        self.center_on_screen()

        # === DRAG & RESIZE SETUP ===
        self._drag_data = {"x": 0, "y": 0}
        self._resize_data = {"active": False, "corner": None}
        self.bind("<Button-1>", self._start_drag)
        self.bind("<B1-Motion>", self._do_drag)

        # Mouse wheel for scaling
        self.bind("<MouseWheel>", self._on_mouse_wheel)
        self.bind("<Button-4>", self._on_mouse_wheel)  # Linux
        self.bind("<Button-5>", self._on_mouse_wheel)  # Linux

        # Canvas setup
        self.canvas = tk.Canvas(self, bg=self.MASK_COLOR, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda e: self.redraw())

        # Bind mouse events to canvas too
        self.canvas.bind("<MouseWheel>", self._on_mouse_wheel)
        self.canvas.bind("<Button-4>", self._on_mouse_wheel)
        self.canvas.bind("<Button-5>", self._on_mouse_wheel)

        self.pad = 8
        self.level = 0
        self._t0 = time.perf_counter()
        self._color_timer = None
        self._running = True

        # Add close button
        self._add_close_button()
        self.start_color_timer()

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

    def _add_close_button(self):
        """Add a subtle black close button that blends with the background"""
        close_btn = tk.Button(self, text="×", font=("Arial", 10),
                              bg="#000000", fg="#666666", border=0, width=2,
                              activebackground="#111111", activeforeground="#888888",
                              command=self.hide)
        close_btn.place(relx=1.0, rely=0.0, anchor="ne", x=-8, y=8)

        # Add hover effect for better usability
        def on_enter(e):
            close_btn.config(fg="#888888", bg="#111111")

        def on_leave(e):
            close_btn.config(fg="#666666", bg="#000000")

        close_btn.bind("<Enter>", on_enter)
        close_btn.bind("<Leave>", on_leave)

    # === MOUSE CONTROLS ===
    def _start_drag(self, e):
        self._drag_data["x"] = e.x_root - self.winfo_x()
        self._drag_data["y"] = e.y_root - self.winfo_y()

    def _do_drag(self, e):
        self.geometry(f"+{e.x_root - self._drag_data['x']}+{e.y_root - self._drag_data['y']}")

    def _on_mouse_wheel(self, event):
        """Handle mouse wheel scaling"""
        SCALE_MIN, SCALE_MAX, SCALE_STEP = 0.3, 2.0, 0.1

        if event.delta > 0 or event.num == 4:  # Scroll up
            new_scale = min(SCALE_MAX, self._scale_factor + SCALE_STEP)
        else:  # Scroll down
            new_scale = max(SCALE_MIN, self._scale_factor - SCALE_STEP)

        if new_scale != self._scale_factor:
            self._scale_factor = new_scale
            self._update_window_size()
            self.log_scale_change()

    def log_scale_change(self):
        """Log scale change"""
        try:
            if hasattr(self.master, 'logln'):
                self.master.logln(f"[avatar] Rings scale: {self._scale_factor:.1f}x")
            else:
                print(f"[avatar] Rings scale: {self._scale_factor:.1f}x")
        except:
            print(f"[avatar] Rings scale: {self._scale_factor:.1f}x")

    # === AVATAR METHODS ===
    def show(self):
        self.deiconify()
        self.lift()

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
        return "#%02x%02x%02x" % (int(r * 255), int(g * 255), int(b * 255))

    def _ring_color(self, k, rings):
        t = (time.perf_counter() - self._t0) * 0.05
        x = ((k / max(1, rings - 1)) + t) % 1.0
        return self._hsv_to_hex(x, 0.9, 1.0)

    def redraw(self):
        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())
        cx, cy = cw // 2, ch // 2

        # Draw circular background
        self.canvas.delete("all")
        r = min(cw, ch) // 2 - self.pad
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

    # === PUBLIC SCALE METHODS ===
    def set_scale(self, scale_factor: float):
        """Programmatically set scale factor"""
        self._scale_factor = max(0.3, min(2.0, scale_factor))
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


class RectAvatarWindow(tk.Toplevel):
    LEVELS = 32
    BG = "#000000"
    MASK_COLOR = "#00FF00"

    # Visual parameters - HORIZONTAL ONLY
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
        super().__init__(master)
        self.title("Avatar - Horizontal Rectangles")

        # === CIRCULAR WINDOW SETUP ===
        self._scale_factor = 1.0
        self._base_diameter = 480

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

        # Canvas
        self.canvas = tk.Canvas(self, bg=self.MASK_COLOR, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda e: self.redraw())
        self.canvas.bind("<MouseWheel>", self._on_mouse_wheel)
        self.canvas.bind("<Button-4>", self._on_mouse_wheel)
        self.canvas.bind("<Button-5>", self._on_mouse_wheel)

        self.pad = 8
        self.level = 0
        self._last_time = time.perf_counter()
        self._particles = []
        self._run = True

        # Close button
        self._add_close_button()
        self._tick = self.after(16, self._loop)

    def _add_close_button(self):
        """Add a subtle black close button"""
        close_btn = tk.Button(self, text="×", font=("Arial", 10),
                              bg="#000000", fg="#666666", border=0, width=2,
                              activebackground="#111111", activeforeground="#888888",
                              command=self.hide)
        close_btn.place(relx=1.0, rely=0.0, anchor="ne", x=-8, y=8)

        # Hover effects
        def on_enter(e):
            close_btn.config(fg="#888888", bg="#111111")

        def on_leave(e):
            close_btn.config(fg="#666666", bg="#000000")

        close_btn.bind("<Enter>", on_enter)
        close_btn.bind("<Leave>", on_leave)

    def _update_window_size(self):
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
        SCALE_MIN, SCALE_MAX, SCALE_STEP = 0.3, 2.0, 0.1

        if event.delta > 0 or event.num == 4:
            new_scale = min(SCALE_MAX, self._scale_factor + SCALE_STEP)
        else:
            new_scale = max(SCALE_MIN, self._scale_factor - SCALE_STEP)

        if new_scale != self._scale_factor:
            self._scale_factor = new_scale
            self._update_window_size()
            self.log_scale_change()

    def log_scale_change(self):
        try:
            if hasattr(self.master, 'logln'):
                self.master.logln(f"[avatar] Horizontal Rectangles scale: {self._scale_factor:.1f}x")
            else:
                print(f"[avatar] Horizontal Rectangles scale: {self._scale_factor:.1f}x")
        except:
            print(f"[avatar] Horizontal Rectangles scale: {self._scale_factor:.1f}x")

    def destroy(self):
        self._run = False
        try:
            if self._tick:
                self.after_cancel(self._tick)
        except Exception:
            pass
        super().destroy()

    def show(self):
        self.deiconify()
        self.lift()

    def hide(self):
        self.withdraw()

    def set_level(self, level: int):
        self.level = max(0, min(self.LEVELS - 1, int(level)))

    def _hsv_to_hex(self, h, s, v):
        i = int(h * 6)
        f = h * 6 - i
        p = v * (1 - s)
        q = v * (1 - f * s)
        t = v * (1 - (1 - f) * s)
        i %= 6
        r, g, b = [(v, t, p), (q, v, p), (p, v, t),
                   (p, q, v), (t, p, v), (v, p, q)][i]
        return "#%02x%02x%02x" % (int(r * 255), int(g * 255), int(b * 255))

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

    def _circle_geom(self):
        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())
        cx, cy = cw // 2, ch // 2
        r = max(1, min(cw, ch) // 2 - self.pad)
        return cx, cy, r

    def _uniform_point_in_disc(self, cx, cy, r):
        inner_r = max(2, r * 0.85)
        u = np.random.random()
        theta = 2 * np.pi * np.random.random()
        rho = inner_r * np.sqrt(u)
        return int(cx + rho * np.cos(theta)), int(cy + rho * np.sin(theta))

    def _spawn(self, n):
        """ONLY HORIZONTAL rectangles - NO vertical ones"""
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

            # ONLY HORIZONTAL RECTANGLES - no vertical option
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
                "color": col, "vertical": False  # Always horizontal
            })

        if len(self._particles) > self.MAX_PARTICLES:
            self._particles = self._particles[-self.MAX_PARTICLES:]

    def _loop(self):
        if not self._run:
            return
        self.redraw()
        self._tick = self.after(16, self._loop)

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

            # Update position - only horizontal movement logic needed
            p["x1"] += p["vx"] * dt
            p["x2"] += p["vx"] * dt
            p["y1"] += p["vy"] * dt
            p["y2"] += p["vy"] * dt

            # Keep within circle bounds - simplified for horizontal only
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

    # Public scale methods
    def set_scale(self, scale_factor: float):
        self._scale_factor = max(0.3, min(2.0, scale_factor))
        self._update_window_size()
        self.log_scale_change()

    def get_scale(self) -> float:
        return self._scale_factor

    def reset_scale(self):
        self._scale_factor = 1.0
        self._update_window_size()
        self.log_scale_change()


class RectAvatarWindow2(tk.Toplevel):
    """Rectangles (circular window) with enhanced features"""

    LEVELS = 32
    BG = "#000000"
    MASK_COLOR = "#00FF00"

    # Enhanced parameters
    CIRCLE_DIAM = 900
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
    CENTER_PULL = 0.07
    VERTICAL_PROPORTION = 0.40
    EDGE_INSET_F = 1.00
    SPAWN_RADIUS_F = 0.90

    # Scaling parameters
    SCALE_MIN = 0.3
    SCALE_MAX = 2.0
    SCALE_STEP = 0.1

    def __init__(self, master):
        super().__init__(master)
        self.title("Avatar — Rectangles 2")

        # === CIRCULAR WINDOW SETUP ===
        self._scale_factor = 1.0
        self._base_diameter = self.CIRCLE_DIAM

        # Make the toplevel circular via transparent color
        try:
            self.overrideredirect(True)
            self.wm_attributes("-transparentcolor", self.MASK_COLOR)
            self.configure(bg=self.MASK_COLOR)
        except Exception:
            # if unsupported, window stays rectangular
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
        self.canvas.bind("<Configure>", lambda e: self.redraw())

        # Bind mouse events to canvas too
        self.canvas.bind("<MouseWheel>", self._on_mouse_wheel)
        self.canvas.bind("<Button-4>", self._on_mouse_wheel)
        self.canvas.bind("<Button-5>", self._on_mouse_wheel)

        self.pad = 8
        self.level = 0
        self._last_time = time.perf_counter()
        self._particles = []
        self._run = True

        # Add close button
        self._add_close_button()
        self._tick = self.after(16, self._loop)

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

    def _add_close_button(self):
        """Add a subtle black close button that blends with the background"""
        close_btn = tk.Button(self, text="×", font=("Arial", 10),
                              bg="#000000", fg="#666666", border=0, width=2,
                              activebackground="#111111", activeforeground="#888888",
                              command=self.hide)
        close_btn.place(relx=1.0, rely=0.0, anchor="ne", x=-8, y=8)

        # Add hover effect for better usability
        def on_enter(e):
            close_btn.config(fg="#888888", bg="#111111")

        def on_leave(e):
            close_btn.config(fg="#666666", bg="#000000")

        close_btn.bind("<Enter>", on_enter)
        close_btn.bind("<Leave>", on_leave)

    # === MOUSE CONTROLS ===
    def _start_drag(self, e):
        self._drag_data["x"] = e.x_root - self.winfo_x()
        self._drag_data["y"] = e.y_root - self.winfo_y()

    def _do_drag(self, e):
        self.geometry(f"+{e.x_root - self._drag_data['x']}+{e.y_root - self._drag_data['y']}")

    def _on_mouse_wheel(self, event):
        """Handle mouse wheel events for scaling"""
        if event.delta > 0 or event.num == 4:  # Scroll up
            new_scale = min(self.SCALE_MAX, self._scale_factor + self.SCALE_STEP)
        else:  # Scroll down
            new_scale = max(self.SCALE_MIN, self._scale_factor - self.SCALE_STEP)

        if new_scale != self._scale_factor:
            self._scale_factor = new_scale
            self._update_window_size()
            self.log_scale_change()

    def log_scale_change(self):
        """Log scale change"""
        try:
            if hasattr(self.master, 'logln'):
                self.master.logln(f"[avatar] Rectangles 2 scale: {self._scale_factor:.1f}x")
            else:
                print(f"[avatar] Rectangles 2 scale: {self._scale_factor:.1f}x")
        except:
            print(f"[avatar] Rectangles 2 scale: {self._scale_factor:.1f}x")

    # === WINDOW MANAGEMENT ===
    def show(self):
        self.deiconify()
        self.lift()

    def hide(self):
        self.withdraw()

    def destroy(self):
        self._run = False
        try:
            if self._tick:
                self.after_cancel(self._tick)
        except Exception:
            pass
        super().destroy()

    # === AVATAR CORE METHODS ===
    def set_level(self, level: int):
        self.level = max(0, min(self.LEVELS - 1, int(level)))

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

    def _spawn_count(self):
        """Calculate how many particles to spawn based on level"""
        if self.level <= self.LEVEL_DEADZONE:
            return 0
        usable = self.LEVELS - 1 - self.LEVEL_DEADZONE
        if usable <= 0:
            return 0
        x = (self.level - self.LEVEL_DEADZONE) / float(usable)
        x = max(0.0, min(1.0, x))
        return int(0.5 + self.MIN_SPAWN +
                   (self.SPAWN_AT_MAX_LVL - self.MIN_SPAWN) * (x ** self.SPAWN_GAMMA))

    def _circle_geom(self):
        """Get circle geometry"""
        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())
        cx, cy = cw // 2, ch // 2
        r = max(1, min(cw, ch) // 2 - self.pad)
        return cx, cy, r

    def _uniform_point_in_disc(self, cx, cy, r):
        """Sample uniformly within spawn radius"""
        inner_r = max(2, r * float(self.SPAWN_RADIUS_F))
        u = np.random.random()
        theta = 2 * np.pi * np.random.random()
        rho = inner_r * np.sqrt(u)
        return int(cx + rho * np.cos(theta)), int(cy + rho * np.sin(theta))

    def _spawn(self, n):
        """Spawn particles within circular bounds"""
        cx, cy, r = self._circle_geom()
        if r <= 4:
            return

        d = 2 * r
        min_len_h = max(6, int(d * (self.RECT_MIN_LEN_F * 1.00)))
        max_len_h = max(min_len_h + 2, int(d * (self.RECT_MAX_LEN_F * 1.00)))
        min_len_v = min_len_h
        max_len_v = max_len_h

        thick = max(3, int(r * (self.RECT_THICK_F * 1.00)))
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
                chord_half = int(math.sqrt(max(0, r * r - dx * dx)) * float(self.EDGE_INSET_F))
                halfL = min(L // 2, chord_half)
                x1, x2 = x0 - thick_v // 2, x0 + thick_v // 2
                y1, y2 = y0 - halfL, y0 + halfL
            else:
                L = np.random.randint(min_len_h, max_len_h)
                dy = abs(y0 - cy)
                chord_half = int(math.sqrt(max(0, r * r - dy * dy)) * float(self.EDGE_INSET_F))
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

    def _loop(self):
        """Main animation loop"""
        if not self._run:
            return
        self.redraw()
        self._tick = self.after(16, self._loop)

    def redraw(self):
        """Redraw the entire avatar"""
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

        # Spawn new particles if needed
        spawn = self._spawn_count()
        if spawn > 0:
            self._spawn(spawn)

        # Update and draw particles
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
            if float(self.CENTER_PULL) > 0.0:
                mx = 0.5 * (p["x1"] + p["x2"])
                my = 0.5 * (p["y1"] + p["y2"])
                pull = float(self.CENTER_PULL) * dt
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

            # Circle-aware clipping by chord (prevents leakage outside circle)
            if p.get("vertical", False):
                dx = mx - cx
                max_half_thick = max(0, int(math.sqrt(max(0, r * r - dx * dx))))
                half_thick = max(1, int((p["x2"] - p["x1"]) * 0.5))
                half_thick = min(half_thick, max_half_thick)
                p["x1"], p["x2"] = mx - half_thick, mx + half_thick

                chord_half = max(0, int(math.sqrt(max(0, r * r - dx * dx)) * float(self.EDGE_INSET_F)))
                halfL = min(int((p["y2"] - p["y1"]) * 0.5), chord_half)
                p["y1"], p["y2"] = my - halfL, my + halfL
            else:
                dy = my - cy
                max_half_thick = max(0, int(math.sqrt(max(0, r * r - dy * dy))))
                half_thick = max(1, int((p["y2"] - p["y1"]) * 0.5))
                half_thick = min(half_thick, max_half_thick)
                p["y1"], p["y2"] = my - half_thick, my + half_thick

                chord_half = max(0, int(math.sqrt(max(0, r * r - dy * dy)) * float(self.EDGE_INSET_F)))
                halfL = min(int((p["x2"] - p["x1"]) * 0.5), chord_half)
                p["x1"], p["x2"] = mx - halfL, mx + halfL

            # Fade via stipple
            t = age / p["life"]
            stipples = ("", "gray12", "gray25", "gray50", "gray75")
            idx = min(len(stipples) - 1, int(t * len(stipples)))
            stipple = stipples[idx]

            # Draw the rectangle
            self.canvas.create_rectangle(
                int(p["x1"]), int(p["y1"]), int(p["x2"]), int(p["y2"]),
                fill=p["color"], outline=p["color"],
                stipple=stipple if stipple else None
            )
            alive.append(p)

        self._particles = alive

    # === PUBLIC SCALE METHODS ===
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


class RadialPulseAvatar(tk.Toplevel):
    """Minimalist avatar: red dot with radial lines that pulse with speech"""

    LEVELS = 32
    BG = "#000000"
    MASK_COLOR = "#00FF00"
    DOT_COLOR = "#FF0000"  # Red dot
    LINE_COLOR = "#FF3333"  # Slightly lighter red for lines

    # Ellipse parameters
    ELLIPSE_WIDTH_RATIO = 0.6  # Width as fraction of height (vertical ellipse)
    BASE_HEIGHT = 600  # Base window height

    # Animation parameters
    MAX_LINES = 24  # Number of radial lines
    MAX_LINE_LENGTH = 0.8  # Max line length as fraction of window size
    PULSE_SMOOTHING = 0.3  # Smoothing factor for level changes

    def __init__(self, master):
        super().__init__(master)
        self.title("Avatar - Radial Pulse")

        # === ELLIPTICAL WINDOW SETUP ===
        self._scale_factor = 1.0
        self._base_height = self.BASE_HEIGHT
        self._base_width = int(self._base_height * self.ELLIPSE_WIDTH_RATIO)

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
        self.canvas.bind("<Configure>", lambda e: self.redraw())
        self.canvas.bind("<MouseWheel>", self._on_mouse_wheel)
        self.canvas.bind("<Button-4>", self._on_mouse_wheel)
        self.canvas.bind("<Button-5>", self._on_mouse_wheel)

        # Animation state
        self.level = 0
        self._smoothed_level = 0.0
        self._last_time = time.perf_counter()
        self._run = True

        # Close button
        self._add_close_button()
        self._tick = self.after(16, self._loop)

    def _update_window_size(self):
        """Update window size based on current scale factor - elliptical shape"""
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

    def _add_close_button(self):
        """Add a subtle black close button"""
        close_btn = tk.Button(self, text="×", font=("Arial", 10),
                              bg="#000000", fg="#666666", border=0, width=2,
                              activebackground="#111111", activeforeground="#888888",
                              command=self.hide)
        close_btn.place(relx=1.0, rely=0.0, anchor="ne", x=-8, y=8)

        def on_enter(e):
            close_btn.config(fg="#888888", bg="#111111")

        def on_leave(e):
            close_btn.config(fg="#666666", bg="#000000")

        close_btn.bind("<Enter>", on_enter)
        close_btn.bind("<Leave>", on_leave)

    def _start_drag(self, e):
        self._drag_data["x"] = e.x_root - self.winfo_x()
        self._drag_data["y"] = e.y_root - self.winfo_y()

    def _do_drag(self, e):
        self.geometry(f"+{e.x_root - self._drag_data['x']}+{e.y_root - self._drag_data['y']}")

    def _on_mouse_wheel(self, event):
        SCALE_MIN, SCALE_MAX, SCALE_STEP = 0.3, 2.0, 0.1

        if event.delta > 0 or event.num == 4:
            new_scale = min(SCALE_MAX, self._scale_factor + SCALE_STEP)
        else:
            new_scale = max(SCALE_MIN, self._scale_factor - SCALE_STEP)

        if new_scale != self._scale_factor:
            self._scale_factor = new_scale
            self._update_window_size()
            self.log_scale_change()

    def log_scale_change(self):
        try:
            if hasattr(self.master, 'logln'):
                self.master.logln(f"[avatar] Radial Pulse scale: {self._scale_factor:.1f}x")
            else:
                print(f"[avatar] Radial Pulse scale: {self._scale_factor:.1f}x")
        except:
            print(f"[avatar] Radial Pulse scale: {self._scale_factor:.1f}x")

    def show(self):
        self.deiconify()
        self.lift()

    def hide(self):
        self.withdraw()

    def destroy(self):
        self._run = False
        try:
            if self._tick:
                self.after_cancel(self._tick)
        except Exception:
            pass
        super().destroy()

    def set_level(self, level: int):
        self.level = max(0, min(self.LEVELS - 1, int(level)))

    def _loop(self):
        """Main animation loop"""
        if not self._run:
            return

        now = time.perf_counter()
        dt = max(0.001, now - self._last_time)
        self._last_time = now

        # Smooth level changes for fluid animation
        target_level = self.level / float(self.LEVELS - 1)
        self._smoothed_level += (target_level - self._smoothed_level) * self.PULSE_SMOOTHING

        self.redraw()
        self._tick = self.after(16, self._loop)

    def redraw(self):
        """Draw the radial pulse avatar"""
        self.canvas.delete("all")

        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())
        cx, cy = cw // 2, ch // 2

        # Draw elliptical background
        ellipse_width = int(cw * 0.95)
        ellipse_height = int(ch * 0.95)
        self.canvas.create_oval(
            cx - ellipse_width // 2, cy - ellipse_height // 2,
            cx + ellipse_width // 2, cy + ellipse_height // 2,
            fill=self.BG, outline=self.BG
        )

        # Calculate pulse intensity with some variation for organic feel
        base_intensity = self._smoothed_level
        pulse_intensity = base_intensity * (0.8 + 0.4 * math.sin(time.perf_counter() * 8))

        # Draw central red dot (size varies slightly with intensity)
        dot_radius = max(2, int(4 + pulse_intensity * 6))
        self.canvas.create_oval(
            cx - dot_radius, cy - dot_radius,
            cx + dot_radius, cy + dot_radius,
            fill=self.DOT_COLOR, outline=self.DOT_COLOR
        )

        # Draw radial lines
        if pulse_intensity > 0.05:  # Only draw lines when there's significant audio
            num_lines = self.MAX_LINES
            max_line_length = min(cw, ch) * self.MAX_LINE_LENGTH * pulse_intensity

            for i in range(num_lines):
                angle = (2 * math.pi * i) / num_lines

                # Add slight randomness to line lengths for organic feel
                line_variation = 0.7 + 0.6 * math.sin(angle * 3 + time.perf_counter() * 6)
                line_length = max_line_length * line_variation

                # Calculate line end point
                end_x = cx + line_length * math.cos(angle)
                end_y = cy + line_length * math.sin(angle)

                # Line width varies with intensity and angle
                line_width = max(1, int(1 + pulse_intensity * 3))

                # Draw the radial line
                self.canvas.create_line(
                    cx, cy, end_x, end_y,
                    fill=self.LINE_COLOR,
                    width=line_width,
                    capstyle=tk.ROUND
                )

    # Public scale methods
    def set_scale(self, scale_factor: float):
        self._scale_factor = max(0.3, min(2.0, scale_factor))
        self._update_window_size()
        self.log_scale_change()

    def get_scale(self) -> float:
        return self._scale_factor

    def reset_scale(self):
        self._scale_factor = 1.0
        self._update_window_size()
        self.log_scale_change()


# === WEB SEARCH WINDOW CLASS ===
class WebSearchWindow(tk.Toplevel):
    def __init__(self, master, log_fn=None):
        super().__init__(master)
        self.title("Web Search")
        self.geometry("980x740")
        self.minsize(780, 620)
        self.protocol("WM_DELETE_WINDOW", self.withdraw)
        self._log = log_fn or (lambda msg: None)

        self.queue = queue.Queue()
        self.in_progress = False

        # Initialize tracking variables
        self.thumb_cache = {}
        self.image_bytes_cache = {}
        self.thumb_target = {}
        self.thumb_placeholder = {}
        self._all_results = []

        # LaTeX preview window for search results
        self.latex_win = None
        self.latex_auto = tk.BooleanVar(value=True)

        self._build_ui()
        self._build_strip_lights()
        self._tick_strip_lights()
        self._poll_queue()

    def _build_ui(self):
        # Main container
        self.main_container = ttk.Frame(self, padding=10)
        self.main_container.pack(fill=tk.BOTH, expand=True, padx=36, pady=36)

        # Query section
        ttk.Label(self.main_container, text="Query:").pack(anchor="w")
        self.txt_in = tk.Text(self.main_container, height=3, wrap="word")
        self.txt_in.pack(fill=tk.X, pady=(2, 8))

        # LaTeX controls for search results
        latex_controls = ttk.Frame(self.main_container)
        latex_controls.pack(fill=tk.X, pady=(0, 8))

        ttk.Checkbutton(
            latex_controls, text="Auto LaTeX preview for search results",
            variable=self.latex_auto
        ).pack(side="left", padx=(0, 10))

        ttk.Button(
            latex_controls, text="Show/Hide LaTeX",
            command=self.toggle_latex
        ).pack(side="left", padx=(0, 10))

        ttk.Button(
            latex_controls, text="Copy Raw LaTeX",
            command=self.copy_raw_latex
        ).pack(side="left")

        self.btn = ttk.Button(self.main_container, text="Search & Summarise", command=self.on_go)
        self.btn.pack(pady=(0, 8), anchor="w")

        # Output section - create a container for results that we can refresh
        self.results_container = ttk.Frame(self.main_container)
        self.results_container.pack(fill=tk.BOTH, expand=True)

        # Create initial results display
        self._create_results_display()

        # Status
        self.status = tk.StringVar(value="Ready.")
        ttk.Label(self.main_container, textvariable=self.status).pack(anchor="w", pady=(6, 0))

    def _create_results_display(self):
        """Create or recreate the results display area"""
        # Clear existing results display if it exists
        if hasattr(self, 'results_frame') and self.results_frame:
            try:
                self.results_frame.destroy()
            except:
                pass

        # Create new results frame
        self.results_frame = ttk.Frame(self.results_container)
        self.results_frame.pack(fill=tk.BOTH, expand=True)

        # Output text area
        ttk.Label(self.results_frame, text="Output:").pack(anchor="w")
        self.txt_out = tk.Text(self.results_frame, wrap="word", height=12)
        self.txt_out.pack(fill=tk.BOTH, expand=True)

        # Scrollbar for text output
        sbr = ttk.Scrollbar(self.results_frame, orient="vertical", command=self.txt_out.yview)
        sbr.place(relx=1.0, rely=0, relheight=0.7, anchor="ne")
        self.txt_out.config(yscrollcommand=sbr.set)

        # Images container
        self.images_container = ttk.Frame(self.results_frame)
        self.images_container.pack(fill=tk.X, pady=8)

    def _ensure_latex_window(self):
        """Create LaTeX window if it doesn't exist"""
        if self.latex_win is None or not self.latex_win.winfo_exists():
            # Use the same settings as the main app's LaTeX window
            self.latex_win = LatexWindow(
                self.master,  # Use master (main app) as parent for consistency
                log_fn=self._log,
                text_family="Segoe UI",
                text_size=12,
                math_pt=8
            )

    def toggle_latex(self):
        """Toggle LaTeX window using main app's system"""
        try:
            if hasattr(self, 'main_app') and self.main_app:
                self.main_app.toggle_latex()
            elif hasattr(self, 'master') and hasattr(self.master, 'ensure_latex_window'):
                latex_win = self.master.ensure_latex_window("search")
                if latex_win.state() == "withdrawn":
                    latex_win.show()
                else:
                    latex_win.hide()
        except Exception as e:
            error_msg = f"[search] toggle latex error: {e}"
            if hasattr(self, 'logln'):
                self.logln(error_msg)
            else:
                print(error_msg)

    def copy_raw_latex(self):
        """Copy raw LaTeX source to clipboard with proper document structure"""
        try:
            content = self._last_text or ""

            if content.strip():
                # Create a complete LaTeX document
                latex_document = f"""\\documentclass{{article}}
    \\usepackage{{amsmath}}
    \\usepackage{{amssymb}}
    \\usepackage{{graphicx}}
    \\usepackage{{hyperref}}
    \\begin{{document}}

    {content}

    \\end{{document}}"""

                self.clipboard_clear()
                self.clipboard_append(latex_document)
                self._log("[latex] Complete LaTeX document copied to clipboard")
            else:
                self.clipboard_clear()
                self.clipboard_append("")
                self._log("[latex] No content to copy")

        except Exception as e:
            self._log(f"[latex] copy raw failed: {e}")





    def preview_latex(self, content: str, context="text"):
        """Preview LaTeX content with append/replace option"""
        if not self.latex_auto.get():
            return

        def _go():
            try:
                latex_win = self.ensure_latex_window(context)
                latex_win.show()

                # === CHECK APPEND MODE ===
                if self.latex_append_mode.get():
                    # APPEND MODE - add to existing content
                    latex_win.append_document(content)
                    self.logln(f"[latex] 📝 Appended to {context} window")
                else:
                    # REPLACE MODE - clear and show new content (original behavior)
                    latex_win.show_document(content)
                    self.logln(f"[latex] 🔄 Showing in {context} window (replace mode)")

                self._current_latex_context = context

            except Exception as e:
                self.logln(f"[latex] preview error ({context}): {e}")

        self.master.after(0, _go)

    def _build_strip_lights(self):
        self.strip_canvas = tk.Canvas(self, highlightthickness=0, bg="white")
        self.strip_canvas.place(relx=0, rely=0, relwidth=1, relheight=1)
        tk.Misc.lower(self.strip_canvas)

        self.strip_items = []
        self.strip_colors = ["#ff3b30", "#ff9500", "#ffcc00", "#34c759", "#5ac8fa", "#007aff", "#af52de"]
        self.strip_step = 0

        margin = 10
        thickness = 40
        seg = 44

        def make_rect(x1, y1, x2, y2):
            rid = self.strip_canvas.create_rectangle(x1, y1, x2, y2, width=0, fill="white")
            self.strip_items.append(rid)

        def rebuild(_evt=None):
            self.strip_canvas.delete("all")
            self.strip_items.clear()
            W = max(1, self.winfo_width())
            H = max(1, self.winfo_height())
            # bottom
            x = margin
            y = H - margin - thickness
            while x + seg <= W - margin:
                make_rect(x, y, x + seg - 3, y + thickness)
                x += seg
            # left
            x = margin
            y = H - margin - thickness
            while y - seg >= margin:
                make_rect(x, y - seg + 3, x + thickness, y)
                y -= seg
            # top
            x = margin + thickness
            y = margin
            while x + seg <= W - margin - thickness:
                make_rect(x, y, x + seg - 3, y + thickness)
                x += seg
            # right
            x = W - margin - thickness
            y = margin + thickness
            while y + seg <= H - margin - thickness:
                make_rect(x, y, x + thickness, y + seg - 3)
                y += seg

        self.bind("<Configure>", rebuild)
        self.rebuild = rebuild
        rebuild()

    def _tick_strip_lights(self):
        if self.in_progress and self.strip_items:
            n = len(self.strip_items)
            self.strip_step = (self.strip_step + 1) % (n * len(self.strip_colors))
            for i, rid in enumerate(self.strip_items):
                idx = (self.strip_step + i) % len(self.strip_colors)
                self.strip_canvas.itemconfig(rid, fill=self.strip_colors[idx])
        else:
            for rid in self.strip_items:
                self.strip_canvas.itemconfig(rid, fill="white")
        self.after(90, self._tick_strip_lights)

    def show(self):
        self.deiconify()
        self.lift()

    def hide(self):
        """Hide the window (called by close button)"""
        try:
            self.withdraw()
        except Exception as e:
            print(f"Error hiding window: {e}")

    def log(self, msg: str):
        self.status.set(msg)
        self.update_idletasks()

    def on_go(self):
        if self.in_progress:
            self.log("Already running…")
            return

        q = self.txt_in.get("1.0", "end").strip()
        if not q:
            messagebox.showinfo("Info", "Please type a query.")
            return

        q = self.normalize_query(q)

        # DON'T recreate the results display - just clear the existing one
        self.txt_out.delete("1.0", "end")

        # Clear the images container
        for widget in self.images_container.winfo_children():
            widget.destroy()

        # Clear tracking variables (keep cache for performance)
        self.image_bytes_cache.clear()
        self.thumb_target.clear()
        self.thumb_placeholder.clear()
        self._all_results.clear()
        # Reset accumulated LaTeX content for new search
        self._accumulated_latex_content = ""

        self.in_progress = True
        self.btn.config(state="disabled")
        self.log("Searching…")

        # Start search in thread
        try:
            threading.Thread(target=self._thread_run, args=(q,), daemon=True).start()
        except Exception as e:
            self.log(f"Failed to start search thread: {e}")
            self.in_progress = False
            self.btn.config(state="normal")

    def _thread_run(self, query: str):
        # === SEARCH PROGRESS AUDIBLE INDICATOR ===
        if hasattr(self.master, 'start_search_progress_indicator'):
            self.master.start_search_progress_indicator()


        try:
            results = self.brave_search(query, 6)
            self.queue.put(("searched", len(results)))

            for idx, it in enumerate(results, 1):
                self.queue.put(("status", f"Processing result {idx}/{len(results)}..."))

                try:
                    html = self.polite_fetch(it.url)
                    if not html:
                        self.log(f"Failed to fetch: {it.url}")
                        self.queue.put(("article", {
                            "title": it.title, "url": it.url, "pubdate": None,
                            "summary": "Fetch failed - could not retrieve content.", "images": []
                        }))
                        continue

                    it.pubdate = self.guess_pubdate(html)
                    it.image_urls = self.extract_images(html, it.url)
                    text = self.extract_readable(html, it.url)

                    if len(text) < 400:
                        it.summary = "Not enough readable text to summarise."
                    else:
                        try:
                            it.summary = self.summarise_with_qwen(text, it.url, it.pubdate)
                            if not it.summary or "failed" in it.summary.lower():
                                it.summary = "Summarization unavailable for this content."
                        except Exception as e:
                            self.log(f"Summarization error: {e}")
                            it.summary = f"Summarization error: {str(e)}"

                    self.queue.put(("article", {
                        "title": it.title, "url": it.url, "pubdate": it.pubdate,
                        "summary": it.summary, "images": it.image_urls
                    }))
                    time.sleep(0.6)

                except Exception as e:
                    self.log(f"Error processing result {idx}: {e}")
                    continue

            self.queue.put(("done", None))
        except Exception as e:
            self.log(f"Search thread error: {e}")
            self.queue.put(("error", str(e)))
        # === STOP ON ERROR ===
        if hasattr(self.master, 'stop_search_progress_indicator'):
            self.master.stop_search_progress_indicator()


    def _create_final_summary(self) -> str:
        """Create a consolidated summary of all search results"""
        if not self._all_results:
            return "No results to summarize."

        # Use the ACCUMULATED content that already has all the equations
        if hasattr(self, '_accumulated_latex_content') and self._accumulated_latex_content:
            final_text = "COMPLETE SEARCH RESULTS WITH EQUATIONS:\n\n" + self._accumulated_latex_content
        else:
            # Fallback to the original summary method
            summary_parts = ["Search Results Summary:"]
            for i, result in enumerate(self._all_results, 1):
                title = result.get('title', 'No title')
                summary = result.get('summary', 'No summary available')

                # Clean up the summary text but preserve equations
                clean_summary = re.sub(r'=+', '', summary)
                clean_summary = re.sub(r'\s+', ' ', clean_summary).strip()
                clean_summary = re.sub(r'[\*#_-]{2,}', '', clean_summary)
                clean_summary = re.sub(r'^(summary|result|findings?):?\s*', '', clean_summary, flags=re.IGNORECASE)

                if len(clean_summary) > 250:
                    clean_summary = clean_summary[:247] + "..."

                summary_parts.append(f"Result {i}: {title}")
                if clean_summary and clean_summary != "No summary available":
                    summary_parts.append(f"{clean_summary}")
                summary_parts.append("")

            final_text = "\n".join(summary_parts)
            final_text = re.sub(r'\n{3,}', '\n\n', final_text)

        # PREVIEW LATEX FOR THE FINAL ACCUMULATED CONTENT
        self.preview_latex(final_text)

        return final_text.strip()

    def _poll_queue(self):
        try:
            while True:
                msg, payload = self.queue.get_nowait()
                try:
                    if msg == "status":
                        self.log(payload)
                    elif msg == "searched":
                        n = payload
                        self.log(f"Found {n} results. Summarising…")
                        self.play_ding()

                    elif msg == "article":
                        if self.in_progress:  # Only process if we're still searching
                            data = payload
                            block = (
                                    f"Title: {data['title']}\n"
                                    f"URL: {data['url']}\n"
                                    + (f"Publish date: {data['pubdate']}\n" if data['pubdate'] else "")
                                    + f"\nSummary:\n{data['summary']}"
                            )
                            self.append_output(block)
                            self._show_images(data.get("images", []), data['title'], data['url'])
                            self._all_results.append(data)

                    elif msg == "done":
                        self.log("Done.")
                        self.in_progress = False
                        self.btn.config(state="normal")
                        self.play_dong()
                        # === STOP PROGRESS INDICATOR ===
                        if hasattr(self.master, 'stop_search_progress_indicator'):
                            self.master.stop_search_progress_indicator()

                        if self._all_results:
                            summary_text = self._create_final_summary()
                            if hasattr(self, 'synthesize_search_results'):
                                self.synthesize_search_results(summary_text)

                    elif msg == "imgbytes":
                        url, data = payload
                        self.image_bytes_cache[url] = data
                        target = self.thumb_target.pop(url, None)
                        if target and self.winfo_exists():
                            self._create_thumb_from_bytes(url, data, target_frame=target)

                    elif msg == "imgfail":
                        url = payload
                        ph = self.thumb_placeholder.pop(url, None)
                        if ph and ph.winfo_exists():
                            ph.config(text="Image blocked or unavailable")

                    elif msg == "error":
                        self.in_progress = False
                        self.btn.config(state="normal")
                        self.log(f"Error: {payload}")
                    # ===  STOP ON ERROR ===
                    if hasattr(self.master, 'stop_search_progress_indicator'):
                        self.master.stop_search_progress_indicator()

                except Exception as e:
                    self.log(f"Error processing queue message {msg}: {e}")

        except queue.Empty:
            pass
        except Exception as e:
            self.log(f"Queue polling error: {e}")
            self.in_progress = False
            self.btn.config(state="normal")
        # ===  STOP ON GENERAL ERROR ===
        if hasattr(self.master, 'stop_search_progress_indicator'):
            self.master.stop_search_progress_indicator()


        if self.winfo_exists():
            self.after(120, self._poll_queue)

    def append_output(self, block: str):
        self.txt_out.insert("end", block + "\n\n")
        self.txt_out.see("end")

        # ACCUMULATE all results for LaTeX preview instead of overwriting
        if not hasattr(self, '_accumulated_latex_content'):
            self._accumulated_latex_content = ""

        # Add this result to the accumulated content
        self._accumulated_latex_content += block + "\n\n"

        # Preview the ACCUMULATED content (all results so far)
        self.preview_latex(self._accumulated_latex_content)

    def _show_images(self, urls: List[str], article_title: str, article_url: str):
        try:
            section = ttk.Frame(self.images_container)
            section.pack(fill=tk.X, pady=6)

            hdr = ttk.Label(section, text=f"Images: {article_title}", cursor="hand2")
            hdr.pack(anchor="w")
            hdr.bind("<Button-1>", lambda e, u=article_url: webbrowser.open(u))

            if not urls:
                ttk.Label(section, text="(No images found.)").pack(anchor="w")
                return

            for i, u in enumerate(urls[:3]):
                try:
                    if u in self.thumb_cache:
                        self._place_thumb(u, self.thumb_cache[u], target_frame=section)
                    elif u in self.image_bytes_cache:
                        self._create_thumb_from_bytes(u, self.image_bytes_cache[u], target_frame=section)
                    else:
                        ph = ttk.Label(section, text=f"Loading image {i + 1}…")
                        ph.pack(side=tk.LEFT, padx=6)
                        self.thumb_target[u] = section
                        self.thumb_placeholder[u] = ph
                        threading.Thread(target=self._thread_fetch_image, args=(u,), daemon=True).start()
                except Exception as e:
                    self.log(f"Error displaying image {i}: {e}")
                    continue
        except Exception as e:
            self.log(f"Error in _show_images: {e}")

    def _thread_fetch_image(self, url: str):
        try:
            headers = {"User-Agent": "LocalAI-ResearchBot/1.0", "Referer": url}
            with httpx.Client(timeout=15.0, headers=headers, follow_redirects=True) as client:
                resp = client.get(url)
                resp.raise_for_status()
                data = resp.content
            self.queue.put(("imgbytes", (url, data)))
        except Exception:
            self.queue.put(("imgfail", url))

    def _create_thumb_from_bytes(self, url: str, data: bytes, target_frame=None):
        try:
            img = Image.open(BytesIO(data))
            img.thumbnail((220, 140))
            tkimg = ImageTk.PhotoImage(img)
            self.thumb_cache[url] = tkimg
            ph = self.thumb_placeholder.pop(url, None)
            if ph and ph.winfo_exists():
                ph.destroy()
            self._place_thumb(url, tkimg, target_frame=target_frame)
        except Exception:
            self.queue.put(("imgfail", url))

    def _place_thumb(self, url: str, tkimg: ImageTk.PhotoImage, target_frame=None):
        parent = target_frame if target_frame is not None else self.images_container
        lbl = ttk.Label(parent, image=tkimg)
        lbl.image = tkimg
        lbl.pack(side=tk.LEFT, padx=6, pady=4)
        lbl.bind("<Button-1>", lambda e, u=url: webbrowser.open(u))

    def play_ding(self):
        """Play search start sound"""
        try:
            fs = 16000
            duration = 0.17
            t = np.linspace(0, duration, int(fs * duration), endpoint=False)
            freq = 988
            beep = 0.3 * np.sin(2 * np.pi * freq * t)
            fade_samples = int(0.01 * fs)
            beep[:fade_samples] *= np.linspace(0, 1, fade_samples)
            beep[-fade_samples:] *= np.linspace(1, 0, fade_samples)
            sd.play(beep, fs, blocking=False)
        except Exception as e:
            print(f"[ding] error: {e}")

    def play_dong(self):
        """Play search complete sound"""
        try:
            fs = 16000
            duration = 0.26
            t = np.linspace(0, duration, int(fs * duration), endpoint=False)
            freq = 659
            beep = 0.3 * np.sin(2 * np.pi * freq * t)
            fade_samples = int(0.01 * fs)
            beep[:fade_samples] *= np.linspace(0, 1, fade_samples)
            beep[-fade_samples:] *= np.linspace(1, 0, fade_samples)
            sd.play(beep, fs, blocking=False)
        except Exception as e:
            print(f"[dong] error: {e}")

    def normalize_query(self, q: str) -> str:
        """Add date context to time-related queries"""
        ql = q.lower()
        now = datetime.now()
        if "today" in ql:
            q += " " + now.strftime("%Y-%m-%d")
        if "yesterday" in ql:
            q += " " + (now - timedelta(days=1)).strftime("%Y-%m-%d")
        if "this week" in ql:
            q += " " + now.strftime("week %G-W%V")
        return q

    def destroy(self):
        """Proper cleanup when window is closed"""
        self.in_progress = False
        # Close LaTeX window if open
        if self.latex_win and self.latex_win.winfo_exists():
            try:
                self.latex_win.destroy()
            except:
                pass
        super().destroy()


# === END WEB SEARCH WINDOW ===
#

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
            "You are the vision AI assistant. You CAN see the provided image(s). "
            "When describing what you see, begin with 'I am the vision AI, I can see the following: ' "
            "and then provide a clear, detailed description. "
            "Answer using only what is visible in the image plus the user question. "
            "If asked to count, return a number. If asked to identify, name the most likely class. "
            "Do not say you are text-based."
        )
        self.vl_model = (
                self.cfg.get("vl_model")
                or self.cfg.get("vl_model_path")
                or "qwen2.5-vl:7b"
        )

        # === ADD SLEEP VARIABLES RIGHT HERE ===
        self.sleep_mode = False
        self.sleep_commands = ["sleep", "go to sleep", "rest mode", "silence", "stop listening"]
        self.wake_commands = ["wake", "wake up", "awaken", "resume", "start listening", "listen again"]

        self.master = master
        master.title("Always Listening — Qwen (local)")
        master.geometry("1080x600")

        # Ensure output dir exists
        os.makedirs("out", exist_ok=True)
        purge_temp_images("out")
        # === INITIALIZE ALL TKINTER VARIABLES HERE ===
        self.tts_engine = tk.StringVar(value="sapi5")
        self.speech_rate_var = tk.IntVar(value=0)
        self.sapi_voice_var = tk.StringVar()
        self.echo_enabled_var = tk.BooleanVar(value=False)
        self.ducking_enable = tk.BooleanVar(value=True)
        self.duck_db = tk.DoubleVar(value=12.0)
        self.duck_thresh = tk.DoubleVar(value=1400.0)
        self.duck_attack = tk.IntVar(value=50)
        self.duck_release = tk.IntVar(value=250)
        self.duck_var = tk.DoubleVar(value=100.0)
        self.rms_var = tk.StringVar(value="RMS: 0")
        self.state = tk.StringVar(value="idle")
        self.device_idx = tk.StringVar()
        self.out_device_idx = tk.StringVar()
        self.duplex_mode = tk.StringVar(value="Half-duplex")
        self.latex_auto = tk.BooleanVar(value=True)
        self.latex_append_mode = tk.BooleanVar(value=False)
        self.speak_math_var = tk.BooleanVar(value=True)
        self.avatar_kind = tk.StringVar(value="Rings")

        self._last_search_query = ""
        self.search_win = None

        # External light window
        self.external_light_win = None

        # === SEARCH PROGRESS VARIABLES ===
        self._search_in_progress = False
        self._search_progress_timer = None
        self._search_progress_count = 0
        self._last_search_progress_time = 0
        # === END SEARCH PROGRESS VARIABLES ===

        # === NEW: Unified playback fencing ===
        self._play_lock = threading.Lock()
        self._play_token = 0
        # === Append Mode ====
        self.latex_append_mode = tk.BooleanVar(value=False)

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

        self._current_speaker = None  # Track which AI is currently speaking
        self._speaker_lock = threading.Lock()

        self._duck_log = bool(self.cfg.get("duck_log", False))
        self._chime_played = False
        self._last_chime_ts = 0.0
        self._beep_once_guard = False

        # === ADD SPEECH RATE VARIABLE HERE ===
        self.speech_rate_var = tk.IntVar(value=0)

        self._duck_gain = 1.0
        self._duck_active = False

        # === NEW: Vision state initialization ===
        self._last_image_path = None
        self._last_vision_reply = ""
        self._last_was_vision = False
        self._vision_context_until = 0.0
        self._vision_turns_left = 0

        # Muting control
        self.text_ai_muted = False
        self.vision_ai_muted = False

        self._mute_lock = threading.Lock()
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
        self.reset_btn.grid(row=2, column=6, padx=6, sticky="w")

        self.start_btn.grid(row=0, column=1, padx=6)
        self.stop_btn.grid(row=0, column=2, padx=6)

        # External light indicator
        self.external_light_btn = ttk.Button(top, text="External Light", command=self.toggle_external_light)
        self.external_light_btn.grid(row=2, column=5, padx=6)  # Adjust column as needed


        # STOP SPEAKING button
        self.stop_speech_btn = ttk.Button(top, text="Stop Speaking", command=self.stop_speaking)
        self.stop_speech_btn.grid(row=0, column=3, padx=6)
        # === MUTE CONTROLS ===
        # Add this after your existing buttons in the top row
        mute_frame = ttk.Frame(top)
        mute_frame.grid(row=0, column=14, padx=6, sticky="w")  # Adjust column number as needed

        # Label above the buttons
        ttk.Label(mute_frame, text="Mute:", font=("Segoe UI", 9)).pack(anchor="w")

        # Close Windows button -
        ttk.Button(top, text="Close Windows", command=self.close_all_windows).grid(row=2, column=9, padx=6)

        # Button container
        mute_buttons_frame = ttk.Frame(mute_frame)
        mute_buttons_frame.pack(fill="x", pady=(2, 0))

        # Text AI mute button
        self.text_mute_btn = ttk.Button(
            mute_buttons_frame,
            text="🔇 Text",
            width=8,
            command=self.toggle_text_ai_mute
        )
        self.text_mute_btn.pack(side="left", padx=(0, 3))

        # Vision AI mute button
        self.vision_mute_btn = ttk.Button(
            mute_buttons_frame,
            text="🔇 Vision",
            width=8,
            command=self.toggle_vision_ai_mute
        )
        self.vision_mute_btn.pack(side="left")

        # Initialize button states
        self.update_mute_buttons()

        # Echo controls
        ttk.Checkbutton(
            top, text="Echo ON", variable=self.echo_enabled_var,
            command=self._sync_echo_state  # Changed from lambda
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

        # === Latex append controls ===
        ttk.Checkbutton(top, text="Append LaTeX", variable=self.latex_append_mode).grid(row=2, column=7, padx=6)
        ttk.Button(top, text="Clear LaTeX", command=self.clear_latex).grid(row=2, column=8, padx=5)

        # Checkbox for speak_math
        self.speak_math_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(top, text="Speak Math", variable=self.speak_math_var,
                        command=self.update_speak_math_setting).grid(row=0, column=12, padx=6)

        # Avatar
        self.avatar_win = None
        self.avatar_kind = tk.StringVar(value="Rings")
        _avatar_bar = ttk.Frame(top)
        _avatar_bar.grid(row=0, column=11, padx=6, sticky="n")
        ttk.Label(_avatar_bar, text="Avatar").pack(anchor="n")
        self.avatar_combo = ttk.Combobox(
            _avatar_bar, textvariable=self.avatar_kind, state="readonly",
            width=14, values=["Rings", "Rectangles", "Rectangles 2", "Radial Pulse"]

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

        # SAPI voices
        self.sapi_voice_var = tk.StringVar()
        try:
            import pyttsx3
            eng = pyttsx3.init()
            voices = [f"{v.id} | {v.name}" for v in eng.getProperty("voices")]
        except Exception:
            voices = ["(no SAPI5 voices)"]
        ttk.Label(self.master, text="SAPI Voice:").grid(row=5, column=0, sticky="e")
        self.sapi_combo = ttk.Combobox(self.master, textvariable=self.sapi_voice_var, values=voices, width=60)
        self.sapi_combo.current(0)
        self.sapi_combo.grid(row=5, column=1, columnspan=5, sticky="we", padx=6, pady=4)

        # === ADD SPEECH SPEED CONTROL HERE ===
        ttk.Label(self.master, text="Speech Speed:").grid(row=6, column=0, sticky="e", pady=4)

        # Slider for precise control
        self.speech_rate_var = tk.IntVar(value=5)
        rate_slider = ttk.Scale(
            self.master,
            from_=-10,
            to=10,
            variable=self.speech_rate_var,
            orient="horizontal",
            length=180
        )
        rate_slider.grid(row=6, column=1, columnspan=3, sticky="we", padx=6, pady=4)

        # Current value display
        self.rate_value_label = ttk.Label(self.master, text="Fast")
        self.rate_value_label.grid(row=6, column=4, sticky="w", padx=5)

        # Preset buttons
        preset_frame = ttk.Frame(self.master)
        preset_frame.grid(row=7, column=1, columnspan=4, sticky="w", pady=2)

        ttk.Button(preset_frame, text="Slow", width=6,
                   command=lambda: self.set_speech_rate(-5)).pack(side="left", padx=2)
        ttk.Button(preset_frame, text="Normal", width=6,
                   command=lambda: self.set_speech_rate(0)).pack(side="left", padx=2)
        ttk.Button(preset_frame, text="Fast", width=6,
                   command=lambda: self.set_speech_rate(5)).pack(side="left", padx=2)
        ttk.Button(preset_frame, text="Reset", width=6,
                   command=lambda: self.set_speech_rate(5)).pack(side="left", padx=2)

        # Initialize the display for fast speed
        self.update_rate_display()



        # Preset buttons
        preset_frame = ttk.Frame(self.master)
        preset_frame.grid(row=7, column=1, columnspan=4, sticky="w", pady=2)

        ttk.Button(preset_frame, text="Slow", width=6,
                   command=lambda: self.set_speech_rate(-5)).pack(side="left", padx=2)
        ttk.Button(preset_frame, text="Normal", width=6,
                   command=lambda: self.set_speech_rate(0)).pack(side="left", padx=2)
        ttk.Button(preset_frame, text="Fast", width=6,
                   command=lambda: self.set_speech_rate(5)).pack(side="left", padx=2)
        ttk.Button(preset_frame, text="Reset", width=6,
                   command=lambda: self.set_speech_rate(0)).pack(side="left", padx=2)

        # Text input - KEEP YOUR EXISTING SECTION BUT CHANGE ROW FROM 6 TO 8
        ttk.Label(self.master, text="Text input:").grid(row=8, column=0, sticky="ne", padx=(6, 0), pady=(0, 6))
        self.text_box = ScrolledText(self.master, width=70, height=10, wrap="word")
        self.text_box.grid(row=8, column=1, columnspan=8, sticky="we", padx=6, pady=(0, 6))
        ttk.Button(self.master, text="Send", command=self.send_text).grid(row=8, column=9, sticky="nw", padx=6,
                                                                          pady=(0, 6))
        self.text_box.bind("<Control-Return>", lambda e: (self.send_text(), "break"))

        # Log -
        ttk.Label(self.master, text="Log:").grid(row=9, column=0, sticky="nw", padx=6)
        self.log = tk.Text(self.master, height=12, width=80)
        self.log.grid(row=9, column=1, columnspan=9, sticky="nsew", padx=6)
        self.master.grid_rowconfigure(9, weight=1)  # CHANGE FROM 7 TO 9
        self.master.grid_columnconfigure(9, weight=1)
        self.master.grid_columnconfigure(1, weight=1)
        self.master.grid_columnconfigure(5, weight=1)

        # MULTIPLE LaTeX windows for different contexts
        self.latex_win_text = None  # Main text AI
        self.latex_win_vision = None  # Vision AI
        self.latex_win_search = None  # Search results
        self.latex_win_weather = None  # Weather/other auto-searches

        self._current_latex_context = "text"  # Track which window is active

        # Create the main text window (keep existing behavior)
        DEFAULT_TEXT_PT = int(self.cfg.get("latex_text_pt", 12))
        DEFAULT_MATH_PT = int(self.cfg.get("latex_math_pt", 8))
        DEFAULT_TEXT_FAMILY = self.cfg.get("latex_text_family", "Segoe UI")

        DEFAULT_TEXT_PT = int(self.cfg.get("latex_text_pt", 12))
        DEFAULT_MATH_PT = int(self.cfg.get("latex_math_pt", 8))
        DEFAULT_TEXT_FAMILY = self.cfg.get("latex_text_family", "Segoe UI")
        self.latex_win_text = LatexWindow(
            self.master,
            log_fn=self.logln,
            text_family=DEFAULT_TEXT_FAMILY,
            text_size=DEFAULT_TEXT_PT,
            math_pt=DEFAULT_MATH_PT
        )
        self.latex_win_text.title("Text AI - LaTeX Preview")  # Rename it
        # TEMPORARY FIX: Keep old reference
        self.latex_win = self.latex_win_text
        if self.latex_auto.get():
            self.latex_win_text.show()

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
        # ===  CRITICAL CONNECTION LINES ===
        # Connect main app to QwenLLM
        self.qwen.set_main_app(self)
        self.logln(
            f"[DEBUG] QwenLLM main_app connected: {hasattr(self.qwen, 'main_app') and self.qwen.main_app is not None}")

        # Connect search handler
        if hasattr(self.qwen, 'set_search_handler'):
            self.qwen.set_search_handler(self.handle_ai_search_request)
            self.logln("[DEBUG] ✅ Search handler connected to QwenLLM")
        else:
            self.logln("[DEBUG] ❌ QwenLLM missing set_search_handler method")
        # === END CRITICAL CONNECTION LINES ===

        # === MODIFIED SYSTEM PROMPT - LESS REPETITIVE ===
        # System prompt with REAL-TIME date awareness
        sys_prompt = (
                self.cfg.get("system_prompt")
                or self.cfg.get("qwen_system_prompt")
                or ""
        )

        # Create a dynamic system prompt that explains the AI has real-time awareness
        from datetime import datetime
        current_datetime = datetime.now()
        current_date = current_datetime.strftime("%B %d, %Y")
        current_time = current_datetime.strftime("%I:%M %p")
        current_day = current_datetime.strftime("%A")

        enhanced_system_prompt = f"""{sys_prompt}
TTS SPEAKING GUIDELINES FOR MATHEMATICS:
- When transitioning from regular text to equations, add a brief pause
- After reading equations, add a slight pause before continuing with text
- Speak mathematical expressions clearly and deliberately
- For complex equations, break them into logical parts
- Use phrases like "the equation shows..." or "mathematically speaking..." to introduce formulas
- when reading a list of instructions or solutions that are numbered, put a pause between the number and the explanation.
EXAMPLE: 
1.We do this...  Put a pause after the 1. Make it sound like 1, ...
2.Do that... put a pause after the 2
3.Somethings else...put a pause after the 3
Put pauses in such instances after the numbers.

ANOTHER EXAMPLE:
Instead of: "The solution is x=5 and then we continue"
Use: "The solution is... x equals five... and then we continue"

Instead of: "We have f(x)=x²+2x+1 which is a parabola"  
Use: "We have the function... f of x equals x squared plus two x plus one... which describes a parabola"

    BACKGROUND KNOWLEDGE: You have access to real-time information. The current date is {current_date} and the time is {current_time}.
CURRENT REAL DATE: {current_day}, {current_date} at {current_time}
*** USE THIS EXACT DATE - DO NOT CALCULATE OR GUESS ***

CRITICAL RULES:
1. When asked about the date, ALWAYS use: "{current_day}, {current_date}"
2. Never calculate days of the week from dates yourself
3. If your internal knowledge conflicts with the date above, TRUST THE DATE PROVIDED
4. October 22, 2025 is {current_day} - your training data may be incorrect for this specific date



USE THIS INFORMATION WHEN:
- Specifically asked about date, time, or scheduling
- Questions require current time context
- Making time-sensitive calculations

DO NOT:
- Calculate or guess days of the week
- Use your internal calendar knowledge for date questions
- Second-guess the provided current date
    USE THIS INFORMATION WHEN:
    - Specifically asked about date, time, or scheduling
    - Questions require current time context
    - Making time-sensitive calculations

    DO NOT:
    - Include date/time in every response unless specifically relevant
    - Repeat "Today is [date]" in casual conversation
    - Mention your access to real-time info unless asked

    For casual conversation, respond naturally without unnecessary date/time mentions.
    """

        self.qwen.system_prompt = enhanced_system_prompt
        self.logln(f"[qwen] ✅ System prompt updated with balanced time awareness")

        self.logln(f"[qwen] 📅 Current date in system: {current_date} at {current_time}")

    def _apply_config_defaults(self):
        """Apply configuration defaults to UI"""
        try:
            if self.cfg.get("voice"):
                # No edge voice to set anymore
                pass
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

            # === ADD VISION STATE INITIALIZATION HERE ===
            self._vision_turns_left = int(self.cfg.get("vision_followup_max_turns", 5))
            self._vision_context_until = 0.0
            self._last_was_vision = False
            self.logln(f"[vision] initialized: max_turns={self._vision_turns_left}")

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


    def update_speak_math_setting(self):
        """Update the speak math setting - can be called when the checkbox changes"""
        self.logln(f"[math] Speak math: {self.speak_math_var.get()}")

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
            # New image = fresh context with full turns
            max_turns = int(self.cfg.get("vision_followup_max_turns", 3))
            self._vision_turns_left = max_turns
            self._vision_context_until = _t.monotonic() + 300  # 5 minutes
            self._last_was_vision = True
            self.logln(f"[vision] new image context: turns={self._vision_turns_left}, window=300s")
            return

        if used_turn and self._vision_turns_left > 0:
            self._vision_turns_left -= 1
            # Always extend context window when using a turn
            self._vision_context_until = _t.monotonic() + 300
            self.logln(f"[vision] used one turn: {self._vision_turns_left} remaining")

    def _should_use_vision_followup(self, text: str) -> bool:
        """Decide if a text turn should reuse the last image."""
        import time as _t

        t = (text or "").strip().lower()
        if not t:
            return False

        now = _t.monotonic()
        has_img = bool(self._last_image_path and os.path.exists(self._last_image_path))

        # Check if we have an active vision context
        has_context = has_img and (now <= float(self._vision_context_until or 0))
        has_turns = has_img and (int(self._vision_turns_left or 0) > 0)

        # Detailed logging
        self.logln(
            f"[vision-route] has_img={has_img}, has_context={has_context}, has_turns={has_turns}, turns_left={self._vision_turns_left}")
        self.logln(
            f"[vision-route] context_until={self._vision_context_until}, now={now}, remaining={self._vision_context_until - now if self._vision_context_until else 0}")

        # If we have an image but context expired, give one more turn
        if has_img and not has_context and self._last_was_vision:
            self.logln("[vision-route] context expired but giving grace turn")
            return True

        if not (has_context and has_turns):
            self.logln(f"[vision-route] context expired: has_context={has_context}, has_turns={has_turns}")
            return False

        # If we have context and turns, be more permissive about what goes to vision
        if self._last_was_vision and len(t) < 100:  # Short follow-ups after vision
            self.logln("[vision-route] short follow-up after vision -> use vision")
            return True

        # Explicit vision cues
        vision_cues = [
            "image", "picture", "photo", "this", "that", "it", "they",
            "what color", "how many", "count", "describe", "explain",
            "on the left", "on the right", "in the middle", "in the background",
            "do you see", "can you see", "find", "locate", "identify"
        ]

        cue_hit = any(cue in t for cue in vision_cues)
        self.logln(f"[vision-route] vision cues matched: {cue_hit}")

        return cue_hit

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

        # === COMMAND ROUTING SHOULD BE FIRST ===
        # Route camera/image commands first - BEFORE any AI processing
        if self._route_command(text):
            self.logln(f"[DEBUG] Command routed, returning early")
            return

        self._sync_image_context_from_window()

        # Route the user question to appropriate model
        try:
            use_vision = self._should_use_vision_followup(text)
            self.logln(f"[DEBUG] use_vision: {use_vision}")

            if use_vision:
                self.logln(f"[vision] follow-up → reuse last image (turns left: {self._vision_turns_left})")
                reply = self._ollama_generate_with_retry(text, images=[self._last_image_path])
                # Only decrement AFTER successful generation
                self._update_vision_state(used_turn=True)
                self._last_was_vision = True
                # PREVIEW VISION RESPONSE
                self.preview_latex(reply, context="vision")
            else:
                # normal text model path - use search-enhanced generation
                if hasattr(self.qwen, 'generate_with_search'):
                    reply = self.qwen.generate_with_search(text)
                else:
                    reply = self.qwen.generate(text)

                # PREVIEW TEXT RESPONSE
                self.preview_latex(reply, context="text")

                # Only reset if we're definitely not in a vision context
                if not self._should_use_vision_followup("dummy"):
                    self._update_vision_state(reset=True)

        except Exception as e:
            self.logln(f"[llm/vision] {e}\n[hint] Is Ollama running?  ollama serve")
            self.set_light("idle")
            return

        self.logln(f"[qwen] {reply}")

        self._set_last_vision_reply(reply)
        clean = clean_for_tts(reply, speak_math=self.speak_math_var.get())

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
                # Use the appropriate window for highlighting
                if use_vision:
                    target_win = self.ensure_latex_window("vision")
                else:
                    target_win = self.ensure_latex_window("text")

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

            # end query

    def mute_text_ai(self):
        """Mute Text AI audio output - prevents Text AI TTS"""
        with self._mute_lock:
            self.text_ai_muted = True
            self.logln("[mute] 🔇 Text AI audio muted - will not speak")
            self.update_mute_buttons()

    def unmute_text_ai(self):
        """Unmute Text AI audio output - allows Text AI TTS"""
        with self._mute_lock:
            self.text_ai_muted = False
            self.logln("[mute] 🔊 Text AI audio unmuted - can speak again")
            self.update_mute_buttons()

    def mute_vision_ai(self):
        """Mute Vision AI audio output - prevents Vision AI TTS"""
        with self._mute_lock:
            self.vision_ai_muted = True
            self.logln("[mute] 🔇 Vision AI audio muted - will not speak")
            self.update_mute_buttons()

    def unmute_vision_ai(self):
        """Unmute Vision AI audio output - allows Vision AI TTS"""
        with self._mute_lock:
            self.vision_ai_muted = False
            self.logln("[mute] 🔊 Vision AI audio unmuted - can speak again")
            self.update_mute_buttons()

    def toggle_text_ai_mute(self):
        """Toggle Text AI mute state - button command"""
        with self._mute_lock:
            self.text_ai_muted = not self.text_ai_muted
            state = "muted" if self.text_ai_muted else "unmuted"
            self.logln(f"[mute] {'🔇' if self.text_ai_muted else '🔊'} Text AI {state}")
            self.update_mute_buttons()

    def update_mute_buttons(self):
        """Update mute button appearance based on current mute state"""
        try:
            # Update Text AI button - shows OPPOSITE state (click mute icon to mute)
            if self.text_ai_muted:
                self.text_mute_btn.config(text="🔊 Text")  # Shows UNMUTE symbol (currently muted)
            else:
                self.text_mute_btn.config(text="🔇 Text")  # Shows MUTE symbol (currently unmuted)

            # Update Vision AI button - shows OPPOSITE state
            if self.vision_ai_muted:
                self.vision_mute_btn.config(text="🔊 Vision")  # Shows UNMUTE symbol (currently muted)
            else:
                self.vision_mute_btn.config(text="🔇 Vision")  # Shows MUTE symbol (currently unmuted)

        except Exception as e:
            # Fail silently if buttons don't exist yet (during initialization)
            pass

    def temporary_mute_for_speech(self, speaking_ai):
        """
        Temporarily mute the other AI only during speech - IMPROVED VERSION
        """
        with self._speaker_lock:
            # Store who is currently speaking
            self._current_speaker = speaking_ai

            # Use a small delay to ensure the mute happens before speech starts
            def apply_mute():
                with self._speaker_lock:
                    # Mute the opposite AI
                    if speaking_ai == "text":
                        if not self.vision_ai_muted:
                            self.vision_ai_muted = True
                            self.logln(f"[mute] 🔇 Temporarily muted Vision AI (Text AI is speaking)")
                            self.update_mute_buttons()
                    elif speaking_ai == "vision":
                        if not self.text_ai_muted:
                            self.text_ai_muted = True
                            self.logln(f"[mute] 🔇 Temporarily muted Text AI (Vision AI is speaking)")
                            self.update_mute_buttons()

            # Apply the mute after a brief delay to ensure it happens before audio playback
            self.master.after(50, apply_mute)

    def unmute_after_speech(self):
        """
        Unmute the other AI after speech finishes - IMPROVED VERSION
        """

        def apply_unmute():
            with self._speaker_lock:
                if self._current_speaker:
                    # Unmute the opposite AI
                    if self._current_speaker == "text":
                        if self.vision_ai_muted:
                            self.vision_ai_muted = False
                            self.logln(f"[mute] 🔊 Unmuted Vision AI (Text AI finished speaking)")
                    elif self._current_speaker == "vision":
                        if self.text_ai_muted:
                            self.text_ai_muted = False
                            self.logln(f"[mute] 🔊 Unmuted Text AI (Vision AI finished speaking)")

                    self._current_speaker = None
                    self.update_mute_buttons()

        # Apply the unmute after a brief delay to ensure speech is completely finished
        self.master.after(100, apply_unmute)

    def toggle_vision_ai_mute(self):
        """Toggle Vision AI mute state - button command"""
        with self._mute_lock:
            self.vision_ai_muted = not self.vision_ai_muted
            state = "muted" if self.vision_ai_muted else "unmuted"
            self.logln(f"[mute] {'🔇' if self.vision_ai_muted else '🔊'} Vision AI {state}")
            self.update_mute_buttons()

    def clear_latex(self):
        """Clear all LaTeX windows - called by Clear LaTeX button"""
        try:
            # Clear the main text window
            if self.latex_win_text and self.latex_win_text.winfo_exists():
                self.latex_win_text.clear()
                self.logln("[latex] Text window cleared")

            # Clear vision window if it exists
            if self.latex_win_vision and self.latex_win_vision.winfo_exists():
                self.latex_win_vision.clear()
                self.logln("[latex] Vision window cleared")

            # Clear search window if it exists
            if self.latex_win_search and self.latex_win_search.winfo_exists():
                self.latex_win_search.clear()
                self.logln("[latex] Search window cleared")

        except Exception as e:
            self.logln(f"[latex] Clear error: {e}")

    # === FIXED: ask_vision ===
    def ask_vision(self, image_path: str, prompt: str):
        """Called by ImageWindow when the user presses 'Ask model'."""

        # remember the most recent image explicitly
        self._last_image_path = image_path
        self._update_vision_state(new_image=True)

        def _worker():
            try:
                self.logln(f"[vision] {os.path.basename(image_path)} | prompt: {prompt}")
                self.preview_latex(prompt, context="vision")
                reply = self._ollama_generate_with_retry(prompt, images=[image_path])

                # FIXED: Use unified state management
                self._update_vision_state(used_turn=True)

                self.logln(f"[qwen] {reply}")
                self.preview_latex(reply, context="vision")

                # CRITICAL FIX: Store the reply IMMEDIATELY after generation
                self._set_last_vision_reply(reply, source="ask_vision")

                clean = clean_for_tts(reply, speak_math=self.speak_math_var.get())

                with self._play_lock:
                    self._play_token += 1
                    my_token = self._play_token
                    self.interrupt_flag = False
                    self.speaking_flag = True

                self.set_light("speaking")

                self._latex_theme("vision")

                try:
                    if self.synthesize_to_wav(clean, self.cfg["out_wav"], role="vision"):
                        # Use VISION window for highlighting
                        vision_win = self.ensure_latex_window("vision")
                        self.master.after(0, vision_win._prepare_word_spans)
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

    def handle_ai_search_request(self, search_query: str) -> str:
        """Handle search requests from the AI using the existing web search system"""
        self._last_search_query = search_query
        self.logln(f"[AI Search] Query: {search_query}")

        # === ADD THIS LINE: START PROGRESS INDICATOR ===
        self.start_search_progress_indicator()

        try:
            # Use your existing brave_search method WITH THE ORIGINAL QUERY
            results = self.brave_search(search_query, 6)
            search_summary = f"Search results for: {search_query}\n\n"

            # Process results using your existing methods
            for i, item in enumerate(results, 1):
                try:
                    html = self.polite_fetch(item.url)
                    if html:
                        # Use your existing text extraction
                        text = self.extract_readable(html, item.url)
                        if len(text) > 400:  # Only summarize if we got substantial content
                            # USE THE ENHANCED SUMMARIZATION (NOW WITH QUERY CONTEXT)
                            summary = self.summarise_for_ai_search(text[:12000], item.url, None)
                            search_summary += f"## Result {i}: {item.title}\n"
                            search_summary += f"URL: {item.url}\n"
                            search_summary += f"Summary: {summary}\n\n"
                        else:
                            search_summary += f"## Result {i}: {item.title}\n"
                            search_summary += f"URL: {item.url}\n"
                            search_summary += f"Snippet: {item.snippet}\n\n"
                    else:
                        search_summary += f"## Result {i}: {item.title}\n"
                        search_summary += f"URL: {item.url}\n"
                        search_summary += f"Snippet: {item.snippet}\n\n"
                except Exception as e:
                    search_summary += f"## Result {i}: {item.title}\n"
                    search_summary += f"URL: {item.url}\n"
                    search_summary += f"Error processing: {str(e)}\n\n"
                    continue

            # ===  STOP PROGRESS INDICATOR ON SUCCESSFUL COMPLETION ===
            self.stop_search_progress_indicator()
            return search_summary

        except Exception as e:
            # ===  STOP PROGRESS INDICATOR ON ERROR ===
            self.stop_search_progress_indicator()
            return f"Search failed: {str(e)}"



    def _process_ai_response(self, response: str, from_search_method: bool = False) -> str:
        """
        Process AI response - don't remove search markers when called from generate_with_search
        """
        import re

        # Look for search commands in the response
        search_pattern = r'\[SEARCH:\s*(.*?)\]'
        searches = re.findall(search_pattern, response, re.IGNORECASE)

        if searches:
            self.logln(f"[AI] Detected {len(searches)} search request(s)")

            # Only remove search markers if NOT called from generate_with_search
            if not from_search_method:
                for search_query in searches:
                    clean_query = search_query.strip()
                    response = response.replace(f"[SEARCH: {search_query}]",
                                                f"[I'm searching for: {clean_query}]")
            else:
                # If called from generate_with_search, keep the markers so it can process them
                self.logln(f"[AI] Preserving search markers for generate_with_search: {searches}")

        return response

    # === ENHANCED: pass_vision_to_text ===
    def pass_vision_to_text(self):
        """
        If a vision reply exists, move it into the Text input box and AUTO-SEND.
        """
        try:
            vision_reply = (getattr(self, '_last_vision_reply', "") or "").strip()
            if not vision_reply:
                self.logln("[pass] nothing to pass from vision")
                return

            # Reset vision state
            self._update_vision_state(reset=True)

            # Create formatted message
            header = "The vision AI provided this information about an image:"
            formatted_message = f"{header}\n\n{vision_reply}\n\nPlease analyze this information, solve any problems mentioned, and discuss as necessary."

            # Move to text box
            self.text_box.delete("1.0", "end")
            self.text_box.insert("1.0", formatted_message)
            self.text_box.focus_set()
            self.text_box.see("end")

            self.logln(f"[pass] vision information sent to Text AI - AUTO-SENDING")

            # AUTO-SEND after a brief delay
            self.master.after(500, self.auto_send_text)

        except Exception as e:
            self.logln(f"[pass] error: {e}")

    def auto_send_text(self):
        """Automatically send the current text box content"""
        try:
            text = self.text_box.get("1.0", "end-1c").strip()
            if text:
                self.logln("[auto-send] Sending to Text AI...")
                # Clear the text box to show it's being processed
                self.text_box.delete("1.0", "end")
                # Process the query
                threading.Thread(target=self.handle_text_query, args=(text,), daemon=True).start()
            else:
                self.logln("[auto-send] No text to send")
        except Exception as e:
            self.logln(f"[auto-send] error: {e}")

    def _set_last_vision_reply(self, reply: str, source: str = "unknown"):
        """Store the most recent vision reply with better tracking."""
        try:
            # Use thread-safe assignment
            self._last_vision_reply = (reply or "").strip()

            # short preview for visibility in your Log panel
            preview = self._last_vision_reply.strip().replace("\n", " ")
            if len(preview) > 120:
                preview = preview[:117] + "..."

            # Enhanced logging with source tracking
            self.logln(f"[vision][cache] ✅ {source}: saved reply ({len(self._last_vision_reply)} chars): {preview}")

            # Debug info
            self.logln(f"[vision][cache] turns_left={self._vision_turns_left}, last_was_vision={self._last_was_vision}")

        except Exception as e:
            self.logln(f"[vision][cache] ❌ {source}: failed to store reply: {e}")

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

    def _wait_for_file_unlock(self, filepath, timeout=5.0):
        """Wait for a file to be unlocked by another process."""
        import time
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                # Try to open the file in exclusive mode
                with open(filepath, 'rb'):
                    return True  # File is unlocked
            except (IOError, OSError, PermissionError):
                # File is still locked, wait a bit
                time.sleep(0.1)
        return False  # Timeout reached

    def _force_close_file_handles(self, filepath):
        """Attempt to force close any processes using the file (Windows only)"""
        try:
            if os.name == 'nt':  # Windows
                import subprocess
                # Use handle.exe from Sysinternals to close file handles
                result = subprocess.run(['handle.exe', filepath],
                                        capture_output=True, text=True, timeout=5)
                if 'No matching handles found' not in result.stdout:
                    self.logln(f"[filelock] found handles to {filepath}, attempting to close")
                    # Could add logic to parse and close handles here if needed
            # For other OS, we rely on the wait method
        except Exception as e:
            self.logln(f"[filelock] force close attempt failed: {e}")

    def debug_vision_state(self):
        """Debug method to show current vision state"""
        import time as _t
        now = _t.monotonic()
        context_valid = now <= float(self._vision_context_until or 0)

        self.logln(f"[vision-debug]")
        self.logln(f"  last_image: {os.path.basename(self._last_image_path) if self._last_image_path else 'None'}")
        self.logln(f"  turns_left: {self._vision_turns_left}")
        self.logln(f"  context_until: {self._vision_context_until} (valid: {context_valid})")
        self.logln(f"  last_was_vision: {self._last_was_vision}")
        self.logln(f"  current_time: {now}")

    def _pass_vision_to_text_voice(self):
        """
        Voice-activated version with auto-send.
        """
        try:
            vision_reply = (getattr(self, '_last_vision_reply', "") or "").strip()
            if not vision_reply:
                self.logln("[send-to-text] No recent vision reply to send.")
                self.play_chime(freq=440, ms=200, vol=0.2)
                return

            # Reset vision state
            self._update_vision_state(reset=True)

            # Create formatted message
            header = "The vision AI provided this information about an image:"
            formatted_message = f"{header}\n\n{vision_reply}\n\nPlease analyze this information, solve any problems mentioned, and discuss as necessary."

            # Move to text box
            self.text_box.delete("1.0", "end")
            self.text_box.insert("1.0", formatted_message)
            self.text_box.focus_set()
            self.text_box.see("end")

            self.logln(f"[send-to-text] Vision information sent to Text AI - AUTO-SENDING")

            # Play confirmation tone
            self.play_chime(freq=880, ms=150, vol=0.15)

            # AUTO-SEND after a brief delay
            self.master.after(800, self.auto_send_text)  # Slightly longer delay for voice

        except Exception as e:
            self.logln(f"[send-to-text] error: {e}")

    # === Core Application Methods ===
    def ensure_latex_window(self, context="text"):
        """Get or create the appropriate LaTeX window for each context"""
        if context == "text":
            if self.latex_win_text is None or not self.latex_win_text.winfo_exists():
                DEFAULT_TEXT_PT = int(self.cfg.get("latex_text_pt", 12))
                DEFAULT_MATH_PT = int(self.cfg.get("latex_math_pt", 8))
                DEFAULT_TEXT_FAMILY = self.cfg.get("latex_text_family", "Segoe UI")

                self.latex_win_text = LatexWindow(
                    self.master, log_fn=self.logln,
                    text_family=DEFAULT_TEXT_FAMILY, text_size=DEFAULT_TEXT_PT, math_pt=DEFAULT_MATH_PT
                )
                self.latex_win_text.title("Text AI - LaTeX Preview")
            return self.latex_win_text

        elif context == "vision":
            if self.latex_win_vision is None or not self.latex_win_vision.winfo_exists():
                DEFAULT_TEXT_PT = int(self.cfg.get("latex_text_pt", 12))
                DEFAULT_MATH_PT = int(self.cfg.get("latex_math_pt", 8))
                DEFAULT_TEXT_FAMILY = self.cfg.get("latex_text_family", "Segoe UI")

                self.latex_win_vision = LatexWindow(
                    self.master, log_fn=self.logln,
                    text_family=DEFAULT_TEXT_FAMILY, text_size=DEFAULT_TEXT_PT, math_pt=DEFAULT_MATH_PT
                )
                self.latex_win_vision.title("Vision AI - LaTeX Preview")
                self.latex_win_vision.set_scheme("vision")  # Blue theme
            return self.latex_win_vision

        elif context == "search":
            if self.latex_win_search is None or not self.latex_win_search.winfo_exists():
                DEFAULT_TEXT_PT = int(self.cfg.get("latex_text_pt", 12))
                DEFAULT_MATH_PT = int(self.cfg.get("latex_math_pt", 8))
                DEFAULT_TEXT_FAMILY = self.cfg.get("latex_text_family", "Segoe UI")

                self.latex_win_search = LatexWindow(
                    self.master, log_fn=self.logln,
                    text_family=DEFAULT_TEXT_FAMILY, text_size=DEFAULT_TEXT_PT, math_pt=DEFAULT_MATH_PT
                )
                self.latex_win_search.title("Search Results - LaTeX Preview")
            return self.latex_win_search

    def start(self):
        if self.running: return
        self.running = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.set_light("idle")

        # ADD THIS LINE - sync echo state when starting
        self._sync_echo_state()

        threading.Thread(target=self.loop, daemon=True).start()

    def stop(self):
        self.running = False
        self.stop_speaking()
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.set_light("idle")

        # Close status light when stopping
        self.close_status_light()

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

    def play_search_results(self, path: str, token=None):
        """Play search results audio with proper interrupt support"""
        try:
            # Apply echo effect if enabled
            play_path = path
            if bool(self.echo_enabled_var.get()):
                try:
                    echo_path = "out/search_results_echo.wav"
                    play_path, _ = self.echo_engine.process_file(path, echo_path)
                    self.logln("[echo] processed search results -> out/search_results_echo.wav")
                except Exception as e:
                    self.logln(f"[echo] processing failed: {e} (playing dry)")

            # Use the existing playback infrastructure with token support
            with self._play_lock:
                self._play_token += 1
                my_token = self._play_token
                self.interrupt_flag = False
                self.speaking_flag = True

            self.set_light("speaking")
            self.play_wav_with_interrupt(play_path, token=my_token)

        except Exception as e:
            self.logln(f"[search][playback] Error: {e}")
        finally:
            self.speaking_flag = False
            self.interrupt_flag = False
            self.set_light("idle")

    def reset_chat(self):
        self.qwen.clear_history()
        self.qwen.system_prompt = self.cfg.get("system_prompt", "")
        self.logln("[info] Chat reset.")

    def what_do_you_see_ui(self):
        """Voice command: 'what do you see' -> open camera, take picture, and describe automatically."""
        try:
            # Ensure image window exists and is visible
            self._ensure_image_window()
            if self._img_win is None:
                self.logln("[vision] camera UI unavailable")
                return

            # Show and raise the window
            self._img_win.deiconify()
            self._img_win.lift()

            # Start camera if not already running
            if not getattr(self._img_win, '_live_mode', False):
                self._img_win.start_camera()
                self.logln("[vision] camera started for 'what do you see'")
                # Wait a moment for camera to initialize
                time.sleep(1.5)

            # Take snapshot
            saved_path = self._img_win.snapshot()
            if saved_path:
                self.logln(f"[vision] snapshot taken: {saved_path}")

                # Use the special vision prompt that includes the "I am the vision AI" prefix
                vision_prompt = (
                    "You are the vision AI. Begin your response with 'I am the vision AI, I can see the following: ' "
                    "and then describe what you see in the image in clear, detailed terms. "
                    "Focus on the main subjects, actions, and any notable details."
                )

                # Call the vision system with the special prompt
                self.ask_vision(saved_path, vision_prompt)
            else:
                self.logln("[vision] failed to take snapshot for 'what do you see'")

        except Exception as e:
            self.logln(f"[vision] 'what do you see' error: {e}")

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
            # === MODIFIED SLEEP MODE CHECK ===
            if self.sleep_mode:
                # Still process audio but ONLY check for wake commands while sleeping
                try:
                    utt = next(it)
                    # Check if we got any audio data (utt is a numpy array)
                    if utt is not None and utt.size > 0:  # ← FIXED LINE
                        text = self.asr.transcribe(utt, self.cfg["sample_rate"])

                        if text:
                            text_lower = text.lower()
                            # ONLY respond to wake commands while sleeping
                            if any(cmd in text_lower for cmd in self.wake_commands):
                                self.exit_sleep_mode()
                                # Don't process this utterance further
                                continue
                            else:
                                self.logln(f"[sleep] Ignored: '{text}'")
                                # Play gentle "I'm sleeping" beep
                                self.play_sleep_reminder_beep()
                except StopIteration:
                    break
                except Exception as e:
                    self.logln(f"[sleep] audio error: {e}")

                continue  # Skip normal processing while sleeping

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
                utt = next(it)  # ← THIS IS THE ONLY utt = next(it) CALL NOW
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

            try:
                use_vision = self._should_use_vision_followup(text)

                if use_vision:
                    self.logln(f"[vision][voice] follow-up → reuse last image (turns left: {self._vision_turns_left})")
                    reply = self._ollama_generate_with_retry(text, images=[self._last_image_path])
                    # Only decrement AFTER successful generation
                    self._update_vision_state(used_turn=True)
                    self._last_was_vision = True  # Ensure this stays True

                    # CRITICAL FIX: Store follow-up replies too
                    self._set_last_vision_reply(reply, source="voice_followup")

                else:
                    # normal text model path
                    # Use search-enhanced generation
                    reply = self.qwen.generate_with_search(text)

                    # Only reset if we're definitely not in a vision context
                    if not self._should_use_vision_followup("dummy"):  # Check if context expired
                        self._update_vision_state(reset=True)

            except Exception as e:
                self.logln(f"[llm/vision] {e}\n[hint] Is Ollama running?  ollama serve")
                self.set_light("idle")
                continue

            self.logln(f"[qwen] {reply}")
            # PREVIEW THE RESPONSE FOR VOICE QUERIES
            self.preview_latex(reply, context="text")

            clean = clean_for_tts(reply, speak_math=self.speak_math_var.get())
            self.speaking_flag = True
            self.interrupt_flag = False
            self.set_light("speaking")

            role = "vision" if use_vision else "text"

            self._latex_theme("vision" if role == "vision" else "default")

            try:
                if self.synthesize_to_wav(clean, self.cfg["out_wav"], role=role):
                    play_path = self.cfg["out_wav"]
                    if bool(self.echo_enabled_var.get()):
                        try:
                            play_path, _ = self.echo_engine.process_file(self.cfg["out_wav"],
                                                                         "out/last_reply_echo.wav")
                            self.logln("[echo] processed -> out/last_reply_echo.wav")
                        except Exception as e:
                            self.logln(f"[echo] processing failed: {e} (playing dry)")
                    self.play_wav_with_interrupt(play_path)
            finally:
                self.speaking_flag = False
                self.interrupt_flag = False
                self.set_light("idle")

                # Only reset theme if we were using vision theme
                if role == "vision":
                    self._latex_theme("default")

        self.stop()

    # === Voice/Audio Methods ===
    def start_bargein_mic(self, device_idx):
        q = deque(maxlen=64)

        def callback(indata, frames, time_info, status):
            if self.speaking_flag and self._bargein_enabled:
                try:
                    # Validate audio data before appending
                    if indata is not None and indata.size > 0:
                        # Check for NaN or infinite values
                        if not np.any(np.isnan(indata)) and not np.any(np.isinf(indata)):
                            q.append(np.copy(indata))
                except Exception as e:
                    self.logln(f"[bargein_callback] Error: {e}")

        try:
            stream = sd.InputStream(
                device=device_idx, samplerate=self.cfg["sample_rate"],
                channels=1, dtype="float32", blocksize=1024, callback=callback
            )
            stream.start()
            return stream, q
        except Exception as e:
            self.logln(f"[bargein_mic] Failed to start: {e}")
            return None, deque(maxlen=64)

    def monitor_interrupt(self):
        import numpy as _np, time as _time
        threshold_interrupt = self.cfg.get("bargein_threshold", 1500)
        trips_needed = int(self.cfg.get("barge_trips_needed", 3))
        trips = 0
        dt = 0.05

        while self.running:
            if self.speaking_flag and self.barge_buffer and len(self.barge_buffer) > 0:
                try:
                    audio = _np.concatenate(list(self.barge_buffer))
                    self.barge_buffer.clear()

                    if audio.size == 0:
                        _time.sleep(dt)
                        continue

                    # SAFE RMS CALCULATION
                    if audio.size > 0:
                        # Clip to safe range to prevent overflow
                        audio_safe = np.clip(audio, -1.0, 1.0)
                        # Calculate RMS with error handling
                        rms_squared = np.mean(audio_safe ** 2)
                        if rms_squared > 0 and not np.isnan(rms_squared) and not np.isinf(rms_squared):
                            rms = np.sqrt(rms_squared) * 32768
                        else:
                            rms = 0.0
                    else:
                        rms = 0.0

                    self._last_rms = float(rms)

                    # Rest of your existing barge-in logic...
                    speech_start_time = getattr(self, '_speech_start_time', 0)
                    is_early_speech = _time.monotonic() - speech_start_time < 1.5

                    is_vision_followup = (
                            self._last_was_vision and
                            self._vision_turns_left > 0 and
                            getattr(self, '_vision_context_until', 0) > _time.monotonic()
                    )

                    effective_threshold = threshold_interrupt
                    if rms > 800:
                        effective_threshold = max(800, threshold_interrupt - 500)

                    if self._bargein_enabled and not is_early_speech and not is_vision_followup:
                        if rms > effective_threshold:
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

                    # Ducking logic (with safe RMS)
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

                except Exception as e:
                    self.logln(f"[monitor_interrupt] Error: {e}")
                    # Continue running even if there's an error in one iteration
                    _time.sleep(dt)

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

    def play_chime(self, freq=880, ms=140, vol=0.20):
        """Play a chime with error handling to avoid file locking issues."""
        try:
            fs = 16000
            n = int(fs * (ms / 1000.0))
            t = np.linspace(0, ms / 1000.0, n, endpoint=False)
            s = np.sin(2 * np.pi * freq * t).astype(np.float32)
            fade = np.linspace(0.0, 1.0, min(16, n), dtype=np.float32)
            s[:fade.size] *= fade
            s[-fade.size:] *= fade[::-1]

            # Get output device with fallback
            try:
                out_dev = self._selected_out_device_index()
                sd.play((vol * s).reshape(-1, 1), fs, blocking=False, device=out_dev)
            except Exception as dev_error:
                # Fallback to default device
                self.logln(f"[beep] device error, using default: {dev_error}")
                sd.play((vol * s).reshape(-1, 1), fs, blocking=False)

        except Exception as e:
            self.logln(f"[beep] {e} - chime skipped")
            # Don't re-raise, this is non-critical

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

    def brief_listen_prompt(self):
        if not self.cfg.get("announce_listening", True):
            return
        prev = self.speaking_flag
        try:
            self.speaking_flag = True
            self.play_chime2("beep.mp3")
        finally:
            self.speaking_flag = prev

    def speak_search_status(self, message="Searching the internet for this information"):
        """Speak search status messages with echo support"""
        if not message or not message.strip():
            return

        self.logln(f"[search-status] 🔊 Processing: {message}")
        # ===  START AUDIBLE PROGRESS INDICATOR ===
        self.start_search_progress_indicator()

        try:
            # Clean the text for TTS
            clean_message = clean_for_tts(message, speak_math=self.speak_math_var.get())
            status_path = "out/search_status.wav"

            if self.synthesize_to_wav(clean_message, status_path, role="text"):
                def play_status():
                    try:
                        time.sleep(0.1)

                        # === APPLY ECHO IF ENABLED ===
                        play_path = status_path
                        if bool(self.echo_enabled_var.get()):
                            try:
                                echo_path = "out/search_status_echo.wav"
                                play_path, _ = self.echo_engine.process_file(status_path, echo_path)
                                self.logln("[search-status] 🔁 Echo applied to status message")
                            except Exception as e:
                                self.logln(f"[search-status] echo processing failed: {e} (playing dry)")

                        # Get the output device
                        out_dev = self._selected_out_device_index()

                        # Load and play the audio file
                        data, fs = sf.read(play_path, dtype="float32")
                        if data.size > 0:
                            sd.play(data, fs, blocking=False, device=out_dev)
                            self.logln(f"[search-status] ✅ Playing: {message}")

                    except Exception as e:
                        self.logln(f"[search-status] play error: {e}")

                threading.Thread(target=play_status, daemon=True).start()

        except Exception as e:
            self.logln(f"[search-status] synthesis error: {e}")

    def play_wav_with_interrupt(self, path, token=None):
        import platform as _plat
        start_time = time.monotonic()
        active_token = token
        # Determine which AI is speaking based on context
        if hasattr(self, '_last_was_vision') and self._last_was_vision:
            speaking_ai = "vision"
        else:
            speaking_ai = "text"

        self.temporary_mute_for_speech(speaking_ai)

        # Track speech start time for barge-in protection
        self._speech_start_time = start_time
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

                    # self._tts_silent = bool(avg_abs < SILENCE_THRESH)
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

                # self._tts_cursor_samples = int(cursor)

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
                    # self.master.after(0, _ui_progress_tick)
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
            self.unmute_after_speech()
            self.speaking_flag = False
            self.interrupt_flag = False
            self.set_light("idle")
            try:
                if self.avatar_win and self.avatar_win.winfo_exists():
                    self.avatar_win.set_level(0)
            except Exception:
                pass

            self._beep_once_guard = False
            dur = time.monotonic() - start_time
            self.logln(f"[audio] playback done ({dur:.2f}s)")

    # Route Commands
    def _route_command(self, raw_text: str) -> bool:
        """
        Handle voice/typed control phrases robustly.
        Normalizes contractions & punctuation and matches many variants.
        Returns True if a command was executed.
        """
        import time as _t

        text = (raw_text or "").strip().lower()
        # Check sleep/wake commands FIRST, before any other processing
        if any(cmd in text for cmd in self.sleep_commands) and not self.sleep_mode:
            self.enter_sleep_mode()
            return True

        if any(cmd in text for cmd in self.wake_commands) and self.sleep_mode:
            self.exit_sleep_mode()
            return True

        # If in sleep mode, ignore ALL other voice commands
        if self.sleep_mode:
            self.logln("[sleep] 💤 Ignoring voice input while sleeping")
            # Play gentle reminder beep
            try:
                fs = 16000
                t = np.linspace(0, 0.1, int(fs * 0.1))
                beep = 0.1 * np.sin(2 * np.pi * 440 * t)
                sd.play(beep, fs, blocking=False)
            except:
                pass
            return True

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

        # Mute Commands
        mute_commands = [
            "mute text", "mute text ai", "silence text", "text ai mute", "text mute",
            "mute vision", "mute vision ai", "silence vision", "vision ai mute", "vision mute",
            "unmute text", "unmute text ai", "enable text audio", "text audio on",
            "unmute vision", "unmute vision ai", "enable vision audio", "vision audio on",
            "toggle text mute", "toggle vision mute", "text mute toggle", "vision mute toggle"
        ]

        # Add these exact conditionals:
        if any(cmd in text for cmd in ["mute text", "silence text"]) and "unmute" not in text:
            self.mute_text_ai()
            return True
        elif any(cmd in text for cmd in ["unmute text", "enable text audio"]):
            self.unmute_text_ai()
            return True
        elif any(cmd in text for cmd in ["mute vision", "silence vision"]) and "unmute" not in text:
            self.mute_vision_ai()
            return True
        elif any(cmd in text for cmd in ["unmute vision", "enable vision audio"]):
            self.unmute_vision_ai()
            return True
        elif any(cmd in text for cmd in ["toggle text mute", "text mute toggle"]):
            self.toggle_text_ai_mute()
            return True
        elif any(cmd in text for cmd in ["toggle vision mute", "vision mute toggle"]):
            self.toggle_vision_ai_mute()
            return True

        # === SEARCH VOICE COMMANDS ===
        search_commands = [
            "search for", "search", "look up", "information on", "find information about",
            "web search", "search the web for", "look for", "That's for", "database", "online"
            # Removed standalone "find" - this was causing the problem
        ]

        # Improved search command detection
        for cmd in search_commands:
            if cmd in text:
                # Extract the actual query by removing the command phrase
                query = text.replace(cmd, "").strip()

                # Handle cases like "search for cats" vs "search cats"
                if not query:  # If command was at the end, try alternative parsing
                    words = text.split()
                    cmd_words = cmd.split()
                    if len(words) > len(cmd_words):
                        query = " ".join(words[len(cmd_words):])

                # === ADD DATE ENHANCEMENT FOR FLIGHT QUERIES ===
                if any(flight_word in query.lower() for flight_word in ['flight', 'fly', 'airline', 'airfare']):
                    from datetime import datetime, timedelta
                    today = datetime.now()

                    # Handle "next monday" type queries
                    if "next monday" in query.lower():
                        days_ahead = 7 - today.weekday()  # Monday is 0
                        if days_ahead <= 0:  # If today is Monday or after
                            days_ahead += 7
                        next_monday = today + timedelta(days=days_ahead)
                        query += f" {next_monday.strftime('%Y-%m-%d')}"
                        self.logln(f"[search] Enhanced flight query with date: {next_monday.strftime('%Y-%m-%d')}")

                    # Add current year if not present and looks like a date-based flight query
                    current_year = today.year
                    if str(current_year) not in query and any(word in query.lower() for word in
                                                              ['january', 'february', 'march', 'april', 'may', 'june',
                                                               'july', 'august', 'september', 'october', 'november',
                                                               'december']):
                        query += f" {current_year}"
                        self.logln(f"[search] Enhanced flight query with year: {current_year}")

                if query:
                    self.logln(f"[search] Voice search: {query}")
                    # === AI says it is search with speech, not for text only though ===
                    self.speak_search_status(f"Searching for {query}")
                    # Ensure search window exists and is visible - FIXED
                    self.toggle_search_window(ensure_visible=True)

                    # Wait a moment for window to appear, then set query and search
                    def do_voice_search():
                        try:
                            if (self.search_win and
                                    self.search_win.winfo_exists() and
                                    not self.search_win.in_progress):

                                # Clear any previous query and set new one
                                self.search_win.txt_in.delete("1.0", "end")
                                self.search_win.txt_in.insert("1.0", query)

                                # Start the search
                                self.search_win.on_go()
                            else:
                                self.logln("[search] Search window not ready for voice command")
                        except Exception as e:
                            self.logln(f"[search] Voice search error: {e}")

                    # Give the window time to appear before starting search
                    self.master.after(500, do_voice_search)
                    return True
        # End search commands
        exit_vision = [
            "new topic", "switch to text", "text mode", "forget image", "clear image", "ignore the image",
            "can I speak to Zen", "stop using the image", "no vision", "text only", "go back to text",
            "can I speak to zen", "back to text", "speak with zen", "speak to zen"
        ]

        # === NEW: Send to text AI commands ===
        send_to_text_commands = [
            "send", "send to text", "send to zen", "send information", "send that",
            "ok send that", "pass to text", "pass to zen", "transfer to text", "send to test",
            "give to zen", "forward to text", "share with zen", "send that to text",
            "send it to text", "pass it to test", "send this to text", "pass this to text", "ok push"
        ]

        stop_commands = [
            "stop", "stop speaking", "stop talking", "be quiet", "shut up",
            "enough", "that's enough", "okay stop", "ok stop", "fuck off"
        ]

        # Close windows commands
        close_windows_commands = [
            "close window", "close all windows", "hide windows", "minimize windows",
            "clean up windows", "tidy windows", "close extra windows", "clean desktop"
        ]

        debug_commands = [
            "debug vision", "vision status", "vision debug", "show vision state"
        ]

        test_vision_commands = [
            "test vision", "vision test", "check vision"
        ]

        # NEW: "what do you see" command patterns
        what_see_commands = [
            "what do you see", "what can you see", "whats happening", "what is happening",
            "whats going on", "what is going on", "describe what you see", "tell me what you see",
            "show me what you see", "what are you seeing", "whats in front of you",
            "describe the scene", "whats around you", "whats in the room"
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
            "switch to vision", "switch to image", "speak with the Guardian"
        ]

        # === DISPATCH ORDER: EXIT COMMANDS FIRST ===

        # 1. EXIT VISION COMMANDS - HIGHEST PRIORITY
        if matched(exit_vision):
            self.logln(f"[command] Exit vision command detected: '{text}'")
            self._last_was_vision = False
            self._vision_context_until = 0.0
            self._vision_turns_left = 0
            try:
                self.latex_win.set_scheme("default")
            except Exception:
                pass
            self.logln("[vision] image context cleared; back to text mode")
            return True

        # 2. SEND TO TEXT AI COMMANDS - NEW
        if matched(send_to_text_commands):
            self.logln(f"[command] Send to text command detected: '{text}'")
            self._pass_vision_to_text_voice()
            return True

        # 3. STOP COMMANDS
        if matched(stop_commands):
            self.logln("[command] Stop command detected - stopping speech")
            self.stop_speaking()
            self.set_light("idle")
            return True

        # 4. Camera control commands
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

        # 5. "What do you see" command
        if matched(what_see_commands):
            self.what_do_you_see_ui()
            self.set_light("idle")
            return True

        # 6. Explain image commands
        if matched(explain_all):
            self.explain_last_image_ui()
            self.set_light("idle")
            return True

        # 7. Vision mode commands
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
            self._vision_turns_left = int(self.cfg.get("vision_followup_max_turns", 3))
            self._vision_context_until = _t.monotonic() + 300  # 5 minutes

            try:
                self.latex_win.set_scheme("vision")
            except Exception:
                pass

            self.logln(f"[vision] takeover requested — turns={self._vision_turns_left}")
            self.set_light("idle")
            return True

        # === TEST VISION COMMANDS ===
        if matched(test_vision_commands):
            self.logln("[vision-test] Running vision system test...")
            self.debug_vision_state()
            # Test if we have a current image
            if self._last_image_path and os.path.exists(self._last_image_path):
                self.logln(f"[vision-test] Current image: {os.path.basename(self._last_image_path)}")
                self.ask_vision(self._last_image_path, "Describe this image briefly for testing.")
            else:
                self.logln("[vision-test] No current image available")
            return True

        # === DEBUG COMMANDS - PUT THIS LAST ===
        if matched(debug_commands):
            self.debug_vision_state()
            self.set_light("idle")
            return True
        # Close windows command
        if matched(close_windows_commands):
            self.logln(f"[command] Close windows command detected: '{text}'")
            self.close_all_windows()
            return True

        return False

    # routine ends here

    # === Awaken and Sleep Methods ===
    def close_all_windows(self):
        """Close all secondary windows except avatar and main window"""
        windows_closed = 0

        # Close status light
        self.close_status_light()

        # List of other windows to close
        windows_to_close = [
            'latex_win_text', 'latex_win_vision', 'latex_win_search', 'latex_win',
            'search_win', '_img_win', '_echo_win'
        ]

        for window_name in windows_to_close:
            window = getattr(self, window_name, None)
            if window and hasattr(window, 'winfo_exists') and window.winfo_exists():
                try:
                    if hasattr(window, 'hide'):
                        window.hide()
                    elif hasattr(window, 'withdraw'):
                        window.withdraw()
                    else:
                        window.iconify()
                    windows_closed += 1
                except Exception as e:
                    self.logln(f"[close] Error closing {window_name}: {e}")

        self.logln(f"[close] Closed {windows_closed} windows (avatar remains open)")
        self.play_chime(freq=660, ms=120, vol=0.15)
        return windows_closed



    def enter_sleep_mode(self):
        """Enter sleep mode - ignore all voice input"""
        if not self.sleep_mode:
            self.sleep_mode = True
            self.set_light("idle")
            self.logln("[sleep] 💤 Sleep mode activated - ignoring voice input")
            self.play_sleep_chime()
            #  Close all windows when entering sleep mode
            windows_closed = self.close_all_windows()
            # Optional: Show sleep status in UI
            try:
                self.master.title("Always Listening — Qwen (SLEEPING)")
            except:
                pass

    def exit_sleep_mode(self):
        """Exit sleep mode - resume normal operation"""
        if self.sleep_mode:
            self.sleep_mode = False
            self.set_light("listening")
            self.logln("[sleep] 🔔 Awake mode activated - listening for voice")
            self.play_wake_chime()

            # Restore normal title
            try:
                self.master.title("Always Listening — Qwen (local)")
            except:
                pass

    def play_sleep_chime(self):
        """Play sleep confirmation chime"""
        try:
            fs = 16000
            duration = 0.3
            t = np.linspace(0, duration, int(fs * duration), endpoint=False)
            # Descending tone for sleep
            freq = np.linspace(660, 220, len(t))
            beep = 0.2 * np.sin(2 * np.pi * freq * t)
            fade = int(0.02 * fs)
            beep[:fade] *= np.linspace(0, 1, fade)
            beep[-fade:] *= np.linspace(1, 0, fade)
            sd.play(beep, fs, blocking=False)
        except Exception as e:
            print(f"[sleep-chime] {e}")

    def play_sleep_reminder_beep(self):
        """Play noticeable beep to indicate sleeping mode when someone talks"""
        try:
            fs = 16000
            duration = 0.25  # Slightly longer
            t = np.linspace(0, duration, int(fs * duration), endpoint=False)

            # More noticeable tone - higher frequency and louder
            freq1 = 440  # A4 note - more noticeable
            freq2 = 330  # E4 note - for a two-tone effect

            # Create a two-tone beep for better noticeability
            beep1 = 0.3 * np.sin(2 * np.pi * freq1 * t[:len(t) // 2])
            beep2 = 0.3 * np.sin(2 * np.pi * freq2 * t[len(t) // 2:])
            beep = np.concatenate([beep1, beep2])

            # Smooth fade in/out
            fade = int(0.02 * fs)
            beep[:fade] *= np.linspace(0, 1, fade)
            beep[-fade:] *= np.linspace(1, 0, fade)

            # Get output device and play
            out_dev = self._selected_out_device_index()
            try:
                sd.play(beep, fs, blocking=False, device=out_dev)
            except Exception as dev_error:
                # Fallback to default device
                sd.play(beep, fs, blocking=False)

            self.logln("[sleep] 💤 (sleep reminder beep)")

        except Exception as e:
            self.logln(f"[sleep-reminder] error: {e}")

    def play_wake_chime(self):
        """Play wake confirmation chime"""
        try:
            fs = 16000
            duration = 0.25
            t = np.linspace(0, duration, int(fs * duration), endpoint=False)
            # Ascending tone for wake
            freq = np.linspace(220, 660, len(t))
            beep = 0.2 * np.sin(2 * np.pi * freq * t)
            fade = int(0.01 * fs)
            beep[:fade] *= np.linspace(0, 1, fade)
            beep[-fade:] *= np.linspace(1, 0, fade)
            sd.play(beep, fs, blocking=False)
        except Exception as e:
            print(f"[wake-chime] {e}")

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

        # Try multiple connection methods
        endpoints = [
            "http://127.0.0.1:11434/api/generate",
            "http://localhost:11434/api/generate"
        ]

        for endpoint in endpoints:
            try:
                self.logln(f"[vision] trying endpoint: {endpoint}")
                r = requests.post(endpoint, json=payload, timeout=60)
                r.raise_for_status()
                response = (r.json().get("response") or "").strip()
                self.logln(f"[vision] ✅ success with {endpoint}")
                return response
            except requests.exceptions.ConnectionError as e:
                self.logln(f"[vision] connection failed to {endpoint}: {e}")
                continue
            except Exception as e:
                self.logln(f"[vision] error with {endpoint}: {e}")
                continue

        # If all endpoints fail
        raise RuntimeError(f"Could not connect to Ollama. Tried: {endpoints}")

    def _ollama_generate_with_retry(self, prompt: str, images=None, max_retries=2):
        """Generate with vision model with retry logic"""
        for attempt in range(max_retries + 1):
            try:
                return self._ollama_generate(prompt, images)
            except Exception as e:
                if attempt < max_retries:
                    self.logln(f"[vision] attempt {attempt + 1} failed: {e}, retrying...")
                    time.sleep(1)  # Brief pause before retry
                else:
                    self.logln(f"[vision] all {max_retries + 1} attempts failed")
                    raise e

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
        # Track previous mode to detect transitions
        previous_mode = getattr(self, '_previous_light_mode', 'idle')

        color = {"idle": "#f1c40f", "listening": "#2ecc71", "speaking": "#e74c3c"}.get(mode, "#f1c40f")

        # Update main light ONLY if external light is not active
        if self.external_light_win is None or not self.external_light_win.winfo_exists() or self.external_light_win.state() == "withdrawn":
            self.light.itemconfig(self.circle, fill=color)
        else:
            # External light is active, keep main light black
            self.light.itemconfig(self.circle, fill="#000000")

        self.state.set(mode)

        # Update external light if it exists and is visible
        try:
            if (self.external_light_win and
                    self.external_light_win.winfo_exists() and
                    self.external_light_win.state() != "withdrawn"):
                self.external_light_win.set_light(color)
        except Exception as e:
            self.logln(f"[light] color update error: {e}")

        # Play beep when transitioning from any state to listening (green) mode
        if mode == "listening" and previous_mode != "listening":
            self.play_chime(freq=660, ms=120, vol=0.12)
            self.logln("[status] ✅ Ready to receive requests")

        # Store current mode for next comparison
        self._previous_light_mode = mode



    def toggle_external_light(self):
        """Toggle the external light window on/off"""
        try:
            if self.external_light_win is None or not self.external_light_win.winfo_exists():
                self.external_light_win = StatusLightWindow(self.master)
                # Set initial color to match current state
                current_mode = self.state.get()
                color = {"idle": "#f1c40f", "listening": "#2ecc71", "speaking": "#e74c3c"}.get(current_mode, "#f1c40f")
                self.external_light_win.set_light(color)
                self.external_light_win.show()

                # Hide the main light in the main window
                self.light.itemconfig(self.circle, fill="#000000")  # Make main light black
                self.logln("[light] Status light opened - main light hidden")
            else:
                if self.external_light_win.state() == "withdrawn":
                    self.external_light_win.show()
                    # Hide main light
                    self.light.itemconfig(self.circle, fill="#000000")
                    self.logln("[light] Status light shown - main light hidden")
                else:
                    self.external_light_win.hide()
                    # Restore main light
                    current_mode = self.state.get()
                    color = {"idle": "#f1c40f", "listening": "#2ecc71", "speaking": "#e74c3c"}.get(current_mode,
                                                                                                   "#f1c40f")
                    self.light.itemconfig(self.circle, fill=color)
                    self.logln("[light] Status light hidden - main light restored")
        except Exception as e:
            self.logln(f"[light] toggle error: {e}")


# ==== AUDIBLE PROGRESS SOUND ===
    def start_search_progress_indicator(self):
        """Start periodic progress tones for ongoing searches"""
        self._search_in_progress = True
        self._search_progress_count = 0
        self._last_search_progress_time = time.time()
        self._play_search_progress_beep()
        self._schedule_next_progress_beep()

    def stop_search_progress_indicator(self):
        """Stop the progress indicator when search completes"""
        if self._search_in_progress:
            self.logln(f"[search] Stopping progress indicator (was at {self._search_progress_count} beeps)")
        self._search_in_progress = False
        if self._search_progress_timer:
            try:
                self.master.after_cancel(self._search_progress_timer)
                self.logln("[search] Progress timer cancelled")
            except:
                pass
            self._search_progress_timer = None


    def set_external_light_color(self, color):
        """Set the color of the external light"""
        try:
            if (self.external_light_win and
                    self.external_light_win.winfo_exists() and
                    self.external_light_win.state() != "withdrawn"):
                self.external_light_win.set_color(color)
        except Exception as e:
            self.logln(f"[light] color change error: {e}")

    def close_status_light(self):
        """Close the status light window"""
        try:
            if self.status_light_win and self.status_light_win.winfo_exists():
                self.status_light_win.destroy()
                self.status_light_win = None
                self.logln("[light] Status light closed")
        except Exception as e:
            self.logln(f"[light] close error: {e}")



    def _play_search_progress_beep(self):
        """Play a progress beep with variation based on search stage"""
        if not self._search_in_progress:
            return

        # === SAFETY CHECK: Stop if too many beeps ===
        if self._search_progress_count > 50:
            self.logln("[search] Safety stop: too many progress beeps")
            self.stop_search_progress_indicator()
            return

        self._search_progress_count += 1

        # Different beep patterns based on search progress
        if self._search_progress_count == 1:
            # First beep - gentle start
            freq = 440  # A4
            duration = 0.08
            vol = 0.1
        elif self._search_progress_count <= 3:
            # Early progress - slightly higher
            freq = 523  # C5
            duration = 0.08
            vol = 0.12
        elif self._search_progress_count <= 6:
            # Middle stage - ascending pattern
            freq = 587  # D5
            duration = 0.09
            vol = 0.14
        else:
            # Extended search - more urgent but not annoying
            freq = 659  # E5
            duration = 0.1
            vol = 0.16

        try:
            fs = 16000
            t = np.linspace(0, duration, int(fs * duration), endpoint=False)

            # Create a pleasant beep with soft attack/decay
            beep = vol * np.sin(2 * np.pi * freq * t)

            # Smooth fade
            fade = int(0.015 * fs)
            if fade > 0:
                beep[:fade] *= np.linspace(0, 1, fade)
                beep[-fade:] *= np.linspace(1, 0, fade)

            # Play non-blocking
            out_dev = self._selected_out_device_index()
            try:
                sd.play(beep, fs, blocking=False, device=out_dev)
            except Exception:
                sd.play(beep, fs, blocking=False)

            self.logln(f"[search] Progress indicator #{self._search_progress_count}")

        except Exception as e:
            self.logln(f"[search] Progress beep error: {e}")


    def _schedule_next_progress_beep(self):
        """Schedule the next progress beep with variable timing"""
        if not self._search_in_progress:
            return

        # Variable timing: more frequent as search takes longer
        if self._search_progress_count <= 3:
            interval = 8000  # 8 seconds for first few beeps
        elif self._search_progress_count <= 6:
            interval = 6000  # 6 seconds for middle stage
        else:
            interval = 5000  # 5 seconds for extended searches

        self._search_progress_timer = self.master.after(interval, self._progress_beep_sequence)

    def _progress_beep_sequence(self):
        """The actual sequence called by the timer"""
        if self._search_in_progress:
            self._play_search_progress_beep()
            self._schedule_next_progress_beep()

    # === END SEARCH PROGRESS METHODS ===

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

    # Add this method to ensure echo state is consistent
    def _sync_echo_state(self):
        """Sync the echo engine state with the UI checkbox"""
        self.echo_engine.enabled = bool(self.echo_enabled_var.get())
        self.logln(f"[echo] state synced: {self.echo_engine.enabled}")

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
            elif kind == "Radial Pulse":  # ADD THIS CASE
                self.avatar_win = RadialPulseAvatar(self.master)

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

    def toggle_search_window(self, ensure_visible=False):
        """Toggle web search window - FIXED VERSION
        Args:
            ensure_visible: If True, always show the window (used for voice searches)
        """
        try:
            if self.search_win is None or not self.search_win.winfo_exists():
                self.search_win = WebSearchWindow(self.master, log_fn=self.logln)
                # Bind ALL the search methods to this app instance
                self.search_win.brave_search = self.brave_search
                self.search_win.polite_fetch = self.polite_fetch
                self.search_win.guess_pubdate = self.guess_pubdate
                self.search_win.extract_images = self.extract_images
                self.search_win.extract_readable = self.extract_readable
                self.search_win.summarise_with_qwen = self.summarise_with_qwen
                self.search_win.synthesize_search_results = self.synthesize_search_results
                self.search_win.normalize_query = self.normalize_query
                self.search_win.play_search_results = self.play_search_results
                # ===  CRITICAL CONNECTIONS ===
                self.search_win.main_app = self  # Give access to main app
                self.search_win.preview_latex = self.preview_latex  # Use main app's method
                self.search_win.ensure_latex_window = self.ensure_latex_window  # Use main app's method
                self.search_win.logln = self.logln  # Use main app's logging
            # Always show the window if ensure_visible is True (for voice searches)
            # or if we're toggling and it's currently withdrawn
            if ensure_visible or self.search_win.state() == "withdrawn":
                self.search_win.deiconify()
                self.search_win.lift()
                self.search_win.focus_set()
            else:
                # Only hide if not forced to be visible
                if not ensure_visible:
                    self.search_win.withdraw()

        except Exception as e:
            self.logln(f"[search] window error: {e}")

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

    def preview_latex(self, content: str, context="text"):
        """Preview LaTeX content with append/replace option - ENHANCED FOR SEARCH"""
        if not self.latex_auto.get():
            return

        def _go():
            try:
                latex_win = self.ensure_latex_window(context)
                latex_win.show()

                # === CHECK APPEND MODE ===
                if self.latex_append_mode.get():
                    # APPEND MODE - add to existing content
                    latex_win.append_document(content)
                    self.logln(f"[latex] 📝 Appended to {context} window")
                else:
                    # REPLACE MODE - clear and show new content (original behavior)
                    latex_win.show_document(content)
                    self.logln(f"[latex] 🔄 Showing in {context} window (replace mode)")

                self._current_latex_context = context

            except Exception as e:
                self.logln(f"[latex] preview error ({context}): {e}")

        self.master.after(0, _go)

    def preview_search_results(self, content: str):
        """Special method for search results that preserves the display"""
        self.preview_latex(content, context="search")

    def _latex_theme(self, mode: str):
        try:
            # Set theme on the appropriate window
            if mode == "vision" and hasattr(self, "latex_win_vision"):
                self.latex_win_vision.set_scheme("vision")
            elif hasattr(self, "latex_win_text"):
                self.latex_win_text.set_scheme("default")
        except Exception:
            pass

    def synthesize_to_wav(self, text, out_wav, role="text"):
        """Synthesize text to WAV with enhanced math speaking"""

        # Check mute states first
        if role == "text" and self.text_ai_muted:
            self.logln("[mute] 🔇 Text AI muted - skipping TTS synthesis completely")
            return False
        elif role == "vision" and self.vision_ai_muted:
            self.logln("[mute] 🔇 Vision AI muted - skipping TTS synthesis completely")
            return False

        # Use the toggle setting
        speak_math = self.speak_math_var.get()
        clean_text = clean_for_tts(text, speak_math=speak_math)

        # Rest of your existing synthesize_to_wav code...
        import time
        engine = self.tts_engine.get()

        # Enhanced file locking protection
        tmp = out_wav + ".tmp.wav"
        max_retries = 3
        retry_delay = 0.5

        for attempt in range(max_retries):
            try:
                # Clean up any existing temp files
                if os.path.exists(tmp):
                    try:
                        os.remove(tmp)
                    except Exception:
                        pass

                # Ensure output directory exists
                os.makedirs(os.path.dirname(out_wav), exist_ok=True)

                # Vision always uses the FIRST SAPI5 voice
                if role == "vision":
                    import pyttsx3
                    voice_id = self._sapi_default_voice_id
                    if not voice_id:
                        voice_id = self.sapi_voice_var.get().split(" | ")[0]
                    eng = pyttsx3.init()
                    eng.setProperty("voice", voice_id)
                    # === Speech TTS Speed ===
                    eng.setProperty("rate", 150 + self.speech_rate_var.get() * 10)

                    eng.save_to_file(clean_text, tmp)
                    eng.runAndWait()
                    eng.stop()

                    # Wait for file to be written
                    time.sleep(0.2)

                    if os.path.exists(tmp) and os.path.getsize(tmp) > 0:
                        if os.path.exists(out_wav):
                            try:
                                os.remove(out_wav)
                            except Exception:
                                pass
                        os.rename(tmp, out_wav)
                        self.logln(f"[tts] vision (sapi5, fixed): {voice_id}")
                        return True
                    else:
                        self.logln("[tts] temp file not created properly")
                        continue

                # Text keeps current selection
                if engine == "sapi5":
                    import pyttsx3
                    voice_id = self.sapi_voice_var.get().split(" | ")[0]
                    eng = pyttsx3.init()
                    eng.setProperty("voice", voice_id)
                    # === TTS Speed  ===
                    eng.setProperty("rate", 150 + self.speech_rate_var.get() * 10)

                    eng.save_to_file(clean_text, tmp)
                    eng.runAndWait()
                    eng.stop()

                    # Wait for file to be written
                    time.sleep(0.2)

                    if os.path.exists(tmp) and os.path.getsize(tmp) > 0:
                        if os.path.exists(out_wav):
                            try:
                                os.remove(out_wav)
                            except Exception:
                                pass
                        os.rename(tmp, out_wav)
                        self.logln(f"[tts] text (sapi5): {voice_id}")
                        return True
                    else:
                        self.logln("[tts] temp file not created properly")
                        continue

            except Exception as e:
                self.logln(f"[tts] attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    self.logln(f"[tts] all {max_retries} attempts failed")
                    # Final cleanup
                    try:
                        if os.path.exists(tmp):
                            os.remove(tmp)
                    except Exception:
                        pass
                    return False

        return False

    def set_speech_rate(self, rate: int):
        """Set speech rate to specific value"""
        self.speech_rate_var.set(rate)
        self.update_rate_display()
        self.logln(f"[tts] Speech rate set to: {rate}")

    def update_rate_display(self):
        """Update the rate display label"""
        rate = self.speech_rate_var.get()
        if hasattr(self, 'rate_value_label'):
            if rate < -5:
                self.rate_value_label.config(text="Very Slow")
            elif rate < 0:
                self.rate_value_label.config(text="Slow")
            elif rate == 0:
                self.rate_value_label.config(text="Normal")
            elif rate <= 5:
                self.rate_value_label.config(text="Fast")
            else:
                self.rate_value_label.config(text="Very Fast")

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

    # === SEARCH METHODS ===

    def brave_search(self, query: str, count: int = 6):
        brave_key = os.getenv("BRAVE_KEY")
        if not brave_key:
            raise RuntimeError("No BRAVE_KEY found in environment")
        # === Logs we are searching the Internet ===
        self.logln(f"[SEARCH] 🚀 Calling Brave API: '{query}'")

        endpoint = "https://api.search.brave.com/res/v1/web/search"
        headers = {"X-Subscription-Token": brave_key, "User-Agent": "LocalAI-ResearchBot/1.0"}
        params = {"q": query, "count": count}

        with httpx.Client(timeout=25.0, headers=headers) as client:
            r = client.get(endpoint, params=params)
            r.raise_for_status()
            data = r.json()

        out = []
        for w in (data.get("web", {}) or {}).get("results", []):
            out.append(
                Item(title=w.get("title", "No title"), url=w.get("url", ""), snippet=w.get("description", "")))
            # === check what its searching  ===
            self.logln(f"[BRAVE API] ✅ Found {len(out)} results for '{query}'")

        return out

    def polite_fetch(self, url: str):
        headers = {"User-Agent": "LocalAI-ResearchBot/1.0"}
        try:
            with httpx.Client(timeout=25.0, headers=headers, follow_redirects=True) as client:
                r = client.get(url)
                r.raise_for_status()
                return r.text
        except Exception:
            return None

    def extract_readable(self, html: str, url: str = None):
        text = trafilatura.extract(html, url=url, include_links=False, include_formatting=False)
        return text or ""

    def guess_pubdate(self, html: str):
        try:
            soup = BeautifulSoup(html, "lxml")
        except Exception:
            return None

        metas = [
            ("property", "article:published_time"), ("property", "og:published_time"),
            ("property", "og:updated_time"), ("name", "pubdate"), ("name", "publication_date"),
            ("name", "date"), ("name", "dc.date"), ("name", "dc.date.issued"),
            ("name", "sailthru.date"), ("itemprop", "datePublished"), ("itemprop", "dateModified"),
        ]

        for key, val in metas:
            tag = soup.find("meta", attrs={key: val})
            if tag and tag.get("content"):
                return tag["content"]

        t = soup.find("time")
        if t and (t.get("datetime") or (t.text and t.text.strip())):
            return t.get("datetime") or t.text.strip()
        return None



    def summarise_for_ai_search(self, text: str, url: str, pubdate: str):
        """Enhanced summarization that preserves practical information"""
        text = text[:18000]

        # Enhanced date context
        if pubdate:
            date_context = f"PUBLICATION DATE: {pubdate}\n"
        else:
            import re
            date_matches = re.findall(
                r'\b(?:20\d{2}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2} (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* 20\d{2})\b',
                text[:3000])
            date_context = f"MENTIONED DATES: {', '.join(date_matches[:3])}\n" if date_matches else ""

        # DETECT QUERY TYPE AND ADAPT SUMMARIZATION
        query_lower = getattr(self, '_last_search_query', '').lower()

        # Flight/travel related queries
        if any(keyword in query_lower for keyword in ['flight', 'fly', 'airline', 'airport', 'travel to']):
            summary_prompt = (
                "Extract COMPLETE flight information with these details:\n\n"
                "## FLIGHT INFORMATION\n"
                "- Airline names and flight numbers\n"
                "- Departure and arrival airports (with codes if available)\n"
                "- Departure and arrival times/dates\n"
                "- Flight duration\n"
                "- Prices and fare classes\n"
                "- Stopovers/layovers\n"
                "- Booking links or airline websites\n\n"
                "## TRAVEL DETAILS\n"
                "- Airport locations and terminals\n"
                "- Booking requirements\n"
                "- Baggage information\n"
                "- Recent deals or promotions\n\n"
                "Include ALL specific numbers, times, prices, and codes. Be very detailed about schedules and availability.\n"
                f"{date_context}"
                f"Source: {url}\n\nCONTENT TO SUMMARIZE:\n{text}"
            )

        # Business/location queries
        elif any(keyword in query_lower for keyword in
                 ['address', 'location', 'where is', 'hours', 'contact', 'phone', 'email']):
            summary_prompt = (
                "EXTRACT ONLY INFORMATION EXPLICITLY STATED IN THE TEXT. NEVER CREATE PLACEHOLDERS OR INVENT INFORMATION.\n\n"
                "CRITICAL RULES:\n"
                "1. ONLY include information that appears VERBATIM in the source text\n"
                "2. NEVER use brackets [ ], parentheses ( ), or placeholder text\n"
                "3. If a website is mentioned, copy the EXACT URL\n"
                "4. If information is missing, OMIT that line entirely\n"
                "5. Do NOT create template responses\n\n"
                "EXTRACTED INFORMATION (ONLY IF FOUND):\n"
                "- Business Name: [copy exact name if found]\n"
                "- Address: [copy exact address if found]\n"
                "- Phone: [copy exact phone number if found]\n"
                "- Email: [copy exact email if found]\n"
                "- Website: [copy exact URL if found]\n"
                "- Hours: [copy exact hours if found]\n\n"
                "EXAMPLES - WRONG:\n"
                "❌ Address: [Address may vary]\n"
                "❌ Phone: [Phone number may vary]  \n"
                "❌ Website: [Website Link]\n"
                "❌ Website: [Website URL if available]\n\n"
                "EXAMPLES - CORRECT:\n"
                "✅ Address: 456 Northshore Road, Unit 2, Glenfield 0678\n"
                "✅ Phone: +64 9 483 5555\n"
                "✅ Website: https://www.serenityspa.co.nz\n"
                "✅ Website: www.serenityspa.com\n"
                "✅ (omit Website line if no URL found)\n\n"
                "If the text contains '456 Glenfield Road, Unit 2, Glenfield 0678' and '+64 9 483 5555' but NO website, output:\n"
                "Address: 456 Glenfield Road, Unit 2, Glenfield 0678\n"
                "Phone: +64 9 483 5555\n\n"
                "DO NOT INVENT WEBSITE INFORMATION. If no website is found, omit the Website line completely.\n"
                f"{date_context}"
                f"Source: {url}\n\nCONTENT TO SUMMARIZE:\n{text}"
            )


        # Product/service queries
        elif any(keyword in query_lower for keyword in ['price', 'cost', 'buy', 'purchase', 'deal', 'sale']):
            summary_prompt = (
                "Extract COMPLETE product/service information:\n\n"
                "## PRICING & AVAILABILITY\n"
                "- Exact prices and currency\n"
                "- Model numbers/specifications\n"
                "- Availability status\n"
                "- Seller/retailer information\n"
                "- Shipping costs and delivery times\n"
                "- Return policies\n\n"
                "## PRODUCT DETAILS\n"
                "- Features and specifications\n"
                "- Dimensions/sizes\n"
                "- Colors/options available\n"
                "- Warranty information\n\n"
                "Include ALL pricing, specifications, and purchase details. Be very specific about numbers and options.\n"
                f"{date_context}"
                f"Source: {url}\n\nCONTENT TO SUMMARIZE:\n{text}"
            )

        # Weather queries
        elif any(keyword in query_lower for keyword in
                 ['weather', 'forecast', 'temperature', 'rain', 'snow', 'humidity']):
            summary_prompt = (
                "Extract COMPLETE weather forecast information:\n\n"
                "## CURRENT CONDITIONS\n"
                "- Temperature and feels-like temperature\n"
                "- Weather description (sunny, rainy, etc.)\n"
                "- Humidity, wind speed and direction\n"
                "- Precipitation chances\n"
                "- Air quality and UV index\n\n"
                "## FORECAST\n"
                "- Hourly and daily forecasts\n"
                "- High/low temperatures\n"
                "- Severe weather alerts\n"
                "- Sunrise/sunset times\n\n"
                "## LOCATION DETAILS\n"
                "- Specific city/region\n"
                "- Geographic details if available\n"
                "- Timezone information\n\n"
                "Include ALL numerical weather data, times, and location specifics.\n"
                f"{date_context}"
                f"Source: {url}\n\nCONTENT TO SUMMARIZE:\n{text}"
            )

        else:
            # General comprehensive summary (for news, general info, etc.)
            summary_prompt = (
                "Create a COMPREHENSIVE summary that PRESERVES practical information:\n\n"
                "## ESSENTIAL DETAILS\n"
                "- Full names of businesses, people, organizations\n"
                "- Complete addresses, phone numbers, contact information\n"
                "- Prices, costs, financial figures\n"
                "- Dates, times, schedules\n"
                "- Locations, coordinates, directions\n"
                "- Website URLs, email addresses\n\n"
                "## KEY INFORMATION\n"
                "- Main facts and findings\n"
                "- Important numbers and statistics\n"
                "- Recent developments\n"
                "- Contact methods\n\n"
                "## ADDITIONAL CONTEXT\n"
                "- Background information\n"
                "- Related services or options\n"
                "- User reviews or ratings if available\n\n"
                "CRITICAL: NEVER omit addresses, phone numbers, prices, or contact information. Include them verbatim.\n"
                f"{date_context}"
                f"Source: {url}\n\nCONTENT TO SUMMARIZE:\n{text}"
            )

        try:
            payload = {
                "model": "qwen2.5:7b-instruct",
                "prompt": summary_prompt,
                "stream": False,
                "temperature": 0.1,  # Lower temperature for more factual accuracy
                "max_tokens": 1200  # More tokens for detailed information
            }

            with httpx.Client(timeout=75.0) as client:
                r = client.post("http://localhost:11434/api/generate", json=payload)
                r.raise_for_status()
                response = r.json().get("response", "").strip()

                # Enhanced fallback for better information extraction
                if len(response) < 100 or "no information" in response.lower():
                    return self._extract_practical_information(text[:12000], query_lower)

                return response

        except Exception as e:
            return self._extract_practical_information(text[:10000], query_lower)

    def _extract_practical_information(self, text: str, query_type: str) -> str:
        """Enhanced fallback extraction focusing on practical information"""
        import re

        sections = []

        # Enhanced address extraction
        address_patterns = [
            # Standard street addresses
            r'\b\d+\s+[A-Za-z0-9\s,]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Place|Pl|Way|Highway|Hwy)\.?\s*(?:#\s*\d+)?\s*,\s*[A-Za-z\s]+,\s*[A-Z]{2}\s*\d{5}(?:-\d{4})?\b',
            # Basic address format
            r'\b\d+\s+[\w\s]+\s+(?:St|Street|Ave|Avenue|Rd|Road|Blvd|Boulevard|Dr|Drive|Ln|Lane|Ct|Court),\s*[\w\s]+,\s*[A-Z]{2}\b',
            # PO Boxes
            r'\b(?:P\.?O\.?\s*Box|PO Box|P O Box)\s+\d+[^.!?]*',
        ]

        addresses = []
        for pattern in address_patterns:
            found = re.findall(pattern, text, re.IGNORECASE)
            addresses.extend(found)

        # Filter out obviously fake or placeholder addresses
        real_addresses = []
        for addr in addresses:
            addr_lower = addr.lower()
            # Skip placeholder text
            if any(placeholder in addr_lower for placeholder in
                   ['address may vary', 'varies', 'please contact', 'call for', 'not available']):
                continue
            # Skip if it's just a city/state without street
            if re.match(r'^[A-Za-z\s]+,?\s*[A-Z]{2}\s*\d{5}', addr) and not re.search(r'\d+', addr):
                continue
            real_addresses.append(addr.strip())

        if real_addresses:
            sections.append("## ADDRESSES FOUND")
            sections.extend([f"- {addr}" for addr in set(real_addresses)[:3]])
        # Extract website URLs (more comprehensive)
        url_patterns = [
            r'https?://[^\s<>"{}|\\^`\[\]]+',
            r'www\.[^\s<>"{}|\\^`\[\]]+\.[a-z]{2,}',
            r'[a-z0-9.-]+\.[a-z]{2,}/[^\s<>"{}|\\^`\[\]]*',
        ]

        urls = []
        for pattern in url_patterns:
            urls.extend(re.findall(pattern, text, re.IGNORECASE))

        # Filter and clean URLs
        clean_urls = []
        for url in urls:
            # Remove trailing punctuation
            url = re.sub(r'[.,;:!?)]+$', '', url)
            # Skip common false positives
            if any(bad in url.lower() for bad in ['example.com', 'website.com', 'yourwebsite', 'domain.com']):
                continue
            # Ensure it looks like a real URL
            if '.' in url and len(url) > 8:
                # Add http:// if missing for www URLs
                if url.startswith('www.') and not url.startswith('http'):
                    url = 'https://' + url
                clean_urls.append(url)

        if clean_urls:
            sections.append("\n## WEBSITES")
            sections.extend([f"- {url}" for url in set(clean_urls)[:3]])

        # Extract phone numbers
        phone_pattern = r'(\+?\d{1,2}?[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})'
        phones = re.findall(phone_pattern, text)
        if phones:
            sections.append(f"\n## PHONE NUMBERS: {', '.join(set(phones)[:3])}")

        # Extract email addresses
        emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
        if emails:
            sections.append(f"\n## EMAIL ADDRESSES: {', '.join(set(emails)[:3])}")

        # Extract prices and costs
        prices = re.findall(r'\$?\d+(?:,\d+)*(?:\.\d+)?\s*(?:dollars?|USD|€|£|¥)?', text)
        if prices:
            sections.append(f"\n## PRICES MENTIONED: {', '.join(set(prices)[:8])}")

        # Flight-specific extraction
        if 'flight' in query_type:
            flight_info = re.findall(r'[A-Z]{2}\d+\s+.*?(?:\d{1,2}:\d{2}|AM|PM)', text)
            if flight_info:
                sections.append("\n## FLIGHT DETAILS")
                sections.extend([f"- {info}" for info in flight_info[:5]])

        # Business hours
        hours = re.findall(
            r'(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)[a-z]*.*?\d{1,2}:\d{2}\s*(?:AM|PM)?.*?\d{1,2}:\d{2}\s*(?:AM|PM)?', text,
            re.IGNORECASE)
        if hours:
            sections.append("\n## BUSINESS HOURS")
            sections.extend([f"- {hour}" for hour in hours[:3]])

        # Weather data extraction
        if 'weather' in query_type:
            temps = re.findall(r'\b\d{1,3}°?F?\b', text)
            if temps:
                sections.append(f"\n## TEMPERATURES: {', '.join(set(temps)[:6])}")

        # If we found practical information, return it
        if sections:
            return "\n".join(sections)
        else:
            # Return meaningful content lines as fallback
            lines = [line.strip() for line in text.split('\n') if len(line.strip()) > 30]
            return "## KEY INFORMATION EXTRACTED\n" + "\n".join([f"- {line}" for line in lines[:10]])

    def _extract_detailed_news(self, text: str) -> str:
        """Enhanced fallback extraction with more structure"""
        import re

        # Extract key information with more context
        sections = []

        # Headlines and key sentences
        sentences = re.split(r'[.!?]+', text)
        key_sentences = []

        important_indicators = [
            'announced', 'reported', 'confirmed', 'revealed', 'disclosed',
            'investigation', 'charged', 'arrested', 'settlement', 'agreement',
            'election', 'resigned', 'appointed', 'launched', 'released',
            'fire', 'accident', 'killed', 'injured', 'missing', 'found',
            'storm', 'flood', 'earthquake', 'weather', 'forecast', 'temperature',
            'budget', 'funding', 'cost', 'price', 'investment'
        ]

        for sentence in sentences:
            sentence = sentence.strip()
            if (len(sentence) > 25 and
                    any(indicator in sentence.lower() for indicator in important_indicators)):
                key_sentences.append(sentence)
                if len(key_sentences) >= 12:
                    break

        if key_sentences:
            sections.append("## KEY DEVELOPMENTS")
            sections.extend([f"- {s}" for s in key_sentences[:10]])

        # Extract numbers and statistics
        numbers = re.findall(r'\b(\$?[£€]?\d+(?:,\d+)*(?:\.\d+)?[%€£$]?(?:\s*(?:million|billion|thousand))?)\b',
                             text[:5000])
        if numbers:
            sections.append(f"\n## KEY NUMBERS: {', '.join(set(numbers[:8]))}")

        # Extract locations
        locations = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', text[:3000])
        unique_locs = list(
            set([loc for loc in locations if len(loc) > 3 and loc not in ['The', 'This', 'That', 'There', 'Here']]))
        if unique_locs:
            sections.append(f"\n## MENTIONED LOCATIONS: {', '.join(unique_locs[:6])}")

        if sections:
            return "\n".join(sections)
        else:
            # Last resort: return structured excerpt
            lines = text.split('\n')
            meaningful_lines = [line.strip() for line in lines if len(line.strip()) > 40][:8]
            return "## CONTENT OVERVIEW\n" + "\n".join([f"- {line}" for line in meaningful_lines])

    def summarise_with_qwen(self, text: str, url: str, pubdate: str):
        text = text[:20000]  # Limit text length
        pd_line = f"(Publish/Update date: {pubdate})\n" if pubdate else ""

        # FIRST PASS: Extract mathematical content specifically
        math_prompt = (
            "Extract ALL mathematical equations, formulas, and technical content from the following text. "
            "Preserve them exactly in their original LaTeX format ($$...$$, \\[...\\], $...$, etc.).\n"
            "Include:\n"
            "- All equations and formulas\n"
            "- Mathematical expressions\n"
            "- Chemical formulas\n"
            "- Code snippets\n"
            "- Important technical definitions\n"
            "Output the mathematical/technical content exactly as found, without summarization.\n"
            f"{pd_line}"
            f"Source: {url}\n\nCONTENT:\n{text[:10000]}"  # Use first 10k chars for math extraction
        )

        # SECOND PASS: Create a summary that REFERENCES the preserved math
        summary_prompt = (
            "Create a comprehensive summary (10-15 bullet points) that includes:\n"
            "- Key findings and conclusions\n"
            "- Important data points and results\n"
            "- References to mathematical content (say 'see equation X' or 'the formula shows')\n"
            "- Main arguments and evidence\n"
            "- Do NOT remove technical details - include them in context\n"
            "- Preserve specific numbers, measurements, and quantitative results\n"
            "Be detailed enough to be useful for technical analysis.\n"
            f"{pd_line}"
            f"Source: {url}\n\nCONTENT:\n{text}"
        )

        try:
            # Get mathematical content
            math_content = self.qwen.generate(math_prompt)

            # Get comprehensive summary
            summary = self.qwen.generate(summary_prompt)

            # Combine both with clear separation
            combined_result = f"MATHEMATICAL CONTENT:\n{math_content}\n\nSUMMARY:\n{summary}"

            return combined_result

        except Exception:
            # Fallback: Use a more math-friendly single prompt
            fallback_prompt = (
                "Create a DETAILED technical summary (12-18 bullet points) that PRESERVES all mathematical content.\n"
                "CRITICAL: Keep ALL equations, formulas, and LaTeX expressions exactly as they appear.\n"
                "Include:\n"
                "- Complete equations in $$...$$, \\[...\\], $...$ format\n"
                "- Mathematical proofs and derivations\n"
                "- Chemical formulas and reactions\n"
                "- Code snippets and algorithms\n"
                "- Quantitative results with exact numbers\n"
                "- Do NOT simplify or remove technical details\n"
                "- Focus on preserving the mathematical richness of the content\n"
                f"{pd_line}"
                f"Source: {url}\n\nCONTENT:\n{text}"
            )
            try:
                payload = {"model": "qwen2.5:7b-instruct", "prompt": fallback_prompt, "stream": False}
                with httpx.Client(timeout=90.0) as client:
                    r = client.post("http://localhost:11434/api/generate", json=payload)
                    r.raise_for_status()
                    return r.json().get("response", "").strip()
            except Exception as e:
                return f"Summarization failed: {e}"

    def extract_images(self, html: str, base_url: str, limit: int = 3):
        urls = []
        try:
            soup = BeautifulSoup(html, "lxml")
        except Exception:
            return urls

        for img in soup.find_all("img"):
            src = img.get("src") or ""
            if not src or src.startswith("data:") or re.search(r"\.svg($|\?)", src, re.I):
                continue

            alt = (img.get("alt") or "").lower()
            src_l = src.lower()
            if any(k in src_l for k in ["sprite", "icon", "logo", "ads", "advert", "pixel"]):
                continue
            if any(k in alt for k in ["icon", "logo"]):
                continue

            full = urljoin(base_url, src)
            if full not in urls:
                urls.append(full)
            if len(urls) >= limit:
                break
        return urls

    def synthesize_search_results(self, text: str):
        """Speak search results using DEDICATED search window"""

        # === STOP PROGRESS INDICATOR IMMEDIATELY ===
        self.stop_search_progress_indicator()

        def _tts_worker():
            if not text or not text.strip():
                return

            try:
                # Use math speaking for search results too
                speak_math = getattr(self, 'speak_math_var', tk.BooleanVar(value=True)).get()
                clean_tts_text = clean_for_tts(text, speak_math=speak_math)

                # === CRITICAL: Use DEDICATED search window ===
                self.preview_search_results(text)

                # Continue with TTS...
                output_path = "out/search_results.wav"

                if self.synthesize_to_wav(clean_tts_text, output_path, role="text"):
                    with self._play_lock:
                        self._play_token += 1
                        my_token = self._play_token
                        self.interrupt_flag = False
                        self.speaking_flag = True

                    self.set_light("speaking")

                    play_path = output_path
                    if bool(self.echo_enabled_var.get()):
                        try:
                            play_path, _ = self.echo_engine.process_file(output_path, "out/search_results_echo.wav")
                            self.logln("[echo] processed search results -> out/search_results_echo.wav")
                        except Exception as e:
                            self.logln(f"[echo] processing failed: {e} (playing dry)")

                    self.play_wav_with_interrupt(play_path, token=my_token)

            except Exception as e:
                self.logln(f"[search][TTS] Error: {e}")
            finally:
                self.speaking_flag = False
                self.interrupt_flag = False
                self.set_light("idle")

        tts_thread = threading.Thread(target=_tts_worker, daemon=True)
        tts_thread.start()


    # End syththesise_search

    def play_search_results(self, path: str, token=None):
        """Play search results audio with proper interrupt support"""
        try:
            # Use the existing playback infrastructure with token support
            with self._play_lock:
                self._play_token += 1
                my_token = self._play_token
                self.interrupt_flag = False
                self.speaking_flag = True

            self.set_light("speaking")
            self.temporary_mute_for_speech("text")  # Search uses text AI voice
            self.play_wav_with_interrupt(path, token=my_token)

        except Exception as e:
            self.logln(f"[search][playback] Error: {e}")
        finally:
            self.speaking_flag = False
            self.interrupt_flag = False
            self.set_light("idle")

    def normalize_query(self, q: str) -> str:
        """Add date context ONLY for specific time-related queries"""
        ql = q.lower()
        now = datetime.now()

        # Only add dates for explicit time references
        if "today" in ql:
            q += " " + now.strftime("%Y-%m-%d")
        elif "yesterday" in ql:
            q += " " + (now - timedelta(days=1)).strftime("%Y-%m-%d")
        elif "this week" in ql:
            q += " " + now.strftime("week %G-W%V")
        # DON'T add dates for "latest", "recent", "current" etc.

        return q


# === END SEARCH METHODS ===
# End of App
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
