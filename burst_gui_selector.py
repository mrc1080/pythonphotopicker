from __future__ import annotations

import os
import sys
import json
import shutil
import tempfile
import threading
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk, ExifTags, ImageOps
import imagehash


# =========================
# App metadata / update URL
# =========================
APP_NAME = "Photo Picker"
APP_VERSION = "1.0.6"

# Optional: host a tiny JSON file somewhere (GitHub raw is fine) like:
# {"version":"1.0.1","notes":"Fixes...","download_url":"https://.../PhotoPicker.exe"}
UPDATE_JSON_URL = "https://raw.githubusercontent.com/mrc1080/pythonphotopicker/main/update.json"

# ------------------ Supported files ------------------
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

_EXIF_TAGS = {v: k for k, v in ExifTags.TAGS.items()}
TAG_DATETIME_ORIG = _EXIF_TAGS.get("DateTimeOriginal")
TAG_DATETIME = _EXIF_TAGS.get("DateTime")


@dataclass
class PhotoItem:
    path: Path
    ts: datetime
    phash_hex: str | None
    thumb_gray: np.ndarray | None
    quality: float
    thumb_pil: Image.Image | None = None


# ------------------ Rotation sidecar ------------------
class RotationOverrides:
    """
    Viewer-only rotations in:
      <root>/_REVIEW_FLAGGED/rotation_overrides.json
    Keyed by relative path, values 0/90/180/270.
    """
    def __init__(self, root_dir: Path | None):
        self.root_dir = root_dir

    def _file(self) -> Path | None:
        if not self.root_dir:
            return None
        p = self.root_dir / "_REVIEW_FLAGGED" / "rotation_overrides.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def _key(self, img_path: Path) -> str:
        if self.root_dir:
            try:
                return str(img_path.relative_to(self.root_dir)).replace("\\", "/")
            except ValueError:
                pass
        return str(img_path).replace("\\", "/")

    def load_all(self) -> dict:
        f = self._file()
        if not f or not f.exists():
            return {}
        try:
            with open(f, "r", encoding="utf-8") as fp:
                d = json.load(fp)
            return d if isinstance(d, dict) else {}
        except Exception:
            return {}

    def save_all(self, data: dict) -> None:
        f = self._file()
        if not f:
            return
        tmp = None
        try:
            with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8", dir=str(f.parent)) as tf:
                tmp = tf.name
                json.dump(data, tf, indent=2)
            Path(tmp).replace(f)
        finally:
            if tmp:
                try:
                    Path(tmp).unlink(missing_ok=True)
                except Exception:
                    pass

    def get(self, img_path: Path) -> int:
        d = self.load_all()
        v = d.get(self._key(img_path), 0)
        try:
            v = int(v) % 360
        except Exception:
            v = 0
        return (v // 90) * 90

    def set(self, img_path: Path, deg: int) -> None:
        d = self.load_all()
        d[self._key(img_path)] = int(deg) % 360
        self.save_all(d)

    def clear(self, img_path: Path) -> bool:
        f = self._file()
        if not f or not f.exists():
            return False
        d = self.load_all()
        k = self._key(img_path)
        if k in d:
            del d[k]
            self.save_all(d)
            return True
        return False


# ------------------ EXIF / decode ------------------
def exif_datetime(path: Path) -> datetime | None:
    try:
        with Image.open(path) as img:
            exif = img.getexif()
            if not exif:
                return None
            dt = exif.get(TAG_DATETIME_ORIG) or exif.get(TAG_DATETIME)
            if not dt:
                return None
            return datetime.strptime(str(dt), "%Y:%m:%d %H:%M:%S")
    except Exception:
        return None


def file_mtime(path: Path) -> datetime:
    return datetime.fromtimestamp(path.stat().st_mtime)


def cv2_read_bgr(path: Path) -> np.ndarray | None:
    """Windows Unicode path-safe OpenCV read."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return bgr
    except Exception:
        return None


def resize_max_dim(bgr: np.ndarray, max_dim: int) -> np.ndarray:
    h, w = bgr.shape[:2]
    m = max(h, w)
    if m <= max_dim:
        return bgr
    scale = max_dim / float(m)
    return cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


# ------------------ Scoring ------------------
_FACE_CASCADE = None

def _get_face_cascade():
    global _FACE_CASCADE
    if _FACE_CASCADE is not None:
        return _FACE_CASCADE
    try:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        c = cv2.CascadeClassifier(cascade_path)
        if c.empty():
            _FACE_CASCADE = None
        else:
            _FACE_CASCADE = c
    except Exception:
        _FACE_CASCADE = None
    return _FACE_CASCADE


def detect_largest_face(gray: np.ndarray, min_face: int) -> tuple[int, int, int, int] | None:
    try:
        face_cascade = _get_face_cascade()
        if face_cascade is None:
            return None
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(min_face, min_face))
        if len(faces) == 0:
            return None
        x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
        return int(x), int(y), int(w), int(h)
    except Exception:
        return None


def lap_var(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def subject_sharpness(bgr: np.ndarray, center_crop: float, min_face: int) -> float:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]

    face = detect_largest_face(gray, min_face=min_face)
    if face is not None:
        x, y, fw, fh = face
        pad = int(0.25 * max(fw, fh))
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(w, x + fw + pad)
        y1 = min(h, y + fh + pad)
        roi = gray[y0:y1, x0:x1]
        if roi.size > 0:
            return lap_var(roi)

    crop = min(max(center_crop, 0.15), 0.95)
    cw, ch = int(w * crop), int(h * crop)
    x0, y0 = (w - cw) // 2, (h - ch) // 2
    roi = gray[y0:y0 + ch, x0:x0 + cw]
    return lap_var(roi)


def exposure_penalty(bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    mean = float(np.mean(gray))
    return abs(mean - 120.0) / 120.0


def quality_score(bgr: np.ndarray, center_crop: float, min_face: int) -> float:
    sharp = subject_sharpness(bgr, center_crop=center_crop, min_face=min_face)
    pen = exposure_penalty(bgr)
    return sharp - (20.0 * pen)


# ------------------ Duplicate features ------------------
def make_thumb_gray(bgr: np.ndarray, size: int) -> np.ndarray:
    t = cv2.resize(bgr, (size, size), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(t, cv2.COLOR_BGR2GRAY)


def phash_hex_from_bgr(bgr: np.ndarray, hash_size: int) -> str:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    return str(imagehash.phash(pil, hash_size=hash_size))


def hamming_distance(hex_a: str, hex_b: str) -> int:
    return imagehash.hex_to_hash(hex_a) - imagehash.hex_to_hash(hex_b)


def pixel_similarity_burst(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)

    a_f = a.astype(np.float32)
    b_f = b.astype(np.float32)

    mad = float(np.mean(cv2.absdiff(a, b)))
    sim_mad = 1.0 - (mad / 255.0)

    a0 = a_f - float(np.mean(a_f))
    b0 = b_f - float(np.mean(b_f))
    denom = float(np.sqrt(np.mean(a0 * a0) * np.mean(b0 * b0)) + 1e-8)
    ncc = float(np.mean(a0 * b0) / denom)
    sim_ncc = (ncc + 1.0) / 2.0

    sim = 0.75 * sim_ncc + 0.25 * sim_mad
    return max(0.0, min(1.0, sim))


# ------------------ Filesystem helpers ------------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def iter_images(root: Path, recursive: bool):
    it = root.rglob("*") if recursive else root.iterdir()
    for p in it:
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            yield p


def move_to_folder_recorded(paths: list[Path], dest_dir: Path) -> list[tuple[Path, Path]]:
    ensure_dir(dest_dir)
    moved: list[tuple[Path, Path]] = []

    for src in paths:
        if not src.exists():
            continue
        dest = dest_dir / src.name
        if dest.exists():
            stem, suf = src.stem, src.suffix
            i = 1
            while True:
                cand = dest_dir / f"{stem}__{i}{suf}"
                if not cand.exists():
                    dest = cand
                    break
                i += 1
        shutil.move(str(src), str(dest))
        moved.append((src, dest))
    return moved


def safe_restore_move(dest_current: Path, original_src: Path) -> Path | None:
    if not dest_current.exists():
        return None

    target = original_src
    if target.exists():
        stem, suf = original_src.stem, original_src.suffix
        i = 1
        while True:
            cand = original_src.parent / f"{stem}__RESTORED__{i}{suf}"
            if not cand.exists():
                target = cand
                break
            i += 1

    ensure_dir(target.parent)
    shutil.move(str(dest_current), str(target))
    return target


# ------------------ Grouping ------------------
def group_by_time(items: list[PhotoItem], window_seconds: int) -> list[list[PhotoItem]]:
    items = sorted(items, key=lambda x: x.ts)
    groups = []
    cur = []
    last_t = None
    for it in items:
        if last_t is None:
            cur = [it]
            last_t = it.ts
            continue
        if (it.ts - last_t).total_seconds() <= window_seconds:
            cur.append(it)
        else:
            groups.append(cur)
            cur = [it]
        last_t = it.ts
    if cur:
        groups.append(cur)
    return groups


def cluster_by_similarity(items: list[PhotoItem], hash_dist: int, sim_thresh: float) -> list[list[PhotoItem]]:
    candidates = [x for x in items if x.phash_hex and x.thumb_gray is not None]
    missing = [x for x in items if (not x.phash_hex) or x.thumb_gray is None]

    if len(candidates) <= 1:
        out = [[x] for x in candidates]
        out.extend([[x] for x in missing])
        return out

    parent = {x.path: x.path for x in candidates}

    def find(p):
        while parent[p] != p:
            parent[p] = parent[parent[p]]
            p = parent[p]
        return p

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(len(candidates)):
        ai = candidates[i]
        for j in range(i + 1, len(candidates)):
            aj = candidates[j]
            if hamming_distance(ai.phash_hex, aj.phash_hex) > hash_dist:
                continue
            if pixel_similarity_burst(ai.thumb_gray, aj.thumb_gray) >= sim_thresh:
                union(ai.path, aj.path)

    comps = defaultdict(list)
    by_path = {x.path: x for x in candidates}
    for x in candidates:
        comps[find(x.path)].append(by_path[x.path])

    out = list(comps.values())
    out.extend([[x] for x in missing])
    return out


# ======================
# Global app settings IO
# ======================
def _app_config_path() -> Path:
    base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
    if base:
        p = Path(base) / APP_NAME.replace(" ", "") / "settings.json"
    else:
        p = Path.home() / f".{APP_NAME.replace(' ', '').lower()}_settings.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def load_app_settings() -> dict:
    p = _app_config_path()
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_app_settings(data: dict) -> None:
    p = _app_config_path()
    tmp = None
    try:
        with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8", dir=str(p.parent)) as tf:
            tmp = tf.name
            json.dump(data, tf, indent=2)
        Path(tmp).replace(p)
    finally:
        if tmp:
            try:
                Path(tmp).unlink(missing_ok=True)
            except Exception:
                pass


def parse_version(v: str) -> tuple[int, int, int]:
    parts = (v or "").strip().split(".")
    nums = []
    for i in range(3):
        try:
            nums.append(int(parts[i]))
        except Exception:
            nums.append(0)
    return (nums[0], nums[1], nums[2])


# =================
# UI App
# =================
class BurstSelectorApp(tk.Tk):
    def __init__(self):
        super().__init__()

        # state
        self.root_dir: Path | None = None
        self.groups: list[list[PhotoItem]] = []
        self.group_index = 0
        self.keep_selected: set[Path] = set()
        self.current_preview_path: Path | None = None
        self.rotation_store = RotationOverrides(None)
        self.undo_stack: list[dict] = []
        self._review_session_id = None  # set when scan starts
        self._moved_total = 0
        self._groups_reviewed = 0


        # preferences (stored in app settings)
        self.include_subfolders = tk.BooleanVar(value=True)
        self.photos_within_seconds = tk.IntVar(value=30)
        self.duplicate_strictness = tk.DoubleVar(value=0.88)  # LOWER = more aggressive
        self.keep_default = tk.IntVar(value=2)
        self.auto_move_on_next = tk.BooleanVar(value=True)
        self.session_subfolder_extras = tk.BooleanVar(value=True)  # put moved extras into a per-session folder
        self.confirm_move_on_next = tk.BooleanVar(value=True)
        self.auto_update_on_launch = tk.BooleanVar(value=True)

        # theme
        self.theme_var = tk.StringVar(value="light")  # light / dark

        # --- default theme values (so UI can build before apply_theme runs) ---
        self._app_bg = "#f5f7fb"
        self._card_bg = "#ffffff"
        self._text = "#111827"
        self._muted = "#6b7280"
        self._border = "#e5e7eb"
        self._preview_bg = "#0b1220"
        self._toast_fg = "#065f46"
        self._overlay_text = "#cbd5e1"

        # internals
        self._max_dim = 2200
        self._thumb_px = 150
        self._thumb_gray_px = 256
        self._hash_size = 16
        self._hash_dist = 56
        self._min_face = 80
        self._center_crop = 0.50

        # ui vars
        self.status_var = tk.StringVar(value="File → Open Folder… then Start.")
        self.keep_count_var = tk.StringVar(value="")
        self.toast_var = tk.StringVar(value="")
        self.phase_var = tk.StringVar(value="")
        self.progress_var = tk.DoubleVar(value=0.0)
        self._overlay = None

        # preview (zoom/pan)
        self._preview_img_tk = None
        self._preview_base_pil: Image.Image | None = None
        self._preview_rotation_deg = 0
        self._zoom = 1.0
        self._fit_zoom = 1.0
        self._pan_x = 0
        self._pan_y = 0
        self._drag_start = None
        self._render_job = None

        # left tiled thumbnails
        self._thumb_tk_refs: dict[Path, ImageTk.PhotoImage] = {}
        self._thumb_cell_vars: dict[Path, tk.BooleanVar] = {}

        # load global settings (theme + recents + prefs)
        self._settings = load_app_settings()
        self._apply_loaded_settings()

        # window + styles
        self.title(f"{APP_NAME} v{APP_VERSION}")
        self.geometry("1320x860")

        self._style = ttk.Style()
        try:
            self._style.theme_use("clam")
        except Exception:
            pass

        self._build_menu()
        self._build_ui()
        self.apply_theme(self.theme_var.get())

        self._bind_shortcuts()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        
        # Auto-check for updates once per day, after UI is ready
        self.after(800, self._auto_check_updates_on_launch)
   

    # ---------- settings load/save ----------
    def _apply_loaded_settings(self):
        prefs = self._settings.get("preferences", {})
        if isinstance(prefs, dict):
            self.include_subfolders.set(bool(prefs.get("include_subfolders", True)))
            self.photos_within_seconds.set(int(prefs.get("photos_within_seconds", 30)))
            self.duplicate_strictness.set(float(prefs.get("duplicate_strictness", 0.88)))
            self.keep_default.set(int(prefs.get("keep_default", 2)))
            self.auto_move_on_next.set(bool(prefs.get("auto_move_on_next", True)))
            self.confirm_move_on_next.set(bool(prefs.get("confirm_move_on_next", True)))
            self.auto_update_on_launch.set(bool(prefs.get("auto_update_on_launch", True)))
            self._last_update_check = str(self._settings.get("last_update_check", ""))
            self.session_subfolder_extras.set(bool(prefs.get("session_subfolder_extras", True)))


        theme = self._settings.get("theme", "light")
        if theme in ("light", "dark"):
            self.theme_var.set(theme)
        
        # auto-update check bookkeeping
        self._last_update_check = str(self._settings.get("last_update_check", ""))
        self._auto_update_enabled = bool(self._settings.get("auto_update_on_launch", True))


    def _persist_settings(self):
        self._settings["theme"] = self.theme_var.get()
        self._settings["preferences"] = {
            "include_subfolders": bool(self.include_subfolders.get()),
            "photos_within_seconds": int(self.photos_within_seconds.get()),
            "duplicate_strictness": float(self.duplicate_strictness.get()),
            "keep_default": int(self.keep_default.get()),
            "auto_move_on_next": bool(self.auto_move_on_next.get()),
            "confirm_move_on_next": bool(self.confirm_move_on_next.get()),
            "auto_update_on_launch": bool(self.auto_update_on_launch.get()),
            "session_subfolder_extras": bool(self.session_subfolder_extras.get()),

        }
        self._settings["auto_update_on_launch"] = bool(getattr(self, "_auto_update_enabled", True))
        self._settings["last_update_check"] = str(getattr(self, "_last_update_check", ""))

        save_app_settings(self._settings)

    def _recent_folders(self) -> list[str]:
        rec = self._settings.get("recent_folders", [])
        if not isinstance(rec, list):
            return []
        out = []
        seen = set()
        for x in rec:
            if not isinstance(x, str):
                continue
            if x in seen:
                continue
            p = Path(x)
            if p.exists() and p.is_dir():
                out.append(x)
                seen.add(x)
        return out[:10]

    def _push_recent_folder(self, folder: Path):
        folder = folder.resolve()
        rec = self._recent_folders()
        s = str(folder)
        rec = [x for x in rec if x != s]
        rec.insert(0, s)
        self._settings["recent_folders"] = rec[:10]
        self._persist_settings()
        self._rebuild_recent_menu()

    # ---------- thread-safe ui helpers ----------
    def _ui(self, fn, *args):
        self.after(0, fn, *args)

    def _show_overlay(self, title_text: str):
        if self._overlay is not None:
            return
        self._overlay = tk.Toplevel(self)
        self._overlay.transient(self)
        self._overlay.grab_set()
        self._overlay.title("Working…")
        self._overlay.resizable(False, False)

        box = ttk.Frame(self._overlay, padding=18)
        box.pack(fill=tk.BOTH, expand=True)
        ttk.Label(box, text=title_text, font=("Segoe UI", 11, "bold")).pack(anchor="w")
        ttk.Label(box, text="Please wait…", foreground=self._muted).pack(anchor="w", pady=(6, 0))

        self.update_idletasks()
        w, h = 380, 135
        x = self.winfo_rootx() + (self.winfo_width() // 2) - (w // 2)
        y = self.winfo_rooty() + (self.winfo_height() // 2) - (h // 2)
        self._overlay.geometry(f"{w}x{h}+{x}+{y}")

    def _hide_overlay(self):
        if self._overlay is None:
            return
        try:
            self._overlay.grab_release()
        except Exception:
            pass
        self._overlay.destroy()
        self._overlay = None

    # =================
    # Menus
    # =================
    def _build_menu(self):
        self.menubar = tk.Menu(self)

        # File
        self.file_menu = tk.Menu(self.menubar, tearoff=0)
        self.file_menu.add_command(label="Open Folder…", command=self.choose_folder)
        self.recent_menu = tk.Menu(self.file_menu, tearoff=0)
        self.file_menu.add_cascade(label="Recent folders", menu=self.recent_menu)
        self.file_menu.add_separator()
        self.file_menu.add_command(label="Exit", command=self.destroy)
        self.menubar.add_cascade(label="File", menu=self.file_menu)

        # Settings
        self.settings_menu = tk.Menu(self.menubar, tearoff=0)

        theme_menu = tk.Menu(self.settings_menu, tearoff=0)
        theme_menu.add_radiobutton(label="Light", variable=self.theme_var, value="light", command=lambda: self.apply_theme("light"))
        theme_menu.add_radiobutton(label="Dark", variable=self.theme_var, value="dark", command=lambda: self.apply_theme("dark"))
        self.settings_menu.add_cascade(label="Theme", menu=theme_menu)

        self.settings_menu.add_command(label="Preferences…", command=self.open_preferences)
        self.menubar.add_cascade(label="Settings", menu=self.settings_menu)

        # Help
        self.help_menu = tk.Menu(self.menubar, tearoff=0)
        self.help_menu.add_command(label="Open Extras Folder", command=self.open_extras_folder)
        self.help_menu.add_command(label="Keyboard shortcuts", command=self.show_shortcuts)
        self.help_menu.add_separator()
        self.help_menu.add_command(label="Check for updates…", command=self.check_for_updates)
        self.help_menu.add_separator()
        self.help_menu.add_command(label="About", command=self.show_about)
        self.menubar.add_cascade(label="Help", menu=self.help_menu)

        self.config(menu=self.menubar)
        self._rebuild_recent_menu()

    def _rebuild_recent_menu(self):
        self.recent_menu.delete(0, tk.END)
        rec = self._recent_folders()
        if not rec:
            self.recent_menu.add_command(label="(none yet)", state="disabled")
            return

        for s in rec:
            self.recent_menu.add_command(
                label=s,
                command=lambda p=s: self._open_recent(Path(p))
            )
        self.recent_menu.add_separator()
        self.recent_menu.add_command(label="Clear list", command=self._clear_recent)

    def _open_recent(self, folder: Path):
        if folder.exists():
            self._set_root_dir(folder)
        else:
            messagebox.showerror("Folder missing", "That folder no longer exists.")
            self._rebuild_recent_menu()

    def _clear_recent(self):
        self._settings["recent_folders"] = []
        self._persist_settings()
        self._rebuild_recent_menu()

    # =================
    # Theme
    # =================
    def apply_theme(self, mode: str):
        mode = (mode or "light").lower()
        if mode not in ("light", "dark"):
            mode = "light"
        self.theme_var.set(mode)

        if mode == "dark":
            self._app_bg = "#0b1220"
            self._card_bg = "#111827"
            self._text = "#e5e7eb"
            self._muted = "#9ca3af"
            self._border = "#1f2937"
            self._preview_bg = "#000000"
            self._toast_fg = "#34d399"
            self._overlay_text = "#cbd5e1"
        else:
            self._app_bg = "#f5f7fb"
            self._card_bg = "#ffffff"
            self._text = "#111827"
            self._muted = "#6b7280"
            self._border = "#e5e7eb"
            self._preview_bg = "#0b1220"
            self._toast_fg = "#065f46"
            self._overlay_text = "#cbd5e1"

        self.configure(bg=self._app_bg)

        s = self._style
        s.configure("App.TFrame", background=self._app_bg)
        s.configure("Card.TFrame", background=self._card_bg, relief="flat")
        s.configure("App.TLabel", background=self._app_bg, foreground=self._text)
        s.configure("Muted.TLabel", background=self._app_bg, foreground=self._muted)
        s.configure("Card.TLabel", background=self._card_bg, foreground=self._text)
        s.configure("Primary.TButton", padding=(10, 6))
        s.configure("TButton", padding=(10, 6))
        s.configure("TSeparator", background=self._border)

        # Checkbuttons: default + dialog-safe mapping
        s.configure("TCheckbutton", background=self._card_bg, foreground=self._text)
        s.configure("Card.TCheckbutton", background=self._card_bg, foreground=self._text)
        s.map(
            "Card.TCheckbutton",
            foreground=[("active", self._text), ("disabled", self._muted)],
            background=[("active", self._card_bg)]
        )

        # tk widgets
        if hasattr(self, "thumb_canvas"):
            self.thumb_canvas.configure(bg=self._card_bg)
        if hasattr(self, "preview_canvas"):
            self.preview_canvas.configure(background=self._preview_bg)
        if hasattr(self, "toast_label"):
            try:
                self.toast_label.configure(foreground=self._toast_fg)
            except Exception:
                pass

        self._persist_settings()
        self._schedule_render()

    # =================
    # Preferences dialog
    # =================
    def open_preferences(self):
        dlg = tk.Toplevel(self)
        dlg.title("Preferences")
        dlg.transient(self)
        dlg.grab_set()
        dlg.resizable(False, False)
        try:
            dlg.configure(bg=self._app_bg)
        except Exception:
            pass

        wrap = ttk.Frame(dlg, padding=14, style="Card.TFrame")
        wrap.pack(fill=tk.BOTH, expand=True)

        ttk.Label(wrap, text="Preferences", font=("Segoe UI", 11, "bold"), style="Card.TLabel").pack(anchor="w", pady=(0, 10))

        row1 = ttk.Frame(wrap, style="Card.TFrame")
        row1.pack(fill=tk.X, pady=4)
        ttk.Checkbutton(row1, text="Include subfolders", variable=self.include_subfolders, style="Card.TCheckbutton").pack(anchor="w")

        row2 = ttk.Frame(wrap, style="Card.TFrame")
        row2.pack(fill=tk.X, pady=4)
        ttk.Label(row2, text="Photos taken within (seconds):", style="Card.TLabel").pack(side=tk.LEFT)
        ttk.Spinbox(row2, from_=1, to=120, textvariable=self.photos_within_seconds, width=6).pack(side=tk.LEFT, padx=8)

        row3 = ttk.Frame(wrap, style="Card.TFrame")
        row3.pack(fill=tk.X, pady=4)
        ttk.Label(row3, text="How similar counts as a duplicate:", style="Card.TLabel").pack(side=tk.LEFT)
        ttk.Spinbox(row3, from_=0.70, to=0.99, increment=0.01, textvariable=self.duplicate_strictness, width=6).pack(side=tk.LEFT, padx=8)
        ttk.Label(row3, text="(Lower = more aggressive)", foreground=self._muted, style="Card.TLabel").pack(side=tk.LEFT, padx=8)

        row4 = ttk.Frame(wrap, style="Card.TFrame")
        row4.pack(fill=tk.X, pady=4)
        ttk.Label(row4, text="Keep this many (default):", style="Card.TLabel").pack(side=tk.LEFT)
        ttk.Spinbox(row4, from_=1, to=5, textvariable=self.keep_default, width=6).pack(side=tk.LEFT, padx=8)

        row5 = ttk.Frame(wrap, style="Card.TFrame")
        row5.pack(fill=tk.X, pady=4)
        ttk.Checkbutton(row5, text="Auto-move extras when you click Next", variable=self.auto_move_on_next, style="Card.TCheckbutton").pack(anchor="w")

        row6 = ttk.Frame(wrap, style="Card.TFrame")
        row6.pack(fill=tk.X, pady=4)
        ttk.Checkbutton(row6, text="Ask before moving extras", variable=self.confirm_move_on_next, style="Card.TCheckbutton").pack(anchor="w")

        row7 = ttk.Frame(wrap, style="Card.TFrame")
        row7.pack(fill=tk.X, pady=4)
        ttk.Checkbutton(row7, text="Check for updates automatically on launch", variable=self.auto_update_on_launch, style="Card.TCheckbutton").pack(anchor="w")

        row8 = ttk.Frame(wrap, style="Card.TFrame")
        row8.pack(fill=tk.X, pady=4)
        ttk.Checkbutton(rowX, text="Put moved Extras into a dated folder (recommended)", variable=self.session_subfolder_extras, style="Card.TCheckbutton").pack(anchor="w")


        btns = ttk.Frame(wrap, style="Card.TFrame")
        btns.pack(fill=tk.X, pady=(14, 0))

        def on_ok():
            self._persist_settings()
            dlg.destroy()

        ttk.Button(btns, text="OK", command=on_ok).pack(side=tk.RIGHT)
        ttk.Button(btns, text="Cancel", command=dlg.destroy).pack(side=tk.RIGHT, padx=8)

        self._center_dialog(dlg)

    def _center_dialog(self, dlg: tk.Toplevel):
        self.update_idletasks()
        dlg.update_idletasks()
        w = dlg.winfo_width()
        h = dlg.winfo_height()
        x = self.winfo_rootx() + (self.winfo_width() // 2) - (w // 2)
        y = self.winfo_rooty() + (self.winfo_height() // 2) - (h // 2)
        dlg.geometry(f"+{x}+{y}")

    # =================
    # UI layout
    # =================
    def _build_ui(self):
        # Top strip
        top = ttk.Frame(self, style="App.TFrame", padding=10)
        top.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(top, text="Open Folder…", command=self.choose_folder, style="Primary.TButton").pack(side=tk.LEFT)
        ttk.Button(top, text="Start", command=self.scan_and_group, style="Primary.TButton").pack(side=tk.LEFT, padx=8)

        # Status row
        nav = ttk.Frame(self, style="App.TFrame", padding=(10, 0))
        nav.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(nav, textvariable=self.status_var, style="App.TLabel").pack(side=tk.LEFT)
        ttk.Button(nav, text="Undo", command=self.undo_last_move).pack(side=tk.RIGHT)
        ttk.Button(nav, text="Next ▶", command=self.next_group).pack(side=tk.RIGHT, padx=6)
        ttk.Button(nav, text="◀ Prev", command=self.prev_group).pack(side=tk.RIGHT, padx=6)
        ttk.Button(nav, text="Pick best for me", command=self.auto_select_best).pack(side=tk.RIGHT, padx=10)

        # Progress row
        prog = ttk.Frame(self, style="App.TFrame", padding=(10, 6))
        prog.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(prog, textvariable=self.phase_var, style="Muted.TLabel").pack(side=tk.LEFT, padx=(0, 10))
        self.progress = ttk.Progressbar(prog, orient="horizontal", mode="determinate", variable=self.progress_var)
        self.progress.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Main panes
        main = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        main.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # LEFT: tiled thumbnails in a scrollable canvas
        left_card = ttk.Frame(main, padding=10, style="Card.TFrame")
        main.add(left_card, weight=1)

        hdr = ttk.Frame(left_card, style="Card.TFrame")
        hdr.pack(fill=tk.X)
        ttk.Label(hdr, text="Burst group photos", font=("Segoe UI", 10, "bold"), style="Card.TLabel").pack(side=tk.LEFT)
        ttk.Label(hdr, textvariable=self.keep_count_var, style="Card.TLabel", foreground=self._muted).pack(side=tk.RIGHT)

        self.thumb_canvas = tk.Canvas(left_card, highlightthickness=0, bg=self._card_bg)
        self.thumb_scroll = ttk.Scrollbar(left_card, orient="vertical", command=self.thumb_canvas.yview)
        self.thumb_canvas.configure(yscrollcommand=self.thumb_scroll.set)

        self.thumb_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.thumb_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, pady=(10, 0))

        self.thumb_frame = ttk.Frame(self.thumb_canvas, style="Card.TFrame")
        self.thumb_frame_id = self.thumb_canvas.create_window((0, 0), window=self.thumb_frame, anchor="nw")

        self.thumb_frame.bind("<Configure>", lambda e: self.thumb_canvas.configure(scrollregion=self.thumb_canvas.bbox("all")))
        self.thumb_canvas.bind("<Configure>", self._on_thumb_canvas_resize)

        self._bind_mousewheel(self.thumb_canvas, self._on_thumb_wheel)
        self._bind_mousewheel(self.thumb_frame, self._on_thumb_wheel)

        # RIGHT: preview
        right_card = ttk.Frame(main, padding=10, style="Card.TFrame")
        main.add(right_card, weight=3)

        self.preview_title_var = tk.StringVar(value="Preview")
        ttk.Label(right_card, textvariable=self.preview_title_var, font=("Segoe UI", 11, "bold"), style="Card.TLabel").pack(side=tk.TOP, anchor="w")

        ctrl = ttk.Frame(right_card, style="Card.TFrame")
        ctrl.pack(side=tk.TOP, fill=tk.X, pady=(10, 8))

        ttk.Button(ctrl, text="Select file", command=self.select_file_in_explorer).pack(side=tk.LEFT)
        ttk.Button(ctrl, text="Open folder", command=self.open_folder).pack(side=tk.LEFT, padx=6)

        ttk.Separator(ctrl, orient="vertical").pack(side=tk.LEFT, fill=tk.Y, padx=12)

        ttk.Button(ctrl, text="Rotate Left", command=self.rotate_ccw).pack(side=tk.LEFT)
        ttk.Button(ctrl, text="Rotate Right", command=self.rotate_cw).pack(side=tk.LEFT, padx=6)
        ttk.Button(ctrl, text="Clear rotation", command=self.clear_saved_rotation).pack(side=tk.LEFT, padx=6)

        ttk.Separator(ctrl, orient="vertical").pack(side=tk.LEFT, fill=tk.Y, padx=12)

        ttk.Button(ctrl, text="Zoom −", command=lambda: self.zoom_step(1/1.15)).pack(side=tk.LEFT)
        ttk.Button(ctrl, text="Zoom +", command=lambda: self.zoom_step(1.15)).pack(side=tk.LEFT, padx=6)
        ttk.Button(ctrl, text="Fit", command=self.zoom_fit).pack(side=tk.LEFT, padx=6)
        ttk.Button(ctrl, text="100%", command=self.zoom_100).pack(side=tk.LEFT, padx=6)
        ttk.Button(ctrl, text="Reset view", command=self.reset_view).pack(side=tk.LEFT, padx=6)

        ttk.Separator(ctrl, orient="vertical").pack(side=tk.LEFT, fill=tk.Y, padx=12)

        self.preview_status_var = tk.StringVar(value="Mouse wheel = zoom, drag = pan")
        self.preview_status_lbl = ttk.Label(ctrl, textvariable=self.preview_status_var, style="Card.TLabel")
        self.preview_status_lbl.pack(side=tk.LEFT)

        self.preview_canvas = tk.Canvas(right_card, background=self._preview_bg, highlightthickness=0)
        self.preview_canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self._bind_mousewheel(self.preview_canvas, self._on_preview_wheel)
        self.preview_canvas.bind("<ButtonPress-1>", self._on_preview_press)
        self.preview_canvas.bind("<B1-Motion>", self._on_preview_drag)
        self.preview_canvas.bind("<Configure>", self._on_preview_resize)

        keep_row = ttk.Frame(right_card, style="Card.TFrame")
        keep_row.pack(side=tk.TOP, fill=tk.X, pady=(10, 0))

        self.keep_this_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(keep_row, text="Keep this photo", variable=self.keep_this_var, command=self._on_keep_toggle, style="Card.TCheckbutton").pack(side=tk.LEFT)
        ttk.Button(keep_row, text="Open Extras folder", command=self.open_extras_folder).pack(side=tk.RIGHT)

        bottom = ttk.Frame(self, style="App.TFrame", padding=(10, 0, 10, 10))
        bottom.pack(side=tk.BOTTOM, fill=tk.X)
        self.toast_label = ttk.Label(bottom, textvariable=self.toast_var, style="App.TLabel")
        self.toast_label.pack(side=tk.LEFT)
        self.toast_label.configure(foreground=self._toast_fg)

    # =================
    # Shortcuts / Help
    # =================
    def _bind_shortcuts(self):
        self.focus_set()

        # Navigation
        self.bind_all("<space>", lambda e: self.next_group())
        self.bind_all("<Right>", lambda e: self.next_group())
        self.bind_all("<Left>", lambda e: self.prev_group())

        # Undo
        self.bind_all("<KeyPress-u>", lambda e: self.undo_last_move())
        self.bind_all("<Control-z>", lambda e: self.undo_last_move())

        # Keep toggle
        self.bind_all("<KeyPress-k>", self._hotkey_toggle_keep)

        # Rotation
        self.bind_all("<KeyPress-r>", lambda e: self.rotate_cw())   # rotate right
        self.bind_all("<KeyPress-e>", lambda e: self.rotate_ccw())  # rotate left

        # Zoom
        self.bind_all("<KeyPress-f>", lambda e: self.zoom_fit())
        self.bind_all("<KeyPress-1>", lambda e: self.zoom_100())
        self.bind_all("<KeyPress-0>", lambda e: self.reset_view())
        self.bind_all("<Escape>", lambda e: self.reset_view())
        self.bind_all("<KeyPress-plus>", lambda e: self.zoom_step(1.15))
        self.bind_all("<KeyPress-equal>", lambda e: self.zoom_step(1.15))
        self.bind_all("<KeyPress-minus>", lambda e: self.zoom_step(1/1.15))
        self.bind_all("<KeyPress-underscore>", lambda e: self.zoom_step(1/1.15))

    def show_shortcuts(self):
        msg = (
            "Keyboard shortcuts:\n\n"
            "Space / Right Arrow: Next group\n"
            "Left Arrow: Previous group\n"
            "K: Keep / Unkeep current photo\n"
            "U or Ctrl+Z: Undo last move\n"
            "Mouse wheel: Zoom (towards cursor)\n"
            "Drag: Pan\n"
            "+ / = : Zoom in\n"
            "- : Zoom out\n"
            "F: Fit\n"
            "1: 100%\n"
            "0 or Esc: Reset view\n"
            "R: Rotate right (preview only)\n"
            "E: Rotate left (preview only)\n"
            "Tip: Double-click a thumbnail to toggle Keep\n"
        )
        messagebox.showinfo("Keyboard shortcuts", msg)

    def show_about(self):
        cfg = _app_config_path()
        msg = (
            f"{APP_NAME}\n"
            f"Version: {APP_VERSION}\n\n"
            "What it does:\n"
            "- Groups burst photos\n"
            "- Lets you keep the best ones\n"
            "- Moves extras into an Extras folder\n"
            "- Preview-only rotation is saved (no file edits)\n\n"
            f"Settings file:\n{cfg}"
        )
        messagebox.showinfo("About", msg)

    def _hotkey_toggle_keep(self, event=None):
        if not self.current_preview_path:
            return
        new_val = not self.keep_this_var.get()
        self.keep_this_var.set(new_val)
        self._on_keep_toggle()
        self.toast_var.set("Kept" if new_val else "Un-kept")

    # =================
    # Mouse wheel binding
    # =================
    def _bind_mousewheel(self, widget, handler):
        widget.bind("<MouseWheel>", handler)          # Windows
        widget.bind("<Button-4>", handler)            # Linux
        widget.bind("<Button-5>", handler)            # Linux

    # Left pane scroll
    def _on_thumb_wheel(self, event):
        if event.num == 4:
            delta = 1
        elif event.num == 5:
            delta = -1
        else:
            delta = int(event.delta / 120)
        self.thumb_canvas.yview_scroll(-delta * 2, "units")

    def _on_thumb_canvas_resize(self, event):
        self.thumb_canvas.itemconfigure(self.thumb_frame_id, width=event.width)
        # Reflow AFTER geometry is stable
        self.after_idle(self._reflow_tiles)

    # =================
    # Folder selection / recent folders
    # =================
    def choose_folder(self):
        folder = filedialog.askdirectory(title="Select image folder")
        if folder:
            self._set_root_dir(Path(folder))

    def _set_root_dir(self, folder: Path):
        self.root_dir = folder.resolve()
        self.rotation_store = RotationOverrides(self.root_dir)
        self.status_var.set(f"Folder: {self.root_dir}  |  Click Start")
        self.toast_var.set("")
        self.phase_var.set("")
        self.progress_var.set(0.0)
        self.keep_count_var.set("")
        self._push_recent_folder(self.root_dir)

    # =================
    # Scan & group
    # =================
    def scan_and_group(self):
        if not self.root_dir:
            messagebox.showerror("No folder", "Choose a folder first (File → Open Folder).")
            return
        self.toast_var.set("")
        self.phase_var.set("")
        self.progress_var.set(0.0)
        self._review_session_id = datetime.now().strftime("%Y-%m-%d_%H%M")
        self._moved_total = 0
        self._groups_reviewed = 0
        t = threading.Thread(target=self._scan_worker, daemon=True)
        t.start()

    def _scan_worker(self):
        self._ui(self._show_overlay, "Organizing photos…")
        try:
            root = self.root_dir
            review_dir = root / "_REVIEW_FLAGGED"

            self._ui(lambda: self.phase_var.set("Scanning files…"))
            paths = []
            for p in iter_images(root, recursive=self.include_subfolders.get()):
                try:
                    p.relative_to(review_dir)
                    continue
                except ValueError:
                    paths.append(p)

            if not paths:
                self._ui(lambda: self.status_var.set("No images found in that folder."))
                self._ui(lambda: self.phase_var.set(""))
                self._ui(lambda: self.progress_var.set(0.0))
                self._ui(lambda: self.keep_count_var.set(""))
                return

            total = len(paths)
            self._ui(lambda: self.phase_var.set(f"Reading photos… (0/{total})"))
            self._ui(lambda: self.progress_var.set(0.0))

            items: list[PhotoItem] = []
            for i, p in enumerate(paths, start=1):
                ts = exif_datetime(p) or file_mtime(p)

                bgr = cv2_read_bgr(p)
                if bgr is None:
                    continue
                bgr = resize_max_dim(bgr, self._max_dim)

                try:
                    q = quality_score(bgr, center_crop=self._center_crop, min_face=self._min_face)
                except Exception:
                    q = -1e9

                ph = None
                tg = None
                try:
                    ph = phash_hex_from_bgr(bgr, hash_size=self._hash_size)
                    tg = make_thumb_gray(bgr, size=self._thumb_gray_px)
                except Exception:
                    pass

                items.append(PhotoItem(path=p, ts=ts, phash_hex=ph, thumb_gray=tg, quality=q))

                if i % 25 == 0 or i == total:
                    pct = (i / total) * 60.0
                    self._ui(lambda i=i, total=total: self.phase_var.set(f"Reading photos… ({i}/{total})"))
                    self._ui(lambda pct=pct: self.progress_var.set(pct))

            self._ui(lambda: self.phase_var.set("Grouping bursts…"))
            time_groups = group_by_time(items, window_seconds=self.photos_within_seconds.get())

            self._ui(lambda: self.phase_var.set("Finding duplicates…"))
            groups: list[list[PhotoItem]] = []
            sim = float(self.duplicate_strictness.get())
            for g in time_groups:
                groups.extend(cluster_by_similarity(g, hash_dist=self._hash_dist, sim_thresh=sim))
            groups.sort(key=lambda g: (-len(g), min(x.ts for x in g)))

            total_thumbs = sum(len(g) for g in groups) if groups else 1
            done = 0
            self._ui(lambda: self.phase_var.set(f"Preparing thumbnails… (0/{total_thumbs})"))
            self._ui(lambda: self.progress_var.set(70.0))

            for g in groups:
                for it in g:
                    try:
                        with Image.open(it.path) as im:
                            im = ImageOps.exif_transpose(im)
                            im = im.convert("RGB")
                            im.thumbnail((self._thumb_px, self._thumb_px))
                            it.thumb_pil = im.copy()
                    except Exception:
                        it.thumb_pil = Image.new("RGB", (self._thumb_px, self._thumb_px), (50, 50, 50))

                    done += 1
                    if done % 60 == 0 or done == total_thumbs:
                        pct = 70.0 + (done / total_thumbs) * 30.0
                        self._ui(lambda done=done, total_thumbs=total_thumbs: self.phase_var.set(f"Preparing thumbnails… ({done}/{total_thumbs})"))
                        self._ui(lambda pct=pct: self.progress_var.set(pct))

            def apply_results():
                self.groups = groups
                self.group_index = 0
                self.keep_selected.clear()
                self.undo_stack.clear()

                self.status_var.set(f"Ready: {len(self.groups)} burst groups found.")
                self.phase_var.set("Ready")
                self.progress_var.set(100.0)

                if self.groups:
                    self._show_group(0)
                    self.auto_select_best()
                    self.toast_var.set("Tip: Check Keep on the best photo(s), then click Next.")
                else:
                    self.toast_var.set("No burst groups found.")
                    self.keep_count_var.set("")

            self._ui(apply_results)

        except Exception as e:
            self._ui(messagebox.showerror, "Error", str(e))
        finally:
            self._ui(self._hide_overlay)

    # =================
    # Tiled thumbnails
    # =================
    def _clear_tiles(self):
        for w in self.thumb_frame.winfo_children():
            w.destroy()
        self._thumb_tk_refs = {}
        self._thumb_cell_vars = {}

    def _show_group(self, idx: int):
        if not self.groups:
            return
        self.group_index = max(0, min(idx, len(self.groups) - 1))
        g = self.groups[self.group_index]
        self.status_var.set(f"Burst group {self.group_index + 1} of {len(self.groups)}  |  Photos: {len(g)}")

        self._clear_tiles()
        g_sorted = sorted(g, key=lambda x: x.ts)

        for it in g_sorted:
            tile = ttk.Frame(self.thumb_frame, padding=6, style="Card.TFrame")

            img_tk = ImageTk.PhotoImage(it.thumb_pil)
            self._thumb_tk_refs[it.path] = img_tk

            img_lbl = ttk.Label(tile, image=img_tk, style="Card.TLabel")
            img_lbl.pack(side=tk.TOP)

            keep_var = tk.BooleanVar(value=(it.path in self.keep_selected))
            self._thumb_cell_vars[it.path] = keep_var
            cb = ttk.Checkbutton(
                tile, text="Keep", variable=keep_var, style="Card.TCheckbutton",
                command=lambda p=it.path, v=keep_var: self._set_keep(p, v.get())
            )
            cb.pack(side=tk.TOP, pady=(4, 0))

            # If file was moved/missing, show note and disable click actions
            is_missing = not it.path.exists()
            if is_missing:
                ttk.Label(tile, text="Moved / missing", foreground=self._muted, style="Card.TLabel").pack(side=tk.TOP, pady=(4, 0))
            else:
                img_lbl.bind("<Button-1>", lambda e, p=it.path: self._set_preview(p))
                tile.bind("<Button-1>", lambda e, p=it.path: self._set_preview(p))
                img_lbl.bind("<Double-Button-1>", lambda e, p=it.path: self._toggle_keep_from_tile(p))

        # Reflow AFTER geometry is stable (fixes "only appears after resize")
        self.after_idle(self._reflow_tiles)

        kept = [x.path for x in g_sorted if x.path in self.keep_selected]
        # pick a valid preview (prefer kept, then first existing)
        default_preview = None
        if kept:
            for p in kept:
                if p.exists():
                    default_preview = p
                    break
        if default_preview is None:
            for it in g_sorted:
                if it.path.exists():
                    default_preview = it.path
                    break

        if default_preview is not None:
            self._set_preview(default_preview)
        else:
            self.current_preview_path = None
            self.preview_title_var.set("Preview: (no files available)")
            self._preview_base_pil = None
            self.preview_canvas.delete("all")

        self.thumb_canvas.yview_moveto(0.0)
        self._update_keep_count()

    def _reflow_tiles(self):
        children = list(self.thumb_frame.winfo_children())
        if not children:
            return

        frame_w = max(1, self.thumb_canvas.winfo_width())
        tile_w = self._thumb_px + 20
        cols = max(1, frame_w // tile_w)
        cols = min(cols, 4)

        for i, tile in enumerate(children):
            r = i // cols
            c = i % cols
            tile.grid(row=r, column=c, padx=8, pady=8, sticky="n")

        for c in range(cols):
            self.thumb_frame.grid_columnconfigure(c, weight=1)

    # =================
    # Keep selection
    # =================
    def _update_keep_count(self):
        if not self.groups:
            self.keep_count_var.set("")
            return
        g = self.groups[self.group_index]
        kept = sum(1 for it in g if it.path in self.keep_selected)
        self.keep_count_var.set(f"Kept: {kept}/{len(g)}")

    def _toggle_keep_from_tile(self, path: Path):
        new_val = path not in self.keep_selected
        self._set_keep(path, new_val)
        v = self._thumb_cell_vars.get(path)
        if v is not None:
            v.set(new_val)
        if self.current_preview_path == path:
            self.keep_this_var.set(new_val)
        self.toast_var.set("Kept" if new_val else "Un-kept")

    def _set_keep(self, path: Path, keep: bool):
        if keep:
            self.keep_selected.add(path)
        else:
            self.keep_selected.discard(path)

        if self.current_preview_path == path:
            self.keep_this_var.set(keep)

        self._update_keep_count()

    def _on_keep_toggle(self):
        if not self.current_preview_path:
            return
        self._set_keep(self.current_preview_path, self.keep_this_var.get())
        v = self._thumb_cell_vars.get(self.current_preview_path)
        if v is not None:
            v.set(self.keep_this_var.get())

    def auto_select_best(self):
        if not self.groups:
            return
        g = self.groups[self.group_index]
        k = max(1, int(self.keep_default.get()))
        ranked = sorted(g, key=lambda x: (x.quality, x.path.stat().st_size if x.path.exists() else 0), reverse=True)

        for it in g:
            self.keep_selected.discard(it.path)
        for it in ranked[:k]:
            self.keep_selected.add(it.path)

        for p, v in self._thumb_cell_vars.items():
            v.set(p in self.keep_selected)

        self.toast_var.set(f"Picked the best {k} photo(s). You can change Keep checkboxes.")
        if self.current_preview_path:
            self.keep_this_var.set(self.current_preview_path in self.keep_selected)

        self._update_keep_count()

    # =================
    # Preview (zoom/pan)
    # =================
    def _set_preview(self, path: Path):
        if not path.exists():
            return
        self.current_preview_path = path
        self.keep_this_var.set(path in self.keep_selected)

        rel = str(path)
        if self.root_dir:
            try:
                rel = str(path.relative_to(self.root_dir))
            except ValueError:
                rel = str(path)
        self.preview_title_var.set(f"Preview: {path.name}   |   {rel}")

        try:
            with Image.open(path) as im:
                im = ImageOps.exif_transpose(im)
                self._preview_base_pil = im.convert("RGB").copy()
        except Exception:
            self._preview_base_pil = Image.new("RGB", (900, 700), (40, 40, 40))

        self._preview_rotation_deg = self.rotation_store.get(path)
        if self._preview_rotation_deg != 0:
            self.preview_status_var.set(f"Saved rotation: {self._preview_rotation_deg}° (preview only)")
            self.toast_var.set("Rotation applied (preview only).")
        else:
            self.preview_status_var.set("Mouse wheel = zoom, drag = pan")

        self._pan_x = 0
        self._pan_y = 0

        # Ensure canvas geometry exists before computing fit (fixes "blank until resize")
        self.update_idletasks()
        self._compute_fit_zoom()
        self._zoom = self._fit_zoom

        self._schedule_render()

    def _compute_fit_zoom(self):
        if self._preview_base_pil is None:
            self._fit_zoom = 1.0
            return
        img = self._preview_base_pil
        if self._preview_rotation_deg % 360 != 0:
            img = img.rotate(self._preview_rotation_deg, expand=True)
        cw = max(1, self.preview_canvas.winfo_width())
        ch = max(1, self.preview_canvas.winfo_height())
        iw, ih = img.size
        self._fit_zoom = min(cw / iw, ch / ih)

    def _on_preview_resize(self, event=None):
        # Keep "fit" behavior stable across resizes if user is effectively at fit
        old_fit = self._fit_zoom
        self._compute_fit_zoom()
        if abs(self._zoom - old_fit) < 0.02:
            self._zoom = self._fit_zoom
            self._pan_x = 0
            self._pan_y = 0
        self._schedule_render()

    def zoom_step(self, factor: float):
        self._zoom = max(0.05, min(self._zoom * factor, 8.0))
        self._schedule_render()

    def zoom_fit(self):
        self._compute_fit_zoom()
        self._zoom = self._fit_zoom
        self._pan_x = 0
        self._pan_y = 0
        self._schedule_render()

    def zoom_100(self):
        self._zoom = 1.0
        self._pan_x = 0
        self._pan_y = 0
        self._schedule_render()

    def reset_view(self):
        self._pan_x = 0
        self._pan_y = 0
        self._compute_fit_zoom()
        self._zoom = self._fit_zoom
        self._schedule_render()

    # Zoom-to-cursor wheel behavior
    def _on_preview_wheel(self, event):
        if self._preview_base_pil is None:
            return

        old_zoom = self._zoom
        factor = 1.15 if (event.num == 4 or getattr(event, "delta", 0) > 0) else (1 / 1.15)
        new_zoom = max(0.05, min(old_zoom * factor, 8.0))
        if abs(new_zoom - old_zoom) < 1e-6:
            return

        img = self._preview_base_pil
        if self._preview_rotation_deg % 360 != 0:
            img = img.rotate(self._preview_rotation_deg, expand=True)
        iw, ih = img.size

        cw = max(1, self.preview_canvas.winfo_width())
        ch = max(1, self.preview_canvas.winfo_height())

        old_w, old_h = int(iw * old_zoom), int(ih * old_zoom)
        new_w, new_h = int(iw * new_zoom), int(ih * new_zoom)

        old_x0 = (cw - old_w) // 2 + self._pan_x
        old_y0 = (ch - old_h) // 2 + self._pan_y

        cx, cy = event.x, event.y
        rx = (cx - old_x0) / max(1, old_w)
        ry = (cy - old_y0) / max(1, old_h)

        new_x0 = cx - int(rx * new_w)
        new_y0 = cy - int(ry * new_h)

        base_x0 = (cw - new_w) // 2
        base_y0 = (ch - new_h) // 2
        self._pan_x = new_x0 - base_x0
        self._pan_y = new_y0 - base_y0

        self._zoom = new_zoom
        self._schedule_render()

    def _on_preview_press(self, event):
        self._drag_start = (event.x, event.y)

    def _on_preview_drag(self, event):
        if not self._drag_start:
            return
        dx = event.x - self._drag_start[0]
        dy = event.y - self._drag_start[1]
        self._drag_start = (event.x, event.y)
        self._pan_x += dx
        self._pan_y += dy
        self._schedule_render()

    def _schedule_render(self):
        if self._render_job is not None:
            try:
                self.after_cancel(self._render_job)
            except Exception:
                pass
        self._render_job = self.after(30, self._render_preview)

    def _render_preview(self):
        self._render_job = None
        self.preview_canvas.delete("all")

        if self._preview_base_pil is None:
            return

        img = self._preview_base_pil
        if self._preview_rotation_deg % 360 != 0:
            img = img.rotate(self._preview_rotation_deg, expand=True)

        iw, ih = img.size
        nw, nh = max(1, int(iw * self._zoom)), max(1, int(ih * self._zoom))
        img_resized = img.resize((nw, nh), Image.LANCZOS)

        self._preview_img_tk = ImageTk.PhotoImage(img_resized)

        cw = max(1, self.preview_canvas.winfo_width())
        ch = max(1, self.preview_canvas.winfo_height())

        x = (cw - nw) // 2 + self._pan_x
        y = (ch - nh) // 2 + self._pan_y

        self.preview_canvas.create_image(x, y, anchor="nw", image=self._preview_img_tk)
        self.preview_canvas.create_text(
            10, 10, anchor="nw",
            fill=self._overlay_text,
            text=f"Zoom: {int(self._zoom * 100)}%  (wheel to zoom, drag to pan)"
        )

    # Preview actions
    def rotate_cw(self):
        if not self.current_preview_path:
            return
        self._preview_rotation_deg = (self._preview_rotation_deg - 90) % 360
        self.rotation_store.set(self.current_preview_path, self._preview_rotation_deg)
        self.preview_status_var.set(f"Rotation saved: {self._preview_rotation_deg}° (preview only)")
        self.toast_var.set("Rotation saved (preview only).")
        self._compute_fit_zoom()
        self._schedule_render()

    def rotate_ccw(self):
        if not self.current_preview_path:
            return
        self._preview_rotation_deg = (self._preview_rotation_deg + 90) % 360
        self.rotation_store.set(self.current_preview_path, self._preview_rotation_deg)
        self.preview_status_var.set(f"Rotation saved: {self._preview_rotation_deg}° (preview only)")
        self.toast_var.set("Rotation saved (preview only).")
        self._compute_fit_zoom()
        self._schedule_render()

    def clear_saved_rotation(self):
        if not self.current_preview_path:
            return
        cleared = self.rotation_store.clear(self.current_preview_path)
        self._preview_rotation_deg = 0
        if cleared:
            self.preview_status_var.set("Saved rotation cleared")
            self.toast_var.set("Saved rotation cleared.")
        else:
            self.preview_status_var.set("No saved rotation to clear")
            self.toast_var.set("No saved rotation was saved for this photo.")
        self._compute_fit_zoom()
        self._schedule_render()

    # =================
    # Explorer helpers
    # =================
    def select_file_in_explorer(self):
        if not self.current_preview_path:
            return
        try:
            import subprocess
            subprocess.run(["explorer", "/select,", str(self.current_preview_path)], check=False)
        except Exception as e:
            messagebox.showerror("Explorer error", str(e))

    def open_folder(self):
        if not self.current_preview_path:
            return
        try:
            os.startfile(str(self.current_preview_path.parent))
        except Exception as e:
            messagebox.showerror("Explorer error", str(e))

    def open_extras_folder(self):
        if not self.root_dir:
            messagebox.showinfo("Extras folder", "Open a folder first.")
            return
        extras = self._extras_dest()
        ensure_dir(extras)
        try:
            os.startfile(str(extras))
        except Exception as e:
            messagebox.showerror("Explorer error", str(e))

    # =================
    # Move extras / navigation
    # =================
    def _show_finish_summary(self):
        if not self.root_dir:
            return

        groups_total = len(self.groups) if self.groups else 0
        extras_dir = self._extras_dest()
        session = self._review_session_id or ""

        msg = (
            "Finished reviewing ✅\n\n"
            f"Folder:\n{self.root_dir}\n\n"
            f"Groups reviewed: {self._groups_reviewed} / {groups_total}\n"
            f"Total moved to Extras: {self._moved_total}\n\n"
            f"Extras location:\n{extras_dir}\n"
        )

        # Simple action dialog
        if messagebox.askyesno("All done", msg + "\nOpen the Extras folder now?"):
            self.open_extras_folder()

    def _extras_dest(self) -> Path:
        base = self.root_dir / "_REVIEW_FLAGGED" / "Extras (Duplicates)"
        if self.session_subfolder_extras.get():
            sid = self._review_session_id or datetime.now().strftime("%Y-%m-%d_%H%M")
            return base / sid
        return base

    def _rejects_for_current_group(self) -> list[Path]:
        g = self.groups[self.group_index]
        keep = {it.path for it in g if it.path in self.keep_selected}
        return [it.path for it in g if it.path not in keep]

    def _auto_move_extras_current_group(self) -> int:
        if not self.root_dir or not self.groups:
            return 0
        g = self.groups[self.group_index]
        if len(g) <= 1:
            return 0
        rejects = self._rejects_for_current_group()
        if not rejects:
            return 0
        moved_pairs = move_to_folder_recorded(rejects, self._extras_dest())
        if moved_pairs:
            self.undo_stack.append({"group_index": self.group_index, "moves": moved_pairs})
        moved_count = len(moved_pairs)
        if moved_count:
            self._moved_total += moved_count
        return moved_count

    def next_group(self):
        if not self.groups:
            return

        if self.auto_move_on_next.get():
            do_move = True

            is_last = (self.group_index >= len(self.groups) - 1)

            # Always confirm on the last group
            if self.confirm_move_on_next.get() or is_last:
                rejects = self._rejects_for_current_group()
                do_move = messagebox.askyesno(
                    "Move extras?",
                    f"This will move {len(rejects)} photo(s) to:\n{self._extras_dest()}\n\n"
                    "You can Undo afterwards.\n\nMove now?"
                )

            if do_move:
                moved = self._auto_move_extras_current_group()
                if moved > 0:
                    self.toast_var.set(f"Moved {moved} extra photo(s) to Extras. (Undo is available.)")
                else:
                    self.toast_var.set("Nothing moved for this group.")
            else:
                self.toast_var.set("Move cancelled.")


        if self.group_index >= len(self.groups) - 1:
            self.status_var.set("All done ✅")
            self.phase_var.set("Done")
            self.progress_var.set(100.0)
            self.toast_var.set("Finished! Use Help → Open Extras Folder if needed.")
            self._show_finish_summary()
            return
            
        self._groups_reviewed = max(self._groups_reviewed, self.group_index + 1)
        self._show_group(self.group_index + 1)
        self.auto_select_best()

    def prev_group(self):
        if not self.groups:
            return
        if self.group_index == 0:
            return
        self._show_group(self.group_index - 1)
        self.toast_var.set("")

    def undo_last_move(self):
        if not self.undo_stack:
            messagebox.showinfo("Undo", "Nothing to undo.")
            return

        last = self.undo_stack.pop()
        moves: list[tuple[Path, Path]] = last.get("moves", [])
        target_group = int(last.get("group_index", self.group_index))

        restored = 0
        for original_src, dest_actual in reversed(moves):
            actual = safe_restore_move(dest_actual, original_src)
            if actual is not None:
                restored += 1

        self.toast_var.set(f"Undo complete: restored {restored} photo(s).")
        messagebox.showinfo("Undo complete", f"Restored {restored} photo(s).")

        self._show_group(target_group)

    # =================
    # Updates (simple, safe)
    # =================
    def _today_key(self) -> str:
        # local date string like "2026-01-17"
        return datetime.now().strftime("%Y-%m-%d")

    def _should_auto_check_updates(self) -> bool:
        if not UPDATE_JSON_URL.strip():
            return False
        if not self.auto_update_on_launch.get():
            return False
        last = getattr(self, "_last_update_check", "")
        return last != self._today_key()
    
    def _auto_check_updates_on_launch(self):
        if not self._should_auto_check_updates():
            return

        # mark as checked (so we don’t spam if offline)
        self._last_update_check = self._today_key()
        self._persist_settings()

        # silent background check; only notify if an update is actually available
        t = threading.Thread(target=self._check_updates_worker_silent, daemon=True)
        t.start()

    def check_for_updates(self):
        if not UPDATE_JSON_URL.strip():
            messagebox.showinfo(
                "Check for updates",
                "Update checking is not configured yet.\n\n"
                "When you're ready, set UPDATE_JSON_URL in the script to a JSON file with:\n"
                '{"version":"1.0.1","notes":"...","download_url":"https://.../PhotoPicker.exe"}'
            )
            return

        self.toast_var.set("Checking for updates…")
        t = threading.Thread(target=self._check_updates_worker, daemon=True)
        t.start()

    def _check_updates_worker(self):
        try:
            with urllib.request.urlopen(UPDATE_JSON_URL, timeout=8) as resp:
                data = resp.read().decode("utf-8", errors="replace")
            info = json.loads(data)

            latest = str(info.get("version", "")).strip()
            notes = str(info.get("notes", "")).strip()
            url = str(info.get("download_url", "")).strip()

            if not latest:
                self._ui(messagebox.showerror, "Updates", "Update JSON didn't include a version.")
                self._ui(lambda: self.toast_var.set(""))
                return

            if parse_version(latest) <= parse_version(APP_VERSION):
                self._ui(messagebox.showinfo, "Updates", f"You're up to date.\n\nInstalled: {APP_VERSION}\nLatest: {latest}")
                self._ui(lambda: self.toast_var.set("You're up to date."))
                return

            msg = f"Update available!\n\nInstalled: {APP_VERSION}\nLatest: {latest}"
            if notes:
                msg += f"\n\nWhat’s new:\n{notes}"
            if not url:
                msg += "\n\n(No download_url provided in update JSON.)"
                self._ui(messagebox.showinfo, "Updates", msg)
                self._ui(lambda: self.toast_var.set("Update info found (no download link)."))
                return

            def ask_download():
                ok = messagebox.askyesno("Update available", msg + "\n\nDownload update now? (The app will close and reopen.)")
                if ok:
                    self._download_update(url, latest)

            self._ui(ask_download)

        except Exception as e:
            self._ui(messagebox.showerror, "Updates", f"Could not check for updates:\n\n{e}")
            self._ui(lambda: self.toast_var.set("Update check failed."))

    def _check_updates_worker_silent(self):
        try:
            with urllib.request.urlopen(UPDATE_JSON_URL, timeout=8) as resp:
                data = resp.read().decode("utf-8", errors="replace")
            info = json.loads(data)

            latest = str(info.get("version", "")).strip()
            notes = str(info.get("notes", "")).strip()
            url = str(info.get("download_url", "")).strip()

            if not latest:
                return  # silent

            if parse_version(latest) <= parse_version(APP_VERSION):
                return  # up to date; silent

            msg = f"Update available!\n\nInstalled: {APP_VERSION}\nLatest: {latest}"
            if notes:
                msg += f"\n\nWhat’s new:\n{notes}"
            if not url:
                msg += "\n\n(No download link provided.)"

            def prompt():
                ok = messagebox.askyesno(
                    "Update available",
                    msg + "\n\nDownload and install now? (The app will close and reopen.)"
                )
                if ok and url:
                    self._download_update(url, latest)

            self._ui(prompt)

        except Exception:
            # silent on launch
            return

    def _download_update(self, url: str, latest: str):
        self.toast_var.set("Downloading update…")
        t = threading.Thread(target=self._download_update_worker, args=(url, latest), daemon=True)
        t.start()

    def _download_update_worker(self, url: str, latest: str):
        try:
            base = Path(sys.executable if getattr(sys, "frozen", False) else __file__).resolve().parent

            # Always download to a consistent "NEW" filename
            new_exe = base / f"{APP_NAME.replace(' ', '')}_NEW.exe"
            part = new_exe.with_suffix(".exe.part")

            # Clean up any previous partial download
            try:
                part.unlink(missing_ok=True)
            except Exception:
                pass

            # Stream download to avoid huge memory spikes
            self._ui(lambda: self.toast_var.set("Downloading update…"))
            with urllib.request.urlopen(url, timeout=30) as resp:
                with open(part, "wb") as f:
                    while True:
                        chunk = resp.read(1024 * 256)
                        if not chunk:
                            break
                        f.write(chunk)

            # Atomic-ish finalize
            try:
                new_exe.unlink(missing_ok=True)
            except Exception:
                pass
            part.replace(new_exe)

            def prompt_install():
                ok = messagebox.askyesno(
                    "Update downloaded",
                    f"Version {latest} has been downloaded.\n\n"
                    "Install it now? (The app will close and reopen.)"
                )
                if ok:
                    self._launch_apply_update_script_and_quit(base, new_exe, APP_VERSION)
                else:
                    messagebox.showinfo(
                        "Update ready",
                        f"Saved as:\n{new_exe}\n\nYou can install next time from Help → Check for updates."
                    )
                    self.toast_var.set("Update downloaded (not installed).")

            self._ui(prompt_install)

        except Exception as e:
            self._ui(messagebox.showerror, "Update download failed", str(e))
            self._ui(lambda: self.toast_var.set("Update download failed."))


    def _launch_apply_update_script_and_quit(self, base: Path, new_exe: Path, old_version: str):
        """
        Safely swaps PhotoPicker.exe in-place after the app exits:
          PhotoPicker.exe -> PhotoPicker_old_<version>.exe
          PhotoPicker_NEW.exe -> PhotoPicker.exe
        Then relaunches.
        """
        if not getattr(sys, "frozen", False):
            messagebox.showinfo(
                "Update downloaded",
                f"Downloaded:\n{new_exe}\n\n"
                "You're running from source (.py), so automatic EXE replacement is skipped."
            )
            return

        current_exe = Path(sys.executable).resolve()
        target_exe = current_exe
        backup_exe = base / f"{APP_NAME.replace(' ', '')}_old_{old_version}.exe"
        script_path = base / "apply_update.cmd"

        cmd = f"""@echo off
    setlocal enableextensions

    set "TARGET={str(target_exe)}"
    set "NEW={str(new_exe)}"
    set "BACKUP={str(backup_exe)}"

    echo Applying update...
    echo TARGET: %TARGET%
    echo NEW:    %NEW%
    echo BACKUP: %BACKUP%

    :: Wait for the app to fully exit by retrying the move (no file modification!)
    set "MOVED_OLD=0"
    for /L %%i in (1,1,60) do (
      if not exist "%TARGET%" (
        set "MOVED_OLD=1"
        goto :SWAP
      )
      del "%BACKUP%" >nul 2>&1
      move /y "%TARGET%" "%BACKUP%" >nul 2>&1
      if exist "%BACKUP%" (
        set "MOVED_OLD=1"
        goto :SWAP
      )
      timeout /t 1 /nobreak >nul
    )

    :SWAP
    if not exist "%NEW%" (
      echo ERROR: New file missing: %NEW%
      goto :FAIL
    )

    :: If old wasn't moved (rare), try one last delete of target (best-effort)
    if exist "%TARGET%" del "%TARGET%" >nul 2>&1

    move /y "%NEW%" "%TARGET%" >nul 2>&1
    if not exist "%TARGET%" (
      echo ERROR: Failed to move new exe into place.
      goto :FAIL
    )

    start "" "%TARGET%"
    del "%~f0" >nul 2>&1
    endlocal
    exit /b 0

    :FAIL
    echo.
    echo Update failed. Your original exe may be in:
    echo %BACKUP%
    echo.
    pause
    endlocal
    exit /b 1
    """

        script_path.write_text(cmd, encoding="utf-8")

        try:
            import subprocess
            subprocess.Popen(["cmd.exe", "/c", str(script_path)], cwd=str(base))
        except Exception as e:
            messagebox.showerror("Update install failed", f"Couldn't start installer script:\n\n{e}")
            return

        self.destroy()


    # =================
    # Close
    # =================
    def _on_close(self):
        self._persist_settings()
        self.destroy()


if __name__ == "__main__":
    app = BurstSelectorApp()
    app.mainloop()
