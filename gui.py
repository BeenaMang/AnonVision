#!/usr/bin/env python3
"""
AnonVision Integrated GUI - Tkinter (intermediate style)
- YuNet + optional MediaPipe ensemble
- Whitelist with green boxes (protected), red boxes (to blur)
- Simple, readable code — not over-engineered
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import os

from dnn_detector import DNNFaceDetector
try:
    from mediapipe_detector import MediaPipeFaceDetector
except Exception:
    MediaPipeFaceDetector = None

from detector_ensemble import fuse_detections_iou
from face_whitelist import FaceWhitelist
from utils_CH import resize_image_for_display, validate_image_path


# ---------- Design tokens (dark theme) ----------
class DesignSystem:
    BG_PRIMARY = '#0F0F12'
    BG_SECONDARY = '#18181B'
    BG_TERTIARY = '#27272A'
    BG_HOVER = '#2D2D32'

    ACCENT_PRIMARY = '#3B82F6'
    ACCENT_SUCCESS = '#10B981'
    ACCENT_ERROR = '#EF4444'

    TEXT_PRIMARY = '#F9FAFB'
    TEXT_SECONDARY = '#9CA3AF'
    TEXT_TERTIARY = '#6B7280'

    BORDER_MEDIUM = '#4A4A4F'

    COLOR_PROTECTED = '#10B981'  # green
    COLOR_BLUR = '#EF4444'       # red
    COLOR_NEUTRAL = '#9CA3AF'    # gray

    FONT_HEADING = ('Segoe UI', 14, 'bold')
    FONT_SUBHEADING = ('Segoe UI', 12, 'bold')
    FONT_BODY = ('Segoe UI', 11)
    FONT_CAPTION = ('Segoe UI', 10)


# ---------- Simple styled widgets ----------
class ModernFrame(tk.Frame):
    def __init__(self, parent, **kwargs):
        kwargs.setdefault('bg', DesignSystem.BG_TERTIARY)
        kwargs.setdefault('relief', 'flat')
        super().__init__(parent, **kwargs)


class ModernLabel(tk.Label):
    def __init__(self, parent, text="", font=None, fg=None, **kwargs):
        kwargs.setdefault('bg', DesignSystem.BG_TERTIARY)
        kwargs.setdefault('fg', fg or DesignSystem.TEXT_PRIMARY)
        kwargs.setdefault('font', font or DesignSystem.FONT_BODY)
        super().__init__(parent, text=text, **kwargs)


class ModernButton(tk.Button):
    def __init__(self, parent, text="", command=None, variant='primary', width=None, **kwargs):
        palette = {
            'primary': (DesignSystem.ACCENT_PRIMARY, 'white', '#2563EB'),
            'success': (DesignSystem.ACCENT_SUCCESS, 'white', '#059669'),
            'danger':  (DesignSystem.ACCENT_ERROR,  'white', '#DC2626'),
            'secondary': (DesignSystem.BG_HOVER, DesignSystem.TEXT_PRIMARY, DesignSystem.BG_SECONDARY),
        }
        bg, fg, active_bg = palette.get(variant, palette['primary'])
        config = {
            'text': text, 'command': command,
            'font': DesignSystem.FONT_BODY,
            'bg': bg, 'fg': fg,
            'activebackground': active_bg, 'activeforeground': fg,
            'relief': 'flat', 'cursor': 'hand2',
            'padx': 15, 'pady': 8
        }
        if width: config['width'] = width
        super().__init__(parent, **config)
        self.bind('<Enter>', lambda e: self.configure(bg=active_bg))
        self.bind('<Leave>', lambda e: self.configure(bg=bg))


class ModernScale(tk.Scale):
    def __init__(self, parent, **kwargs):
        kwargs.setdefault('bg', DesignSystem.BG_TERTIARY)
        kwargs.setdefault('fg', DesignSystem.TEXT_PRIMARY)
        kwargs.setdefault('troughcolor', DesignSystem.BG_SECONDARY)
        kwargs.setdefault('activebackground', DesignSystem.ACCENT_PRIMARY)
        kwargs.setdefault('highlightthickness', 0)
        kwargs.setdefault('relief', 'flat')
        kwargs.setdefault('font', DesignSystem.FONT_CAPTION)
        kwargs.setdefault('orient', 'horizontal')
        super().__init__(parent, **kwargs)


# ---------- Whitelist Panel ----------
class WhitelistPanel(ModernFrame):
    def __init__(self, parent, whitelist_manager, on_update_callback=None):
        super().__init__(parent)
        self.whitelist = whitelist_manager
        self.on_update = on_update_callback
        self._build()
        self._refresh()

    def _build(self):
        header = ModernFrame(self, bg=DesignSystem.BG_SECONDARY)
        header.pack(fill='x', padx=10, pady=(10, 5))

        ModernLabel(header, text="🛡️ Protected Faces",
                    font=DesignSystem.FONT_SUBHEADING, bg=DesignSystem.BG_SECONDARY).pack(side='left', padx=10, pady=5)

        ModernButton(header, text="➕ Add", variant='success',
                     command=self._add, width=8).pack(side='right', padx=5)

        container = ModernFrame(self)
        container.pack(fill='both', expand=True, padx=10, pady=5)

        self.canvas = tk.Canvas(container, bg=DesignSystem.BG_TERTIARY, highlightthickness=0)
        sb = ttk.Scrollbar(container, orient='vertical', command=self.canvas.yview)
        self.scroll = ModernFrame(self.canvas)
        self.scroll.bind('<Configure>', lambda e: self.canvas.configure(scrollregion=self.canvas.bbox('all')))
        self.canvas.create_window((0, 0), window=self.scroll, anchor='nw')
        self.canvas.configure(yscrollcommand=sb.set)
        self.canvas.pack(side='left', fill='both', expand=True)
        sb.pack(side='right', fill='y')

        self.status = ModernLabel(self, text="0 faces protected",
                                  font=DesignSystem.FONT_CAPTION, fg=DesignSystem.TEXT_SECONDARY)
        self.status.pack(pady=(5, 10))

    def _add(self):
        path = filedialog.askopenfilename(
            title="Select Face Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
        )
        if not path:
            return
        name = NameInputDialog(self, "Enter name for this face:").run()
        if not name:
            return
        if self.whitelist.add_face(path, name):
            messagebox.showinfo("Added", f"Added {name} to whitelist")
            self._refresh()
            if self.on_update: self.on_update()
        else:
            messagebox.showerror("Error", "Failed to add face. Make sure image has a detectable face.")

    def _remove(self, name):
        if messagebox.askyesno("Confirm", f"Remove {name} from whitelist?"):
            self.whitelist.remove_face(name)
            self._refresh()
            if self.on_update: self.on_update()

    def _refresh(self):
        for w in self.scroll.winfo_children(): w.destroy()
        names = self.whitelist.get_whitelisted_names()
        self.status.config(text=f"{len(names)} face(s) protected")
        for n in names:
            row = ModernFrame(self.scroll, bg=DesignSystem.BG_SECONDARY)
            row.pack(fill='x', padx=5, pady=3)

            thumb = tk.Label(row, bg=DesignSystem.BG_SECONDARY, text="👤", font=('Arial', 20))
            thumb.pack(side='left', padx=10, pady=5)

            # Load small thumbnail if exists
            tpath = self.whitelist.get_thumbnail_path(n)
            if tpath and os.path.exists(tpath):
                try:
                    im = Image.open(tpath)
                    im.thumbnail((50, 50))
                    ph = ImageTk.PhotoImage(im)
                    thumb.configure(image=ph)
                    thumb.image = ph
                except Exception:
                    pass

            ModernLabel(row, text=n, font=DesignSystem.FONT_BODY, bg=DesignSystem.BG_SECONDARY)\
                .pack(side='left', padx=10, pady=5, fill='x', expand=True)

            ModernButton(row, text="❌", variant='danger', width=3, command=lambda name=n: self._remove(name))\
                .pack(side='right', padx=10, pady=5)


class NameInputDialog(tk.Toplevel):
    def __init__(self, parent, prompt="Enter name:"):
        super().__init__(parent)
        self.resizable(False, False)
        self.configure(bg=DesignSystem.BG_TERTIARY)
        self.title("Input Name")
        self.result = None
        ModernLabel(self, text=prompt).pack(pady=20)
        self.entry = tk.Entry(self, font=DesignSystem.FONT_BODY,
                              bg=DesignSystem.BG_SECONDARY, fg=DesignSystem.TEXT_PRIMARY,
                              insertbackground=DesignSystem.TEXT_PRIMARY)
        self.entry.pack(pady=10, padx=20, fill='x')
        self.entry.focus()
        buttons = ModernFrame(self); buttons.pack(pady=10)
        ModernButton(buttons, text="OK", command=self._ok).pack(side='left', padx=5)
        ModernButton(buttons, text="Cancel", variant='secondary', command=self.destroy).pack(side='left', padx=5)
        self.entry.bind('<Return>', lambda e: self._ok())
        self.transient(parent); self.grab_set()

    def run(self):
        self.wait_window(self)
        return self.result

    def _ok(self):
        self.result = self.entry.get().strip()
        self.destroy()


# ---------- Main App ----------
class AnonVisionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AnonVision - Face Privacy")
        self.root.geometry("1400x800")
        self.root.configure(bg=DesignSystem.BG_PRIMARY)

        # State
        self.current_image = None       # BGR
        self.current_image_path = None
        self.detected_faces = []        # [(x,y,w,h), ...]
        self.face_matches = []          # [{bbox, is_whitelisted, matched_name, confidence}, ...]
        self.photo_image = None         # PhotoImage for display

        # UI state
        self.yunet_confidence = tk.DoubleVar(value=0.50)
        self.mediapipe_confidence = tk.DoubleVar(value=0.50)
        self.blur_intensity = tk.IntVar(value=25)
        self.use_ensemble = tk.BooleanVar(value=True)
        self.show_boxes = tk.BooleanVar(value=True)
        self.whitelist_threshold = tk.DoubleVar(value=0.60)
        self.iou_merge = tk.DoubleVar(value=0.30)

        # Detectors
        self._init_detectors()

        # Whitelist
        self.whitelist = FaceWhitelist()

        # UI
        self._build_ui()
        self._status("Ready. Load an image to begin.")

    # -- helpers --
    def _status(self, msg):
        self.status_label.config(text=msg)
        self.root.update_idletasks()

    def _hex_to_bgr(self, hex_color):
        hex_color = hex_color.lstrip('#')
        r, g, b = (int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        return (b, g, r)

    # -- detectors --
    def _init_detectors(self):
        try:
            self.yunet_detector = DNNFaceDetector(prefer_gpu=False, score_threshold=0.5)
            self._status("✓ YuNet ready")
        except Exception as e:
            self.yunet_detector = None
            print("YuNet init failed:", e)

        try:
            self.mediapipe_detector = (MediaPipeFaceDetector(min_detection_confidence=float(self.mediapipe_confidence.get()))
                                       if MediaPipeFaceDetector else None)
            if self.mediapipe_detector:
                self._status("✓ MediaPipe ready")
            else:
                self._status("MediaPipe not installed (YuNet-only)")
        except Exception as e:
            self.mediapipe_detector = None
            print("MediaPipe init failed:", e)

    # -- UI --
    def _build_ui(self):
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=0)
        self.root.grid_columnconfigure(1, weight=1)

        self._build_sidebar()
        self._build_main_area()
        self._build_status()

    def _build_sidebar(self):
        sidebar = ModernFrame(self.root, bg=DesignSystem.BG_SECONDARY, width=350)
        sidebar.grid(row=0, column=0, sticky='nsew', padx=(0,1))
        sidebar.grid_propagate(False)

        title = ModernFrame(sidebar, bg=DesignSystem.BG_PRIMARY); title.pack(fill='x')
        ModernLabel(title, text="🛡️ AnonVision", font=DesignSystem.FONT_HEADING, bg=DesignSystem.BG_PRIMARY).pack(pady=20)

        controls = ModernFrame(sidebar, bg=DesignSystem.BG_SECONDARY); controls.pack(fill='x', padx=20, pady=10)
        ModernButton(controls, text="📁 Load Image", command=self._load_image).pack(fill='x', pady=5)
        ModernButton(controls, text="🔍 Detect Faces", command=self._detect_faces).pack(fill='x', pady=5)
        ModernButton(controls, text="🔮 Apply Anonymization", command=self._apply_blur, variant='success').pack(fill='x', pady=5)
        ModernButton(controls, text="💾 Save Result", command=self._save).pack(fill='x', pady=5)

        tk.Frame(sidebar, height=2, bg=DesignSystem.BORDER_MEDIUM).pack(fill='x', padx=20, pady=20)

        settings = ModernFrame(sidebar, bg=DesignSystem.BG_SECONDARY); settings.pack(fill='x', padx=20)
        ModernLabel(settings, text="⚙️ Detection Settings", font=DesignSystem.FONT_SUBHEADING).pack(anchor='w', pady=(10,5))

        tk.Checkbutton(settings, text="Use Ensemble Detection", variable=self.use_ensemble,
                       font=DesignSystem.FONT_CAPTION, bg=DesignSystem.BG_SECONDARY, fg=DesignSystem.TEXT_SECONDARY,
                       selectcolor=DesignSystem.BG_SECONDARY, activebackground=DesignSystem.BG_SECONDARY).pack(anchor='w', pady=5)

        tk.Checkbutton(settings, text="Show Detection Boxes", variable=self.show_boxes,
                       font=DesignSystem.FONT_CAPTION, bg=DesignSystem.BG_SECONDARY, fg=DesignSystem.TEXT_SECONDARY,
                       selectcolor=DesignSystem.BG_SECONDARY, activebackground=DesignSystem.BG_SECONDARY,
                       command=self._update_display).pack(anchor='w', pady=5)

        ModernLabel(settings, text=f"YuNet Confidence: {self.yunet_confidence.get():.2f}",
                    font=DesignSystem.FONT_CAPTION, fg=DesignSystem.TEXT_SECONDARY).pack(anchor='w', pady=(10,0))
        ModernScale(settings, from_=0.10, to=1.00, resolution=0.05,
                    variable=self.yunet_confidence,
                    command=lambda v: self._set_yunet_threshold(float(v))).pack(fill='x', pady=5)

        ModernLabel(settings, text=f"MediaPipe Confidence: {self.mediapipe_confidence.get():.2f}",
                    font=DesignSystem.FONT_CAPTION, fg=DesignSystem.TEXT_SECONDARY).pack(anchor='w', pady=(10,0))
        ModernScale(settings, from_=0.10, to=1.00, resolution=0.05,
                    variable=self.mediapipe_confidence).pack(fill='x', pady=5)

        # properly indented extra controls
        ModernLabel(settings, text=f"Whitelist Match Thr: {self.whitelist_threshold.get():.2f}",
                    font=DesignSystem.FONT_CAPTION, fg=DesignSystem.TEXT_SECONDARY).pack(anchor='w', pady=(10,0))
        ModernScale(settings, from_=0.40, to=0.80, resolution=0.02,
                    variable=self.whitelist_threshold).pack(fill='x', pady=5)

        ModernLabel(settings, text=f"Ensemble IoU Merge: {self.iou_merge.get():.2f}",
                    font=DesignSystem.FONT_CAPTION, fg=DesignSystem.TEXT_SECONDARY).pack(anchor='w', pady=(10,0))
        ModernScale(settings, from_=0.20, to=0.60, resolution=0.02,
                    variable=self.iou_merge).pack(fill='x', pady=5)

        ModernLabel(settings, text=f"Blur Intensity: {self.blur_intensity.get()}",
                    font=DesignSystem.FONT_CAPTION, fg=DesignSystem.TEXT_SECONDARY).pack(anchor='w', pady=(10,0))
        ModernScale(settings, from_=5, to=51, resolution=2, variable=self.blur_intensity).pack(fill='x', pady=5)

        tk.Frame(sidebar, height=2, bg=DesignSystem.BORDER_MEDIUM).pack(fill='x', padx=20, pady=20)

        container = ModernFrame(sidebar, bg=DesignSystem.BG_SECONDARY)
        container.pack(fill='both', expand=True, padx=10, pady=(0,10))
        self.whitelist_panel = WhitelistPanel(container, self.whitelist, on_update_callback=self._on_whitelist_change)
        self.whitelist_panel.pack(fill='both', expand=True)

    def _build_main_area(self):
        main = ModernFrame(self.root, bg=DesignSystem.BG_PRIMARY)
        main.grid(row=0, column=1, sticky='nsew')
        main.grid_rowconfigure(0, weight=1); main.grid_columnconfigure(0, weight=1)

        image_container = ModernFrame(main, bg=DesignSystem.BG_TERTIARY)
        image_container.grid(row=0, column=0, sticky='nsew', padx=20, pady=20)

        self.image_label = tk.Label(image_container, text="Use 'Load Image'",
                                    font=DesignSystem.FONT_HEADING, bg=DesignSystem.BG_TERTIARY,
                                    fg=DesignSystem.TEXT_TERTIARY)
        self.image_label.pack(expand=True, fill='both')
        # Recompute preview size when container changes
        self.image_label.bind('<Configure>', lambda e: self._update_display())

        info = ModernFrame(main, bg=DesignSystem.BG_SECONDARY, height=100)
        info.grid(row=1, column=0, sticky='ew', padx=20, pady=(0,20))
        info.grid_propagate(False)

        self.info_label = ModernLabel(info, text="No image loaded",
                                      font=DesignSystem.FONT_CAPTION, fg=DesignSystem.TEXT_SECONDARY,
                                      bg=DesignSystem.BG_SECONDARY)
        self.info_label.pack(pady=10, padx=20)

    def _build_status(self):
        bar = ModernFrame(self.root, bg=DesignSystem.BG_SECONDARY, height=30)
        bar.grid(row=1, column=0, columnspan=2, sticky='ew')
        bar.grid_propagate(False)
        self.status_label = ModernLabel(bar, text="Ready",
                                        font=DesignSystem.FONT_CAPTION, fg=DesignSystem.TEXT_SECONDARY,
                                        bg=DesignSystem.BG_SECONDARY)
        self.status_label.pack(side='left', padx=20, pady=5)

    # -- UI actions --
    def _set_yunet_threshold(self, value):
        if hasattr(self, 'yunet_detector') and self.yunet_detector:
            try:
                self.yunet_detector.set_score_threshold(value)
            except Exception:
                pass

    def _on_whitelist_change(self):
        if self.current_image is not None and self.detected_faces:
            self._match_whitelist()
            self._update_display()

    def _load_image(self):
        path = filedialog.askopenfilename(title="Select Image",
                                          filetypes=[("Image files","*.jpg *.jpeg *.png *.bmp"), ("All files","*.*")])
        if not path:
            return
        if not validate_image_path(path):
            messagebox.showerror('Invalid File', 'Please select a valid image file (jpg, png, bmp, tiff).')
            return
        img = cv2.imread(path)
        if img is None or img.size == 0:
            messagebox.showerror('Error', 'Could not read image. The file may be corrupted or unsupported.')
            return
        if img is None:
            messagebox.showerror("Error", "Could not read image")
            return
        self.current_image = img
        self.current_image_path = path
        self.detected_faces = []
        self.face_matches = []
        self._update_display()
        h, w = img.shape[:2]
        self.info_label.config(text=f"Image: {os.path.basename(path)} | Size: {w}x{h} | Click 'Detect Faces'")
        self._status(f"Loaded: {os.path.basename(path)}")

    def _detect_faces(self):
        if self.current_image is None:
            messagebox.showwarning("No Image", "Please load an image first")
            return

        self._status("Detecting faces...")
        faces_yunet, faces_mp = [], []

        try:
            if self.yunet_detector:
                _, faces_yunet = self.yunet_detector.detect_faces(
                    self.current_image, confidence_threshold=self.yunet_confidence.get()
                )
            if self.mediapipe_detector:
                _, faces_mp = self.mediapipe_detector.detect_faces(self.current_image)

            if self.use_ensemble.get() and faces_yunet and faces_mp:
                self.detected_faces = fuse_detections_iou(faces_yunet, faces_mp,
                                                          iou_merge_thresh=float(self.iou_merge.get()))
            else:
                self.detected_faces = faces_yunet or faces_mp or []

            self._match_whitelist()
            self._update_display()

            protected = sum(1 for m in self.face_matches if m['is_whitelisted'])
            to_blur = max(0, len(self.detected_faces) - protected)
            self.info_label.config(text=f"Detected: {len(self.detected_faces)} | Protected: {protected} | To blur: {to_blur}")
            self._status(f"Detection complete: {len(self.detected_faces)} faces")

        except Exception as e:
            messagebox.showerror("Detection Error", str(e))
            self._status("Detection failed")

    def _match_whitelist(self):
        if not self.detected_faces:
            self.face_matches = []
            return
        names = self.whitelist.get_whitelisted_names()
        if not names:
            self.face_matches = [ {'bbox': b, 'is_whitelisted': False, 'matched_name': None, 'confidence': 0.0}
                                  for b in self.detected_faces ]
            return
        self.face_matches = self.whitelist.match_detected_faces(
            self.current_image, self.detected_faces, threshold=float(self.whitelist_threshold.get())
        )

    def _update_display(self):
        if self.current_image is None: return
        disp = self.current_image.copy()

        if self.show_boxes.get() and self.detected_faces:
            wl_names = self.whitelist.get_whitelisted_names()
            for m in self.face_matches:
                x, y, w, h = m['bbox']
                if not wl_names:
                    color = self._hex_to_bgr(DesignSystem.COLOR_NEUTRAL)
                elif m['is_whitelisted']:
                    color = self._hex_to_bgr(DesignSystem.COLOR_PROTECTED)
                else:
                    color = self._hex_to_bgr(DesignSystem.COLOR_BLUR)
                cv2.rectangle(disp, (x,y), (x+w, y+h), color, 3)
                if m['is_whitelisted'] and m['matched_name']:
                    label = f"{m['matched_name']} ({m['confidence']:.0%})"
                    label_y = y - 10 if y - 10 > 20 else y + h + 20
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(disp, (x, label_y - th - 6), (x + tw + 10, label_y + 6), color, -1)
                    cv2.putText(disp, label, (x + 5, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        # -> PIL, aspect fit, then PhotoImage
        pil = Image.fromarray(cv2.cvtColor(disp, cv2.COLOR_BGR2RGB))
        max_w = max(100, self.image_label.winfo_width() or 800)
        max_h = max(100, self.image_label.winfo_height() or 600)
        pil = resize_image_for_display(pil, max_width=max_w, max_height=max_h)
        self.photo_image = ImageTk.PhotoImage(pil)
        self.image_label.configure(image=self.photo_image, text='')
        # Keep a strong reference on the instance to avoid GC
        self.image_label.image = self.photo_image

    def _apply_blur(self):
        if self.current_image is None:
            messagebox.showwarning("No Image", "Please load an image first"); return
        if not self.detected_faces:
            messagebox.showinfo("No Faces", "No faces detected. Run detection first."); return

        self._status("Applying anonymization...")
        out = self.current_image.copy()
        k = int(self.blur_intensity.get())
        if k % 2 == 0: k += 1
        blurred_count = 0

        for m in self.face_matches:
            if not m['is_whitelisted']:
                x, y, w, h = m['bbox']
                face = out[y:y+h, x:x+w]
                if face.size == 0: continue
                out[y:y+h, x:x+w] = cv2.GaussianBlur(face, (k, k), 0)
                blurred_count += 1

        self.current_image = out
        self._update_display()
        protected = len(self.detected_faces) - blurred_count
        self.info_label.config(text=f"Anonymization complete! Blurred: {blurred_count} | Protected: {protected}")
        self._status(f"Anonymized {blurred_count}, protected {protected}")

    def _save(self):
        if self.current_image is None:
            messagebox.showwarning("No Image", "No image to save"); return
        path = filedialog.asksaveasfilename(title="Save Result", defaultextension=".jpg",
                                            filetypes=[("JPEG","*.jpg"),("PNG","*.png"),("All files","*.*")])
        if not path: return
        try:
            cv2.imwrite(path, self.current_image)
            messagebox.showinfo("Saved", f"Saved to:\n{path}")
            self._status(f"Saved: {os.path.basename(path)}")
        except Exception as e:
            messagebox.showerror("Save Error", str(e))

# ---------- entry ----------
def main():
    root = tk.Tk()
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    app = AnonVisionApp(root)
    root.mainloop()

if __name__ == "__main__":
    print("Starting AnonVision Integrated...")
    print("=" * 50)
    main()
