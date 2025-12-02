# GUI Module for AnonVision


import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import os
import cv2
import numpy as np
from pathlib import Path

# Silence OpenCV logs and disable OpenCL
try:
    os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
    if hasattr(cv2, "utils") and hasattr(cv2.utils, "logging"):
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
    if hasattr(cv2, "ocl"):
        cv2.ocl.setUseOpenCL(False)
except Exception:
    pass

from utils import validate_image_path, resize_image_for_display
from dnn_detector import DNNFaceDetector
from face_detector import FaceDetector  # Haar fallback


class AnonVisionGUI:
    """Main GUI class for AnonVision application"""

    def __init__(self, root: tk.Tk):
        # ---- root/window ----
        self.root = root
        self.root.title("AnonVision — Face Privacy Protection")
        self.root.geometry("1280x780")
        self.root.minsize(1080, 680)

        # ---- basic dark-ish ttk theme ----
        self._init_style()

        # ---- state ----
        self.current_image_path = None
        self.pil_image = None          # PIL.Image (original)
        self.img_bgr = None            # numpy BGR (for detection)
        self.original_bgr = None       # keep pristine original
        self.detected_faces = []       # list of (x, y, w, h)
        self.photo_image = None        # Tk image for display

        # protect selections (index-based)
        self.face_vars = []            # list[tk.BooleanVar], one per detected face
        self.face_thumbs = []          # list[ImageTk.PhotoImage] to keep refs

        # ---- detectors ----
        self.haar = None
        self.dnn = None

        try:
            self.dnn = DNNFaceDetector(score_threshold=0.5)
        except Exception as e:
            print(f"[WARN] YuNet/DNN unavailable: {e}")
            self.dnn = None

        try:
            self.haar = FaceDetector()
        except Exception as e:
            print(f"[WARN] Haar unavailable: {e}")
            self.haar = None

        # ---- detection settings ----
        self.detection_method = tk.StringVar(value="dnn" if self.dnn else "haar")
        self.dnn_confidence = tk.DoubleVar(value=0.50)
        self.scale_factor = tk.DoubleVar(value=1.10)
        self.min_neighbors = tk.IntVar(value=5)
        self.min_size = tk.IntVar(value=30)
        self.show_boxes = tk.BooleanVar(value=True)

        # ---- ANONYMIZATION SETTINGS (STRONG DEFAULTS) ----
        # Method: blur, pixelate, blackout
        self.anon_method = tk.StringVar(value="blur")
        
        # Blur settings (MUCH STRONGER)
        self.blur_strength = tk.IntVar(value=71)  # Very strong blur kernel
        self.blur_passes = tk.IntVar(value=6)     # Multiple passes for complete anonymization
        
        # Pixelation settings
        self.pixel_size = tk.IntVar(value=20)     # Large blocks
        
        # Coverage settings
        self.edge_feather = tk.IntVar(value=25)   # Slight feathering
        self.region_expansion = tk.IntVar(value=35)  # Expand 35% to cover hair/neck

        # ---- build UI ----
        self._build_ui()
        self._set_status("Ready — load an image to begin")

    # ---------------- UI ----------------

    def _init_style(self):
        """Minimal dark-ish ttk styling."""
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("TFrame", background="#18181B")
        style.configure("TLabel", background="#18181B", foreground="#E5E7EB")
        style.configure("TLabelframe", background="#18181B", foreground="#E5E7EB")
        style.configure("TLabelframe.Label", background="#18181B", foreground="#E5E7EB")
        style.configure("TButton", background="#27272A", foreground="#E5E7EB")
        style.configure("TCheckbutton", background="#18181B", foreground="#E5E7EB")
        style.configure("TRadiobutton", background="#18181B", foreground="#E5E7EB")
        style.configure("TSeparator", background="#3A3A3F")

    def _build_ui(self):
        # overall grid: left panel (controls) + right main (image)
        self.root.columnconfigure(0, weight=0)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        # LEFT: control panel
        left = ttk.Frame(self.root, padding=10)
        left.grid(row=0, column=0, sticky="nsw")
        left.columnconfigure(0, weight=1)

        self._build_top_bar(left)          # file + actions
        self._build_anon_settings(left)    # anonymization controls
        self._build_faces_panel(left)      # detected faces (protect selection)
        self._build_advanced(left)         # advanced options (collapsible)
        self._build_status(left)           # status bar under controls

        # RIGHT: image area
        self._build_image_area()

    def _build_top_bar(self, parent):
        wrap = ttk.LabelFrame(parent, text="Controls", padding=8)
        wrap.grid(row=0, column=0, sticky="ew")
        wrap.columnconfigure(10, weight=1)

        ttk.Button(wrap, text="📁 Select Image", command=self.select_image).grid(row=0, column=0, padx=4, pady=4, sticky="ew")
        ttk.Button(wrap, text="🔍 Detect", command=self.detect_faces).grid(row=0, column=1, padx=4, pady=4, sticky="ew")
        ttk.Button(wrap, text="🛡 Apply Anonymization", command=self.apply_anonymization).grid(row=0, column=2, padx=4, pady=4, sticky="ew")
        ttk.Button(wrap, text="💾 Save Result", command=self.save_image).grid(row=0, column=3, padx=4, pady=4, sticky="ew")

        ttk.Separator(wrap, orient="horizontal").grid(row=1, column=0, columnspan=4, sticky="ew", pady=(6, 4))

        ttk.Checkbutton(wrap, text="Show boxes", variable=self.show_boxes, command=self._refresh_display)\
            .grid(row=2, column=0, padx=4, pady=2, sticky="w")

        ttk.Label(wrap, text="Method:").grid(row=2, column=1, sticky="e")
        ttk.Radiobutton(wrap, text="DNN (YuNet)", value="dnn",
                        variable=self.detection_method,
                        command=self._on_method_change).grid(row=2, column=2, padx=(4, 0), sticky="w")
        ttk.Radiobutton(wrap, text="Haar", value="haar",
                        variable=self.detection_method,
                        command=self._on_method_change).grid(row=2, column=3, padx=4, sticky="w")

    def _build_anon_settings(self, parent):
        """Anonymization method and settings"""
        anon_frame = ttk.LabelFrame(parent, text="🛡 Anonymization Method", padding=10)
        anon_frame.grid(row=1, column=0, sticky="ew", pady=(8, 6))
        anon_frame.columnconfigure(1, weight=1)

        # Method selection
        row = 0
        method_frame = ttk.Frame(anon_frame)
        method_frame.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        
        ttk.Radiobutton(
            method_frame,
            text="Heavy Blur",
            variable=self.anon_method,
            value="blur",
            command=self._update_method_ui
        ).pack(side="left", padx=4)
        
        ttk.Radiobutton(
            method_frame,
            text="Pixelate",
            variable=self.anon_method,
            value="pixelate",
            command=self._update_method_ui
        ).pack(side="left", padx=4)
        
        ttk.Radiobutton(
            method_frame,
            text="Black Box",
            variable=self.anon_method,
            value="blackout",
            command=self._update_method_ui
        ).pack(side="left", padx=4)

        # Settings container (switches based on method)
        self.settings_container = ttk.Frame(anon_frame)
        self.settings_container.grid(row=1, column=0, columnspan=2, sticky="ew")
        
        # BLUR SETTINGS
        self.blur_settings = ttk.Frame(self.settings_container)
        
        ttk.Label(self.blur_settings, text="Blur Strength:").grid(row=0, column=0, sticky="w", pady=2)
        blur_frame = ttk.Frame(self.blur_settings)
        blur_frame.grid(row=0, column=1, sticky="ew", padx=(10, 0))
        blur_frame.columnconfigure(0, weight=1)
        
        self.blur_scale = ttk.Scale(
            blur_frame,
            from_=41,
            to=99,
            orient="horizontal",
            variable=self.blur_strength,
            command=lambda x: self.blur_strength.set(int(float(x)) // 2 * 2 + 1)
        )
        self.blur_scale.grid(row=0, column=0, sticky="ew")
        self.blur_label = ttk.Label(blur_frame, text="71")
        self.blur_label.grid(row=0, column=1, padx=(8, 0))
        self.blur_strength.trace('w', lambda *args: self.blur_label.config(text=str(self.blur_strength.get())))
        
        ttk.Label(self.blur_settings, text="Blur Passes:").grid(row=1, column=0, sticky="w", pady=2)
        passes_frame = ttk.Frame(self.blur_settings)
        passes_frame.grid(row=1, column=1, sticky="ew", padx=(10, 0))
        passes_frame.columnconfigure(0, weight=1)
        
        self.passes_scale = ttk.Scale(
            passes_frame,
            from_=3,
            to=10,
            orient="horizontal",
            variable=self.blur_passes
        )
        self.passes_scale.grid(row=0, column=0, sticky="ew")
        self.passes_label = ttk.Label(passes_frame, text="6")
        self.passes_label.grid(row=0, column=1, padx=(8, 0))
        self.blur_passes.trace('w', lambda *args: self.passes_label.config(text=str(int(self.blur_passes.get()))))
        
        # PIXELATE SETTINGS
        self.pixel_settings = ttk.Frame(self.settings_container)
        
        ttk.Label(self.pixel_settings, text="Pixel Block Size:").grid(row=0, column=0, sticky="w", pady=2)
        pixel_frame = ttk.Frame(self.pixel_settings)
        pixel_frame.grid(row=0, column=1, sticky="ew", padx=(10, 0))
        pixel_frame.columnconfigure(0, weight=1)
        
        self.pixel_scale = ttk.Scale(
            pixel_frame,
            from_=8,
            to=50,
            orient="horizontal",
            variable=self.pixel_size
        )
        self.pixel_scale.grid(row=0, column=0, sticky="ew")
        self.pixel_label = ttk.Label(pixel_frame, text="20")
        self.pixel_label.grid(row=0, column=1, padx=(8, 0))
        self.pixel_size.trace('w', lambda *args: self.pixel_label.config(text=str(int(self.pixel_size.get()))))
        
        # COMMON SETTINGS
        ttk.Separator(anon_frame, orient="horizontal").grid(row=2, column=0, columnspan=2, sticky="ew", pady=8)
        
        ttk.Label(anon_frame, text="Edge Softness:").grid(row=3, column=0, sticky="w")
        feather_frame = ttk.Frame(anon_frame)
        feather_frame.grid(row=3, column=1, sticky="ew", padx=(10, 0))
        feather_frame.columnconfigure(0, weight=1)
        
        ttk.Scale(
            feather_frame,
            from_=0,
            to=50,
            orient="horizontal",
            variable=self.edge_feather
        ).grid(row=0, column=0, sticky="ew")
        self.feather_label = ttk.Label(feather_frame, text="25%")
        self.feather_label.grid(row=0, column=1, padx=(8, 0))
        self.edge_feather.trace('w', lambda *args: self.feather_label.config(text=f"{self.edge_feather.get()}%"))
        
        ttk.Label(anon_frame, text="Coverage Area:").grid(row=4, column=0, sticky="w")
        expand_frame = ttk.Frame(anon_frame)
        expand_frame.grid(row=4, column=1, sticky="ew", padx=(10, 0))
        expand_frame.columnconfigure(0, weight=1)
        
        ttk.Scale(
            expand_frame,
            from_=10,
            to=60,
            orient="horizontal",
            variable=self.region_expansion
        ).grid(row=0, column=0, sticky="ew")
        self.expand_label = ttk.Label(expand_frame, text="35%")
        self.expand_label.grid(row=0, column=1, padx=(8, 0))
        self.region_expansion.trace('w', lambda *args: self.expand_label.config(text=f"{self.region_expansion.get()}%"))
        
        # Show initial method settings
        self._update_method_ui()

    def _update_method_ui(self):
        """Show/hide settings based on selected anonymization method"""
        # Hide all
        self.blur_settings.pack_forget()
        self.pixel_settings.pack_forget()
        
        # Show relevant
        method = self.anon_method.get()
        if method == "blur":
            self.blur_settings.pack(fill="x")
        elif method == "pixelate":
            self.pixel_settings.pack(fill="x")
        # blackout has no extra settings

    def _build_faces_panel(self, parent):
        self.faces_box = ttk.LabelFrame(parent, text="Detected Faces — tick to Protect (won't be anonymized)", padding=8)
        self.faces_box.grid(row=2, column=0, sticky="nsew", pady=(8, 6))
        self.faces_box.columnconfigure(0, weight=1)
        
        # scrollable area
        self.faces_canvas = tk.Canvas(self.faces_box, bg="#1f1f23", highlightthickness=0, height=180)
        self.faces_scroll = ttk.Scrollbar(self.faces_box, orient="vertical", command=self.faces_canvas.yview)
        self.faces_inner = ttk.Frame(self.faces_canvas)
        self.faces_inner.bind("<Configure>", lambda e: self.faces_canvas.configure(scrollregion=self.faces_canvas.bbox("all")))
        self.faces_window = self.faces_canvas.create_window((0, 0), window=self.faces_inner, anchor="nw")
        self.faces_canvas.configure(yscrollcommand=self.faces_scroll.set)

        self.faces_canvas.grid(row=0, column=0, sticky="nsew")
        self.faces_scroll.grid(row=0, column=1, sticky="ns")
        self.faces_box.rowconfigure(0, weight=1)

        # actions
        btns = ttk.Frame(self.faces_box)
        btns.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        ttk.Button(btns, text="Protect All", command=self._protect_all).pack(side="left", padx=4)
        ttk.Button(btns, text="Clear Protection", command=self._clear_protection).pack(side="left", padx=4)

        # placeholder
        self._refresh_faces_list()

    def _build_advanced(self, parent):
        cont = ttk.Frame(parent)
        cont.grid(row=3, column=0, sticky="ew")

        self._adv_open = tk.BooleanVar(value=False)
        self.adv_btn = ttk.Button(cont, text="▶ Show Detection Settings", command=self._toggle_advanced)
        self.adv_btn.pack(fill="x", pady=(2, 0))

        self.adv_frame = ttk.LabelFrame(cont, text="Detection Settings", padding=10)

        # DNN
        dnn_row = ttk.Frame(self.adv_frame)
        dnn_row.pack(fill="x", pady=(4, 2))
        ttk.Label(dnn_row, text="DNN Confidence:").pack(side="left")
        self.dnn_conf_sb = ttk.Spinbox(dnn_row, from_=0.10, to=0.95, increment=0.05, textvariable=self.dnn_confidence, width=6)
        self.dnn_conf_sb.pack(side="left", padx=6)

        # Haar
        haar = ttk.Frame(self.adv_frame)
        haar.pack(fill="x", pady=(6, 2))
        ttk.Label(haar, text="Haar Scale:").grid(row=0, column=0, sticky="w")
        ttk.Spinbox(haar, from_=1.01, to=2.0, increment=0.01, textvariable=self.scale_factor, width=6).grid(row=0, column=1, padx=6)
        ttk.Label(haar, text="Min Neighbors:").grid(row=0, column=2, sticky="w", padx=(12, 0))
        ttk.Spinbox(haar, from_=1, to=10, increment=1, textvariable=self.min_neighbors, width=6).grid(row=0, column=3, padx=6)
        ttk.Label(haar, text="Min Size(px):").grid(row=0, column=4, sticky="w", padx=(12, 0))
        ttk.Spinbox(haar, from_=10, to=200, increment=5, textvariable=self.min_size, width=6).grid(row=0, column=5, padx=6)

        # re-detect button
        ttk.Button(self.adv_frame, text="Re-detect with current settings", command=self._redetect).pack(pady=8)

        self._show_hide_advanced()

    def _build_status(self, parent):
        s = ttk.Frame(parent)
        s.grid(row=4, column=0, sticky="ew", pady=(6, 0))
        self.status_label = ttk.Label(s, text="Ready", anchor="w")
        self.status_label.pack(fill="x")

    def _build_image_area(self):
        # right side takes remaining space
        right = ttk.Frame(self.root, padding=(6, 10, 10, 10))
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)

        wrap = ttk.LabelFrame(right, text="Image Preview", padding=6)
        wrap.grid(row=0, column=0, sticky="nsew")
        wrap.columnconfigure(0, weight=1)
        wrap.rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(
            wrap,
            bg="#2A2A2E",
            highlightthickness=1,
            highlightbackground="#3A3A3F"
        )
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas.bind("<Configure>", lambda e: self._refresh_display())

        self._draw_placeholder()

    # ---------------- Actions ----------------

    def select_image(self):
        ft = [
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"),
            ("All files", "*.*")
        ]
        path = filedialog.askopenfilename(title="Select an image", filetypes=ft, initialdir=os.getcwd())
        if not path:
            self._set_status("No image selected")
            return
        if not self._validate_image(path):
            messagebox.showerror("Invalid File", "Please select a valid image file (JPG/PNG/BMP/TIFF).")
            return

        try:
            self.current_image_path = path
            self.pil_image = Image.open(path).convert("RGB")
            self.img_bgr = cv2.imread(path)
            self.original_bgr = self.img_bgr.copy()  # Keep pristine original
            self.detected_faces = []
            self.face_vars = []
            self.face_thumbs = []
            self._refresh_faces_list()
            self._set_status(f"Loaded: {Path(path).name} — click Detect")
            self._refresh_display()
        except Exception as e:
            messagebox.showerror("Load Error", f"Could not load image:\n{e}")
            self._set_status("Load failed")

    def detect_faces(self):
        if self.img_bgr is None:
            messagebox.showinfo("No image", "Load an image first.")
            return

        method = self.detection_method.get()
        self._set_status(f"Detecting faces ({method})...")
        faces = []

        try:
            if method == "dnn" and self.dnn:
                _, faces = self.dnn.detect_faces(self.img_bgr, confidence_threshold=float(self.dnn_confidence.get()))
            else:
                if not self.haar or not hasattr(self.haar, "face_cascade"):
                    messagebox.showerror("Haar unavailable", "Haar cascade not initialized.")
                    return
                gray = cv2.cvtColor(self.img_bgr, cv2.COLOR_BGR2GRAY)
                faces = self.haar.face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=float(self.scale_factor.get()),
                    minNeighbors=int(self.min_neighbors.get()),
                    minSize=(int(self.min_size.get()), int(self.min_size.get()))
                )
        except Exception as e:
            messagebox.showerror("Detection Error", str(e))
            faces = []

        self.detected_faces = [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]
        
        # Reset to original after detection
        self.img_bgr = self.original_bgr.copy()
        
        self._build_face_checklist()
        self._refresh_display()
        self._set_status(f"Detected: {len(self.detected_faces)} face(s)")

    def apply_anonymization(self):
        """Apply selected anonymization method to non-protected faces"""
        if self.original_bgr is None or not self.detected_faces:
            messagebox.showinfo("Nothing to do", "Load an image and detect faces first.")
            return

        method = self.anon_method.get()
        self.img_bgr = self._anonymize_faces(self.original_bgr.copy())
        self._refresh_display()

    def _anonymize_faces(self, image):
        """
        Apply strong anonymization to unprotected faces.
        Methods: Heavy blur, Pixelation, Black boxes
        """
        out = image.copy()
        method = self.anon_method.get()
        
        feather_pct = self.edge_feather.get() / 100.0
        expansion_pct = self.region_expansion.get() / 100.0
        
        protected = 0
        anonymized = 0

        for idx, (x, y, w, h) in enumerate(self.detected_faces):
            is_protected = (idx < len(self.face_vars)) and bool(self.face_vars[idx].get())
            if is_protected:
                protected += 1
                continue
            
            # Calculate expanded region
            ext_x = max(0, int(x - w * expansion_pct))
            ext_y = max(0, int(y - h * expansion_pct))
            ext_w = min(image.shape[1] - ext_x, int(w * (1 + 2 * expansion_pct)))
            ext_h = min(image.shape[0] - ext_y, int(h * (1 + 2 * expansion_pct)))
            
            # Extract region
            roi = out[ext_y:ext_y+ext_h, ext_x:ext_x+ext_w].copy()
            
            if roi.size == 0:
                continue
            
            # Apply anonymization method
            if method == "blur":
                # HEAVY BLUR - multiple passes with large kernel
                anonymized_roi = roi.copy()
                kernel = self.blur_strength.get()
                if kernel % 2 == 0:
                    kernel += 1
                passes = int(self.blur_passes.get())
                
                for _ in range(passes):
                    anonymized_roi = cv2.GaussianBlur(anonymized_roi, (kernel, kernel), 0)
                
            elif method == "pixelate":
                # PIXELATION
                pixel_size = int(self.pixel_size.get())
                h_roi, w_roi = roi.shape[:2]
                
                # Shrink
                temp = cv2.resize(roi, (w_roi // pixel_size, h_roi // pixel_size), interpolation=cv2.INTER_LINEAR)
                # Expand back (nearest neighbor for blocky effect)
                anonymized_roi = cv2.resize(temp, (w_roi, h_roi), interpolation=cv2.INTER_NEAREST)
                
            else:  # blackout
                # SOLID BLACK BOX
                anonymized_roi = np.zeros_like(roi)
            
            # Apply edge feathering if enabled (except for blackout)
            if feather_pct > 0 and method != "blackout":
                mask = self._create_feather_mask(ext_h, ext_w, feather_pct)
                mask_3ch = cv2.merge([mask, mask, mask])
                blended = (anonymized_roi * mask_3ch + roi * (1 - mask_3ch)).astype(np.uint8)
                out[ext_y:ext_y+ext_h, ext_x:ext_x+ext_w] = blended
            else:
                out[ext_y:ext_y+ext_h, ext_x:ext_x+ext_w] = anonymized_roi
            
            anonymized += 1
        
        method_names = {
            "blur": "Heavy Blur",
            "pixelate": "Pixelation",
            "blackout": "Black Box"
        }
        
        self._set_status(f"✓ Anonymized: {anonymized} | Protected: {protected} | Method: {method_names[method]}")
        
        return out

    def _create_feather_mask(self, height, width, feather_pct):
        """Create gradient mask for edge feathering"""
        y_coords = np.linspace(-1, 1, height)
        x_coords = np.linspace(-1, 1, width)
        x_grid, y_grid = np.meshgrid(x_coords, y_coords)
        
        dist_from_center = np.sqrt(x_grid**2 + y_grid**2)
        max_dist = np.sqrt(2)
        dist_normalized = dist_from_center / max_dist
        
        feather_start = 1.0 - feather_pct
        mask = np.clip((feather_start - dist_normalized) / feather_pct + 1, 0, 1)
        mask = cv2.GaussianBlur(mask.astype(np.float32), (21, 21), 0)
        
        return mask

    def save_image(self):
        if self.img_bgr is None:
            messagebox.showinfo("No image", "Nothing to save.")
            return

        path = filedialog.asksaveasfilename(
            title="Save Image",
            defaultextension=".png",
            filetypes=[("PNG (lossless)", "*.png"), ("JPEG", "*.jpg"), ("All files", "*.*")]
        )
        if not path:
            return

        try:
            cv2.imwrite(path, self.img_bgr)
            self._set_status(f"Saved: {Path(path).name}")
            messagebox.showinfo("Saved", f"Saved to:\n{path}")
        except Exception as e:
            messagebox.showerror("Save Error", str(e))

    # ---------------- Helpers ----------------

    def _validate_image(self, filepath: str) -> bool:
        if not validate_image_path(filepath):
            return False
        try:
            with Image.open(filepath) as im:
                im.verify()
            return True
        except Exception:
            return False

    def _draw_placeholder(self):
        self.canvas.delete("all")
        w = max(200, self.canvas.winfo_width())
        h = max(200, self.canvas.winfo_height())
        self.canvas.create_text(
            w // 2, h // 2,
            text="No image loaded\nUse 'Select Image' to begin",
            fill="#9CA3AF",
            font=("Segoe UI", 14),
            justify="center"
        )

    def _refresh_display(self):
        """Refresh image rendering on canvas"""
        if self.pil_image is None:
            self._draw_placeholder()
            return

        c_w = max(200, self.canvas.winfo_width())
        c_h = max(200, self.canvas.winfo_height())
        max_w = min(c_w - 20, 1600)
        max_h = min(c_h - 20, 1600)

        bgr_to_show = self.img_bgr.copy() if self.img_bgr is not None else None
        if bgr_to_show is None:
            self._draw_placeholder()
            return

        if self.show_boxes.get() and self.detected_faces:
            for idx, (x, y, w, h) in enumerate(self.detected_faces):
                is_protected = (idx < len(self.face_vars)) and bool(self.face_vars[idx].get())
                color = (16, 185, 129) if is_protected else (239, 68, 68)
                
                # Show extended region
                expansion_pct = self.region_expansion.get() / 100.0
                if expansion_pct > 0 and not is_protected:
                    ext_x = max(0, int(x - w * expansion_pct))
                    ext_y = max(0, int(y - h * expansion_pct))
                    ext_w = min(bgr_to_show.shape[1] - ext_x, int(w * (1 + 2 * expansion_pct)))
                    ext_h = min(bgr_to_show.shape[0] - ext_y, int(h * (1 + 2 * expansion_pct)))
                    
                    cv2.rectangle(bgr_to_show, (ext_x, ext_y), (ext_x + ext_w, ext_y + ext_h), 
                                (128, 128, 128), 1)
                
                cv2.rectangle(bgr_to_show, (x, y), (x + w, y + h), color, 2)

        rgb = cv2.cvtColor(bgr_to_show, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)

        disp = resize_image_for_display(pil, max_width=max_w, max_height=max_h, fill_background=(42, 42, 46))
        self.photo_image = ImageTk.PhotoImage(disp)

        self.canvas.delete("all")
        self.canvas.create_image(c_w // 2, c_h // 2, image=self.photo_image, anchor="center")

    def _on_method_change(self):
        m = self.detection_method.get()
        if m == "dnn" and not self.dnn:
            messagebox.showinfo("DNN unavailable", "YuNet/DNN not initialized; falling back to Haar.")
            self.detection_method.set("haar")
        self._set_status(f"Method set to: {self.detection_method.get()}")

    def _toggle_advanced(self):
        self._adv_open.set(not self._adv_open.get())
        self._show_hide_advanced()

    def _show_hide_advanced(self):
        if self._adv_open.get():
            self.adv_btn.config(text="▼ Hide Detection Settings")
            self.adv_frame.pack(fill="x", pady=(6, 0))
        else:
            self.adv_btn.config(text="▶ Show Detection Settings")
            self.adv_frame.pack_forget()

    def _redetect(self):
        if not self.current_image_path:
            messagebox.showinfo("No Image", "Please load an image first.")
            return
        self.detect_faces()

    def _set_status(self, msg: str):
        self.status_label.config(text=msg)
        self.root.update_idletasks()

    # ---------- Faces checklist (protect) ----------

    def _refresh_faces_list(self):
        """Clear faces panel"""
        for w in self.faces_inner.winfo_children():
            w.destroy()
        self.face_vars = []
        self.face_thumbs = []
        ttk.Label(self.faces_inner, text="No faces yet — click Detect").pack(anchor="w")

    def _build_face_checklist(self):
        """Populate faces panel with thumbnails + protect checkboxes"""
        for w in self.faces_inner.winfo_children():
            w.destroy()
        self.face_vars = []
        self.face_thumbs = []

        if self.img_bgr is None or not self.detected_faces:
            ttk.Label(self.faces_inner, text="No faces found").pack(anchor="w")
            return

        for idx, (x, y, w, h) in enumerate(self.detected_faces):
            row = ttk.Frame(self.faces_inner)
            row.pack(fill="x", pady=4)

            # thumbnail
            try:
                x0, y0 = max(0, x), max(0, y)
                x1, y1 = min(self.img_bgr.shape[1], x + w), min(self.img_bgr.shape[0], y + h)
                crop_bgr = self.img_bgr[y0:y1, x0:x1]
                if crop_bgr.size == 0:
                    raise ValueError("Empty crop")
                thumb_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                thumb_pil = Image.fromarray(thumb_rgb).resize((64, 64), Image.LANCZOS)
                thumb_tk = ImageTk.PhotoImage(thumb_pil)
                self.face_thumbs.append(thumb_tk)
                lbl = ttk.Label(row, image=thumb_tk)
                lbl.image = thumb_tk
                lbl.pack(side="left", padx=(0, 8))
            except Exception:
                ttk.Label(row, text="[face]").pack(side="left", padx=(0, 8))

            var = tk.BooleanVar(value=False)
            self.face_vars.append(var)
            cb = ttk.Checkbutton(row, text=f"Face {idx+1} — Protect", variable=var, command=self._refresh_display)
            cb.pack(side="left")

    def _protect_all(self):
        for v in self.face_vars:
            v.set(True)
        self._refresh_display()

    def _clear_protection(self):
        for v in self.face_vars:
            v.set(False)
        self._refresh_display()


# -------- entry --------
def main():
    root = tk.Tk()
    # High-DPI awareness (Windows)
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    app = AnonVisionGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()