#!/usr/bin/env python3
"""
AnonVision Professional - Modern Three-Panel Interface
YuNet Only Version (No MediaPipe)
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw, ImageFilter
import cv2
import numpy as np
import os
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import time
import sys

# Silence OpenCV warnings
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
cv2.setUseOptimized(True)

# Import required modules with proper error handling
try:
    from utils import validate_image_path
except ImportError:
    print("WARNING: utils.py not found - image validation disabled")
    def validate_image_path(path):
        return os.path.isfile(path)

try:
    from dnn_detector import DNNFaceDetector
except ImportError:
    print("ERROR: dnn_detector.py not found - YuNet detector unavailable")
    DNNFaceDetector = None

try:
    from face_whitelist import FaceWhitelist
except ImportError:
    print("WARNING: face_whitelist.py not found - whitelist features disabled")
    class FaceWhitelist:
        def __init__(self):
            pass


# ============================================================================
# PROFESSIONAL DESIGN SYSTEM
# ============================================================================

class ProDesign:
    """Professional design system inspired by modern video editing apps"""
    
    # Main Colors (Dark Theme)
    BG_DARKEST = '#0A0A0B'      # Main background
    BG_DARKER = '#141416'       # Panel backgrounds
    BG_DARK = '#1C1C1F'         # Sub-panels
    BG_MEDIUM = '#252528'       # Elevated surfaces
    BG_LIGHT = '#2D2D31'        # Hover states
    BG_LIGHTER = '#37373B'      # Active states
    
    # Accent Colors
    ACCENT_BLUE = '#4A9EFF'     # Primary actions
    ACCENT_GREEN = '#4ADE80'    # Success/Protected
    ACCENT_RED = '#F87171'      # Danger/Anonymize
    ACCENT_PURPLE = '#A78BFA'   # Special features
    ACCENT_YELLOW = '#FBBF24'   # Warnings
    
    # Text Colors
    TEXT_PRIMARY = '#FFFFFF'    # Main text
    TEXT_SECONDARY = '#B4B4B8'  # Secondary text
    TEXT_TERTIARY = '#7C7C82'   # Disabled/hints
    TEXT_ACCENT = '#4A9EFF'     # Links/highlights
    
    # Borders
    BORDER_SUBTLE = '#28282C'   # Panel borders
    BORDER_DEFAULT = '#37373B'  # Input borders
    BORDER_FOCUS = '#4A9EFF'    # Focused inputs
    
    # Fonts
    FONT_BRAND = ('Segoe UI', 18, 'bold')
    FONT_HEADING = ('Segoe UI', 13, 'bold')
    FONT_SUBHEADING = ('Segoe UI', 11, 'bold')
    FONT_BODY = ('Segoe UI', 10)
    FONT_SMALL = ('Segoe UI', 9)
    FONT_MONO = ('Consolas', 10)
    
    # Spacing & Sizing
    SIDEBAR_WIDTH = 260
    PROPERTIES_WIDTH = 320
    TOOLBAR_HEIGHT = 48
    STATUS_HEIGHT = 28
    CORNER_RADIUS = 6
    
    @classmethod
    def apply_theme(cls, root):
        """Apply professional theme to root window"""
        style = ttk.Style(root)
        style.theme_use('clam')
        
        # Configure ttk widgets
        style.configure('Pro.TFrame', background=cls.BG_DARKER, relief='flat')
        style.configure('Panel.TFrame', background=cls.BG_DARK, relief='flat', borderwidth=1)
        style.configure('Toolbar.TFrame', background=cls.BG_MEDIUM, relief='flat')
        
        style.configure('Pro.TLabel', background=cls.BG_DARKER, foreground=cls.TEXT_PRIMARY)
        style.configure('Heading.TLabel', background=cls.BG_DARKER, foreground=cls.TEXT_PRIMARY, 
                       font=cls.FONT_HEADING)
        style.configure('Small.TLabel', background=cls.BG_DARKER, foreground=cls.TEXT_SECONDARY,
                       font=cls.FONT_SMALL)
        
        # Button styles
        style.configure('Pro.TButton', 
                       background=cls.BG_MEDIUM,
                       foreground=cls.TEXT_PRIMARY,
                       borderwidth=0,
                       focuscolor='none',
                       padding=(10, 6))
        style.map('Pro.TButton',
                 background=[('active', cls.BG_LIGHT), ('pressed', cls.BG_LIGHTER)])
        
        style.configure('Primary.TButton',
                       background=cls.ACCENT_BLUE,
                       foreground='white',
                       borderwidth=0,
                       focuscolor='none',
                       padding=(12, 8))
        style.map('Primary.TButton',
                 background=[('active', '#5BA5FF'), ('pressed', '#3A8EEF')])


# ============================================================================
# CUSTOM MODERN WIDGETS
# ============================================================================

class ModernButton(tk.Frame):
    """Modern button with icon support and hover effects"""
    def __init__(self, parent, text="", icon="", command=None, variant="default", 
                 width=None, height=36, **kwargs):
        super().__init__(parent, bg=parent['bg'], **kwargs)
        
        # Color schemes
        schemes = {
            'default': {
                'bg': ProDesign.BG_MEDIUM,
                'fg': ProDesign.TEXT_PRIMARY,
                'hover': ProDesign.BG_LIGHT,
                'press': ProDesign.BG_LIGHTER
            },
            'primary': {
                'bg': ProDesign.ACCENT_BLUE,
                'fg': '#FFFFFF',
                'hover': '#5BA5FF',
                'press': '#3A8EEF'
            },
            'success': {
                'bg': ProDesign.ACCENT_GREEN,
                'fg': '#FFFFFF',
                'hover': '#5BE491',
                'press': '#3ACE70'
            },
            'danger': {
                'bg': ProDesign.ACCENT_RED,
                'fg': '#FFFFFF',
                'hover': '#F98282',
                'press': '#F76060'
            },
            'ghost': {
                'bg': 'transparent',
                'fg': ProDesign.TEXT_SECONDARY,
                'hover': ProDesign.BG_LIGHT,
                'press': ProDesign.BG_LIGHTER
            }
        }
        
        self.colors = schemes.get(variant, schemes['default'])
        self.command = command
        
        # Create button
        self.button = tk.Label(
            self,
            text=f"{icon}  {text}" if icon else text,
            bg=self.colors['bg'] if self.colors['bg'] != 'transparent' else parent['bg'],
            fg=self.colors['fg'],
            font=ProDesign.FONT_BODY,
            cursor='hand2',
            pady=8,
            padx=12
        )
        self.button.pack(fill='both', expand=True)
        
        if width:
            self.button.configure(width=width)
        
        # Bindings
        self.button.bind('<Enter>', self._on_enter)
        self.button.bind('<Leave>', self._on_leave)
        self.button.bind('<Button-1>', self._on_press)
        self.button.bind('<ButtonRelease-1>', self._on_release)
    
    def _on_enter(self, e):
        if self.colors['bg'] != 'transparent':
            self.button.configure(bg=self.colors['hover'])
    
    def _on_leave(self, e):
        bg = self.colors['bg'] if self.colors['bg'] != 'transparent' else self.master['bg']
        self.button.configure(bg=bg)
    
    def _on_press(self, e):
        self.button.configure(bg=self.colors['press'])
    
    def _on_release(self, e):
        self._on_enter(e)
        if self.command:
            self.command()


class IconButton(tk.Label):
    """Compact icon-only button for toolbars"""
    def __init__(self, parent, icon, command=None, tooltip="", size=32):
        super().__init__(
            parent,
            text=icon,
            bg=parent['bg'],
            fg=ProDesign.TEXT_SECONDARY,
            font=('Segoe UI', 12),
            cursor='hand2',
            width=size//8,
            height=1
        )
        self.command = command
        self.default_fg = ProDesign.TEXT_SECONDARY
        self.hover_fg = ProDesign.TEXT_PRIMARY
        
        self.bind('<Enter>', lambda e: self.configure(fg=self.hover_fg, bg=ProDesign.BG_LIGHT))
        self.bind('<Leave>', lambda e: self.configure(fg=self.default_fg, bg=parent['bg']))
        self.bind('<Button-1>', lambda e: command() if command else None)


class PanelHeader(tk.Frame):
    """Styled panel header with title and actions"""
    def __init__(self, parent, title, icon="", **kwargs):
        super().__init__(parent, bg=ProDesign.BG_DARK, height=36, **kwargs)
        self.pack_propagate(False)
        
        # Title
        title_label = tk.Label(
            self,
            text=f"{icon}  {title}" if icon else title,
            bg=ProDesign.BG_DARK,
            fg=ProDesign.TEXT_PRIMARY,
            font=ProDesign.FONT_SUBHEADING
        )
        title_label.pack(side='left', padx=12, pady=8)
        
        # Separator
        separator = tk.Frame(self, bg=ProDesign.BORDER_SUBTLE, height=1)
        separator.pack(side='bottom', fill='x')


class SceneItem(tk.Frame):
    """Scene/Layer item for left panel"""
    def __init__(self, parent, index, thumbnail=None, title="", subtitle="", 
                 is_protected=False, on_select=None, on_toggle_protect=None):
        super().__init__(parent, bg=ProDesign.BG_DARKER, height=72)
        self.pack_propagate(False)
        
        self.index = index
        self.is_selected = False
        self.is_protected = is_protected
        self.on_select = on_select
        self.on_toggle_protect = on_toggle_protect
        
        # Main container
        container = tk.Frame(self, bg=ProDesign.BG_MEDIUM, bd=1, relief='flat')
        container.pack(fill='both', expand=True, padx=4, pady=2)
        
        # Index badge
        index_label = tk.Label(
            container,
            text=str(index + 1),
            bg=ProDesign.BG_LIGHTER,
            fg=ProDesign.TEXT_TERTIARY,
            font=ProDesign.FONT_SMALL,
            width=3
        )
        index_label.pack(side='left', padx=(8, 4), pady=8)
        
        # Thumbnail
        if thumbnail:
            thumb_label = tk.Label(container, image=thumbnail, bg=ProDesign.BG_MEDIUM)
            thumb_label.image = thumbnail
            thumb_label.pack(side='left', padx=4)
        else:
            placeholder = tk.Label(
                container,
                text="👤",
                bg=ProDesign.BG_LIGHTER,
                fg=ProDesign.TEXT_TERTIARY,
                font=('Segoe UI', 20),
                width=3,
                height=2
            )
            placeholder.pack(side='left', padx=4, pady=8)
        
        # Info
        info_frame = tk.Frame(container, bg=ProDesign.BG_MEDIUM)
        info_frame.pack(side='left', fill='both', expand=True, padx=8)
        
        title_label = tk.Label(
            info_frame,
            text=title,
            bg=ProDesign.BG_MEDIUM,
            fg=ProDesign.TEXT_PRIMARY,
            font=ProDesign.FONT_BODY,
            anchor='w'
        )
        title_label.pack(fill='x', pady=(8, 2))
        
        subtitle_label = tk.Label(
            info_frame,
            text=subtitle,
            bg=ProDesign.BG_MEDIUM,
            fg=ProDesign.TEXT_TERTIARY,
            font=ProDesign.FONT_SMALL,
            anchor='w'
        )
        subtitle_label.pack(fill='x')
        
        # Protection toggle
        self.protect_btn = tk.Label(
            container,
            text="🛡️" if is_protected else "⚪",
            bg=ProDesign.BG_MEDIUM,
            fg=ProDesign.ACCENT_GREEN if is_protected else ProDesign.TEXT_TERTIARY,
            font=('Segoe UI', 14),
            cursor='hand2'
        )
        self.protect_btn.pack(side='right', padx=8)
        self.protect_btn.bind('<Button-1>', lambda e: self._toggle_protect())
        
        # Selection binding
        for widget in [container, index_label, title_label, subtitle_label, info_frame]:
            widget.bind('<Button-1>', lambda e: self._on_click())
        
        self.container = container
    
    def _on_click(self):
        if self.on_select:
            self.on_select(self.index)
    
    def _toggle_protect(self):
        self.is_protected = not self.is_protected
        self.protect_btn.configure(
            text="🛡️" if self.is_protected else "⚪",
            fg=ProDesign.ACCENT_GREEN if self.is_protected else ProDesign.TEXT_TERTIARY
        )
        if self.on_toggle_protect:
            self.on_toggle_protect(self.index, self.is_protected)
    
    def set_selected(self, selected):
        self.is_selected = selected
        if selected:
            self.container.configure(bg=ProDesign.ACCENT_BLUE)
            for widget in self.container.winfo_children():
                if isinstance(widget, tk.Frame):
                    widget.configure(bg=ProDesign.ACCENT_BLUE)
                    for child in widget.winfo_children():
                        if hasattr(child, 'configure'):
                            try:
                                child.configure(bg=ProDesign.ACCENT_BLUE)
                            except:
                                pass
        else:
            self.container.configure(bg=ProDesign.BG_MEDIUM)
            for widget in self.container.winfo_children():
                if isinstance(widget, tk.Frame):
                    widget.configure(bg=ProDesign.BG_MEDIUM)
                    for child in widget.winfo_children():
                        if hasattr(child, 'configure'):
                            try:
                                child.configure(bg=ProDesign.BG_MEDIUM)
                            except:
                                pass


class PropertySlider(tk.Frame):
    """Professional property slider with label and value display"""
    def __init__(self, parent, label, variable, from_=0, to=100, resolution=1, 
                 unit="", command=None, **kwargs):
        super().__init__(parent, bg=ProDesign.BG_DARK, **kwargs)
        
        # Label row
        label_row = tk.Frame(self, bg=ProDesign.BG_DARK)
        label_row.pack(fill='x', pady=(8, 4))
        
        tk.Label(
            label_row,
            text=label,
            bg=ProDesign.BG_DARK,
            fg=ProDesign.TEXT_SECONDARY,
            font=ProDesign.FONT_SMALL
        ).pack(side='left')
        
        self.value_label = tk.Label(
            label_row,
            text=f"{variable.get()}{unit}",
            bg=ProDesign.BG_DARK,
            fg=ProDesign.TEXT_PRIMARY,
            font=ProDesign.FONT_SMALL
        )
        self.value_label.pack(side='right')
        
        # Slider
        self.scale = ttk.Scale(
            self,
            from_=from_,
            to=to,
            orient='horizontal',
            variable=variable,
            command=self._on_change
        )
        self.scale.pack(fill='x', padx=2)
        
        self.variable = variable
        self.unit = unit
        self.resolution = resolution
        self.external_command = command
    
    def _on_change(self, value):
        val = round(float(value) / self.resolution) * self.resolution
        self.variable.set(val)
        self.value_label.configure(text=f"{val:.1f}{self.unit}" if self.resolution < 1 else f"{int(val)}{self.unit}")
        if self.external_command:
            self.external_command(val)


# ============================================================================
# MAIN APPLICATION
# ============================================================================

class AnonVisionPro:
    """Professional AnonVision with modern three-panel interface - YuNet Only"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("AnonVision Professional")
        self.root.geometry("1440x900")
        self.root.configure(bg=ProDesign.BG_DARKEST)
        
        # Apply theme
        ProDesign.apply_theme(root)
        
        # State
        self.current_image = None  # BGR numpy array
        self.original_image = None
        self.current_image_path = None
        self.detected_faces = []
        self.face_protections = []  # List of bool for each face
        self.selected_face_index = None
        self.photo_image = None
        self.scene_items = []
        
        # Settings
        self.yunet_confidence = tk.DoubleVar(value=0.5)
        self.blur_intensity = tk.IntVar(value=45)
        self.blur_passes = tk.IntVar(value=3)
        self.expansion = tk.IntVar(value=20)
        self.show_boxes = tk.BooleanVar(value=True)
        
        # Initialize detector
        self._init_detector()
        
        # Initialize whitelist
        self.whitelist = FaceWhitelist()
        
        # Build interface
        self._build_interface()
        
        # Set initial status
        self._set_status("Ready - YuNet detector initialized")
    
    def _init_detector(self):
        """Initialize YuNet face detector with error handling"""
        if DNNFaceDetector:
            try:
                self.yunet_detector = DNNFaceDetector(prefer_gpu=False)
                print("✓ YuNet detector initialized")
            except Exception as e:
                print(f"ERROR: Failed to initialize YuNet: {e}")
                self.yunet_detector = None
                messagebox.showerror(
                    "Detector Error",
                    f"Failed to initialize YuNet detector:\n{str(e)}\n\n"
                    "Face detection will not work."
                )
        else:
            self.yunet_detector = None
            messagebox.showerror(
                "Missing Detector",
                "dnn_detector.py module not found.\n\n"
                "YuNet detector is unavailable.\n"
                "Face detection will not work."
            )
    
    def _build_interface(self):
        """Build the main interface layout"""
        # Main container
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_columnconfigure(0, weight=0)
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_columnconfigure(2, weight=0)
        
        # Top toolbar
        self._build_toolbar()
        
        # Left panel - Scenes/Faces
        self._build_left_panel()
        
        # Center - Canvas
        self._build_center_canvas()
        
        # Right panel - Properties
        self._build_right_panel()
        
        # Bottom status bar
        self._build_status_bar()
    
    def _build_toolbar(self):
        """Build top toolbar"""
        toolbar = tk.Frame(
            self.root,
            bg=ProDesign.BG_MEDIUM,
            height=ProDesign.TOOLBAR_HEIGHT
        )
        toolbar.grid(row=0, column=0, columnspan=3, sticky='ew')
        toolbar.pack_propagate(False)
        
        # Left section - File operations
        left_section = tk.Frame(toolbar, bg=ProDesign.BG_MEDIUM)
        left_section.pack(side='left', padx=8)
        
        ModernButton(left_section, "Open", "📁", self._load_image, 'default').pack(side='left', padx=2)
        ModernButton(left_section, "Save", "💾", self._save_image, 'default').pack(side='left', padx=2)
        
        # Separator
        tk.Frame(toolbar, bg=ProDesign.BORDER_SUBTLE, width=1).pack(side='left', fill='y', padx=8, pady=8)
        
        # Center section - Main actions
        center_section = tk.Frame(toolbar, bg=ProDesign.BG_MEDIUM)
        center_section.pack(side='left', padx=8)
        
        ModernButton(center_section, "Detect Faces", "🔍", self._detect_faces, 'primary').pack(side='left', padx=2)
        ModernButton(center_section, "Apply Anonymization", "🛡️", self._apply_anonymization, 'success').pack(side='left', padx=2)
        
        # Right section - View options
        right_section = tk.Frame(toolbar, bg=ProDesign.BG_MEDIUM)
        right_section.pack(side='right', padx=16)
        
        tk.Checkbutton(
            right_section,
            text="Show Boxes",
            variable=self.show_boxes,
            bg=ProDesign.BG_MEDIUM,
            fg=ProDesign.TEXT_SECONDARY,
            selectcolor=ProDesign.BG_MEDIUM,
            activebackground=ProDesign.BG_MEDIUM,
            font=ProDesign.FONT_SMALL,
            command=self._update_canvas
        ).pack(side='right', padx=8)
    
    def _build_left_panel(self):
        """Build left panel for scenes/faces"""
        left_panel = tk.Frame(
            self.root,
            bg=ProDesign.BG_DARKER,
            width=ProDesign.SIDEBAR_WIDTH
        )
        left_panel.grid(row=1, column=0, sticky='nsew')
        left_panel.pack_propagate(False)
        
        # Header
        PanelHeader(left_panel, "Detected Faces", "👥")
        
        # Action buttons
        action_frame = tk.Frame(left_panel, bg=ProDesign.BG_DARKER)
        action_frame.pack(fill='x', padx=8, pady=8)
        
        ModernButton(
            action_frame, 
            "Protect All", 
            "✓",
            self._protect_all,
            'ghost',
            height=28
        ).pack(side='left', padx=2, fill='x', expand=True)
        
        ModernButton(
            action_frame,
            "Clear All",
            "✗",
            self._unprotect_all,
            'ghost',
            height=28
        ).pack(side='left', padx=2, fill='x', expand=True)
        
        # Scrollable face list
        list_frame = tk.Frame(left_panel, bg=ProDesign.BG_DARKER)
        list_frame.pack(fill='both', expand=True, padx=4)
        
        # Canvas for scrolling
        self.faces_canvas = tk.Canvas(
            list_frame,
            bg=ProDesign.BG_DARKER,
            highlightthickness=0
        )
        scrollbar = ttk.Scrollbar(list_frame, orient='vertical', command=self.faces_canvas.yview)
        self.faces_inner = tk.Frame(self.faces_canvas, bg=ProDesign.BG_DARKER)
        
        self.faces_inner.bind(
            '<Configure>',
            lambda e: self.faces_canvas.configure(scrollregion=self.faces_canvas.bbox('all'))
        )
        
        self.faces_canvas.create_window((0, 0), window=self.faces_inner, anchor='nw')
        self.faces_canvas.configure(yscrollcommand=scrollbar.set)
        
        self.faces_canvas.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        
        # Initial empty state
        self._show_empty_faces_list()
    
    def _build_center_canvas(self):
        """Build center canvas area"""
        center_frame = tk.Frame(self.root, bg=ProDesign.BG_DARKEST)
        center_frame.grid(row=1, column=1, sticky='nsew')
        center_frame.grid_rowconfigure(0, weight=1)
        center_frame.grid_columnconfigure(0, weight=1)
        
        # Canvas with subtle border
        canvas_container = tk.Frame(
            center_frame,
            bg=ProDesign.BORDER_SUBTLE,
            bd=1
        )
        canvas_container.grid(row=0, column=0, sticky='nsew', padx=8, pady=8)
        canvas_container.grid_rowconfigure(0, weight=1)
        canvas_container.grid_columnconfigure(0, weight=1)
        
        self.main_canvas = tk.Canvas(
            canvas_container,
            bg=ProDesign.BG_DARKER,
            highlightthickness=0
        )
        self.main_canvas.grid(row=0, column=0, sticky='nsew')
        
        # Bind resize
        self.main_canvas.bind('<Configure>', lambda e: self._update_canvas())
        
        # Initial placeholder
        self._show_canvas_placeholder()
    
    def _build_right_panel(self):
        """Build right properties panel"""
        right_panel = tk.Frame(
            self.root,
            bg=ProDesign.BG_DARKER,
            width=ProDesign.PROPERTIES_WIDTH
        )
        right_panel.grid(row=1, column=2, sticky='nsew')
        right_panel.pack_propagate(False)
        
        # Header
        PanelHeader(right_panel, "Properties", "⚙️")
        
        # Scrollable content
        content = tk.Frame(right_panel, bg=ProDesign.BG_DARKER)
        content.pack(fill='both', expand=True)
        
        # Detection Settings Section
        detect_section = tk.Frame(content, bg=ProDesign.BG_DARK)
        detect_section.pack(fill='x', padx=8, pady=8)
        
        tk.Label(
            detect_section,
            text="Detection Settings",
            bg=ProDesign.BG_DARK,
            fg=ProDesign.TEXT_PRIMARY,
            font=ProDesign.FONT_SUBHEADING
        ).pack(anchor='w', padx=8, pady=(8, 0))
        
        PropertySlider(
            detect_section,
            "YuNet Confidence",
            self.yunet_confidence,
            0.1, 0.9, 0.05
        ).pack(fill='x', padx=8)
        
        # Anonymization Settings Section
        anon_section = tk.Frame(content, bg=ProDesign.BG_DARK)
        anon_section.pack(fill='x', padx=8, pady=8)
        
        tk.Label(
            anon_section,
            text="Anonymization Settings",
            bg=ProDesign.BG_DARK,
            fg=ProDesign.TEXT_PRIMARY,
            font=ProDesign.FONT_SUBHEADING
        ).pack(anchor='w', padx=8, pady=(8, 0))
        
        PropertySlider(
            anon_section,
            "Blur Intensity",
            self.blur_intensity,
            15, 99, 2,
            "px"
        ).pack(fill='x', padx=8)
        
        PropertySlider(
            anon_section,
            "Blur Passes",
            self.blur_passes,
            1, 10, 1,
            "x"
        ).pack(fill='x', padx=8)
        
        PropertySlider(
            anon_section,
            "Coverage Expansion",
            self.expansion,
            0, 50, 5,
            "%"
        ).pack(fill='x', padx=8, pady=(0, 8))
        
        # Stats Section
        stats_section = tk.Frame(content, bg=ProDesign.BG_DARK)
        stats_section.pack(fill='x', padx=8, pady=8)
        
        tk.Label(
            stats_section,
            text="Statistics",
            bg=ProDesign.BG_DARK,
            fg=ProDesign.TEXT_PRIMARY,
            font=ProDesign.FONT_SUBHEADING
        ).pack(anchor='w', padx=8, pady=(8, 4))
        
        self.stats_label = tk.Label(
            stats_section,
            text="No image loaded",
            bg=ProDesign.BG_DARK,
            fg=ProDesign.TEXT_TERTIARY,
            font=ProDesign.FONT_SMALL,
            justify='left'
        )
        self.stats_label.pack(anchor='w', padx=8, pady=(4, 8))
    
    def _build_status_bar(self):
        """Build bottom status bar"""
        status_bar = tk.Frame(
            self.root,
            bg=ProDesign.BG_MEDIUM,
            height=ProDesign.STATUS_HEIGHT
        )
        status_bar.grid(row=2, column=0, columnspan=3, sticky='ew')
        status_bar.pack_propagate(False)
        
        self.status_label = tk.Label(
            status_bar,
            text="Ready",
            bg=ProDesign.BG_MEDIUM,
            fg=ProDesign.TEXT_TERTIARY,
            font=ProDesign.FONT_SMALL
        )
        self.status_label.pack(side='left', padx=16, pady=4)
        
        # Right side info
        self.info_label = tk.Label(
            status_bar,
            text="YuNet Detector",
            bg=ProDesign.BG_MEDIUM,
            fg=ProDesign.TEXT_TERTIARY,
            font=ProDesign.FONT_SMALL
        )
        self.info_label.pack(side='right', padx=16, pady=4)
    
    # ============ FUNCTIONALITY ============
    
    def _load_image(self):
        """Load an image file"""
        filepath = filedialog.askopenfilename(
            title="Open Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp"),
                ("All files", "*.*")
            ]
        )
        
        if not filepath:
            return
        
        try:
            # Validate path
            if not validate_image_path(filepath):
                messagebox.showerror("Invalid File", "Selected file is not a valid image")
                return
            
            # Load image
            self.current_image = cv2.imread(filepath)
            if self.current_image is None:
                raise ValueError("Failed to load image")
            
            self.original_image = self.current_image.copy()
            self.current_image_path = filepath
            self.detected_faces = []
            self.face_protections = []
            self.selected_face_index = None
            
            # Clear faces list
            self._show_empty_faces_list()
            
            # Update canvas
            self._update_canvas()
            
            # Update stats
            h, w = self.current_image.shape[:2]
            self._update_stats(f"Image: {Path(filepath).name}\nSize: {w}×{h} pixels")
            
            self._set_status(f"Loaded: {Path(filepath).name}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image:\n{str(e)}")
    
    def _detect_faces(self):
        """Detect faces in current image using YuNet"""
        if self.current_image is None:
            messagebox.showinfo("No Image", "Please load an image first")
            return
        
        if not self.yunet_detector:
            messagebox.showerror(
                "No Detector",
                "YuNet detector is not available.\n\n"
                "Please ensure dnn_detector.py is properly installed."
            )
            return
        
        self._set_status("Detecting faces with YuNet...")
        self.root.update()
        
        try:
            # YuNet detection
            _, faces = self.yunet_detector.detect_faces(
                self.current_image,
                confidence_threshold=self.yunet_confidence.get()
            )
            
            self.detected_faces = faces
            
        except Exception as e:
            messagebox.showerror("Detection Error", f"Face detection failed:\n{str(e)}")
            self.detected_faces = []
            return
        
        # Initialize protections
        self.face_protections = [False] * len(self.detected_faces)
        
        # Reset image
        self.current_image = self.original_image.copy()
        
        # Update UI
        self._populate_faces_list()
        self._update_canvas()
        self._set_status(f"Detected {len(self.detected_faces)} face(s)")
        self._update_stats_with_faces()
        
        if len(self.detected_faces) == 0:
            messagebox.showinfo("No Faces", "No faces were detected in the image.")
    
    def _apply_anonymization(self):
        """Apply blur to unprotected faces"""
        if self.original_image is None or not self.detected_faces:
            messagebox.showinfo("No Faces", "Please detect faces first")
            return
        
        self._set_status("Applying anonymization...")
        self.root.update()
        
        result = self.original_image.copy()
        protected_count = 0
        blurred_count = 0
        
        blur_size = self.blur_intensity.get()
        if blur_size % 2 == 0:
            blur_size += 1
        passes = self.blur_passes.get()
        expansion = self.expansion.get() / 100.0
        
        for idx, (x, y, w, h) in enumerate(self.detected_faces):
            if self.face_protections[idx]:
                protected_count += 1
                continue
            
            # Expand region
            ex = max(0, int(x - w * expansion))
            ey = max(0, int(y - h * expansion))
            ew = min(result.shape[1] - ex, int(w * (1 + 2 * expansion)))
            eh = min(result.shape[0] - ey, int(h * (1 + 2 * expansion)))
            
            # Extract and blur
            roi = result[ey:ey+eh, ex:ex+ew]
            for _ in range(passes):
                roi = cv2.GaussianBlur(roi, (blur_size, blur_size), 0)
            result[ey:ey+eh, ex:ex+ew] = roi
            
            blurred_count += 1
        
        self.current_image = result
        self._update_canvas()
        
        self._set_status(f"Anonymized {blurred_count} face(s), protected {protected_count}")
        messagebox.showinfo(
            "Complete",
            f"✓ Anonymized: {blurred_count} face(s)\n"
            f"🛡️ Protected: {protected_count} face(s)"
        )
    
    def _save_image(self):
        """Save the current image"""
        if self.current_image is None:
            messagebox.showinfo("No Image", "Nothing to save")
            return
        
        filepath = filedialog.asksaveasfilename(
            title="Save Image",
            defaultextension=".jpg",
            filetypes=[
                ("JPEG", "*.jpg"),
                ("PNG", "*.png"),
                ("All files", "*.*")
            ]
        )
        
        if filepath:
            try:
                cv2.imwrite(filepath, self.current_image)
                self._set_status(f"Saved: {Path(filepath).name}")
                messagebox.showinfo("Saved", f"Image saved to:\n{filepath}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save image:\n{str(e)}")
    
    def _show_empty_faces_list(self):
        """Show empty state in faces list"""
        for widget in self.faces_inner.winfo_children():
            widget.destroy()
        
        empty_label = tk.Label(
            self.faces_inner,
            text="No faces detected\n\nLoad an image and\nclick 'Detect Faces'",
            bg=ProDesign.BG_DARKER,
            fg=ProDesign.TEXT_TERTIARY,
            font=ProDesign.FONT_SMALL,
            justify='center'
        )
        empty_label.pack(expand=True, pady=32)
    
    def _populate_faces_list(self):
        """Populate faces list with detected faces"""
        # Clear existing
        for widget in self.faces_inner.winfo_children():
            widget.destroy()
        self.scene_items = []
        
        for idx, (x, y, w, h) in enumerate(self.detected_faces):
            # Create thumbnail
            try:
                face_img = self.current_image[y:y+h, x:x+w]
                face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                face_pil = Image.fromarray(face_rgb)
                face_pil.thumbnail((48, 48))
                thumbnail = ImageTk.PhotoImage(face_pil)
            except:
                thumbnail = None
            
            # Create scene item
            item = SceneItem(
                self.faces_inner,
                idx,
                thumbnail,
                f"Face {idx + 1}",
                f"{w}×{h} pixels",
                self.face_protections[idx],
                self._select_face,
                self._toggle_face_protection
            )
            item.pack(fill='x', pady=2)
            self.scene_items.append(item)
    
    def _select_face(self, index):
        """Select a face from the list"""
        self.selected_face_index = index
        
        # Update selection visual
        for i, item in enumerate(self.scene_items):
            item.set_selected(i == index)
        
        self._update_canvas()
    
    def _toggle_face_protection(self, index, protected):
        """Toggle protection status of a face"""
        if index < len(self.face_protections):
            self.face_protections[index] = protected
            self._update_canvas()
            self._update_stats_with_faces()
    
    def _protect_all(self):
        """Protect all faces"""
        self.face_protections = [True] * len(self.detected_faces)
        for item in self.scene_items:
            item.is_protected = True
            item.protect_btn.configure(
                text="🛡️",
                fg=ProDesign.ACCENT_GREEN
            )
        self._update_canvas()
        self._update_stats_with_faces()
    
    def _unprotect_all(self):
        """Unprotect all faces"""
        self.face_protections = [False] * len(self.detected_faces)
        for item in self.scene_items:
            item.is_protected = False
            item.protect_btn.configure(
                text="⚪",
                fg=ProDesign.TEXT_TERTIARY
            )
        self._update_canvas()
        self._update_stats_with_faces()
    
    def _show_canvas_placeholder(self):
        """Show placeholder in canvas"""
        self.main_canvas.delete('all')
        w = self.main_canvas.winfo_width()
        h = self.main_canvas.winfo_height()
        
        if w > 100:  # Canvas initialized
            self.main_canvas.create_text(
                w // 2, h // 2,
                text="Drop an image here or click 'Open'",
                fill=ProDesign.TEXT_TERTIARY,
                font=ProDesign.FONT_HEADING
            )
    
    def _update_canvas(self):
        """Update the main canvas display"""
        if self.current_image is None:
            self._show_canvas_placeholder()
            return
        
        # Get canvas dimensions
        canvas_w = max(100, self.main_canvas.winfo_width())
        canvas_h = max(100, self.main_canvas.winfo_height())
        
        # Create display image
        display_img = self.current_image.copy()
        
        # Draw boxes if enabled
        if self.show_boxes.get() and self.detected_faces:
            expansion = self.expansion.get() / 100.0
            
            for idx, (x, y, w, h) in enumerate(self.detected_faces):
                is_protected = self.face_protections[idx] if idx < len(self.face_protections) else False
                is_selected = idx == self.selected_face_index
                
                # Main box color
                if is_protected:
                    color = (128, 222, 74)  # Green
                else:
                    color = (113, 135, 248)  # Blue
                
                thickness = 3 if is_selected else 2
                
                # Draw expansion area if not protected
                if not is_protected and expansion > 0:
                    ex = max(0, int(x - w * expansion))
                    ey = max(0, int(y - h * expansion))
                    ew = min(display_img.shape[1] - ex, int(w * (1 + 2 * expansion)))
                    eh = min(display_img.shape[0] - ey, int(h * (1 + 2 * expansion)))
                    cv2.rectangle(display_img, (ex, ey), (ex + ew, ey + eh), (100, 100, 100), 1)
                
                # Draw main box
                cv2.rectangle(display_img, (x, y), (x + w, y + h), color, thickness)
                
                # Draw label
                label = f"Face {idx + 1}"
                if is_protected:
                    label += " 🛡️"
                
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                label_y = y - 8 if y > 20 else y + h + 16
                
                cv2.rectangle(
                    display_img,
                    (x, label_y - label_size[1] - 4),
                    (x + label_size[0] + 8, label_y + 2),
                    color,
                    -1
                )
                cv2.putText(
                    display_img,
                    label,
                    (x + 4, label_y - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1
                )
        
        # Convert to PIL
        rgb = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        
        # Resize to fit canvas
        img_w, img_h = pil_img.size
        scale = min(canvas_w / img_w, canvas_h / img_h, 1.0)
        new_w = int(img_w * scale)
        new_h = int(img_h * scale)
        
        if scale < 1.0:
            pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)
        
        # Convert to PhotoImage
        self.photo_image = ImageTk.PhotoImage(pil_img)
        
        # Display
        self.main_canvas.delete('all')
        self.main_canvas.create_image(
            canvas_w // 2, canvas_h // 2,
            image=self.photo_image
        )
    
    def _set_status(self, message):
        """Update status bar"""
        self.status_label.configure(text=message)
        timestamp = time.strftime("%H:%M:%S")
        self.info_label.configure(text=f"YuNet | {timestamp}")
    
    def _update_stats(self, text):
        """Update statistics display"""
        self.stats_label.configure(text=text)
    
    def _update_stats_with_faces(self):
        """Update stats with face information"""
        if self.current_image is None:
            return
        
        h, w = self.current_image.shape[:2]
        total = len(self.detected_faces)
        protected = sum(self.face_protections)
        to_blur = total - protected
        
        stats = f"Image: {Path(self.current_image_path).name if self.current_image_path else 'Untitled'}\n"
        stats += f"Size: {w}×{h} pixels\n"
        stats += f"─────────────────\n"
        stats += f"Total Faces: {total}\n"
        stats += f"Protected: {protected} 🛡️\n"
        stats += f"To Anonymize: {to_blur}"
        
        self._update_stats(stats)


# ============================================================================
# ENTRY POINT
# ============================================================================

def main():
    """Main entry point"""
    root = tk.Tk()
    
    # Windows DPI awareness
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except:
        pass
    
    app = AnonVisionPro(root)
    root.mainloop()


if __name__ == "__main__":
    print("=" * 70)
    print("AnonVision Professional - YuNet Only")
    print("=" * 70)
    main()