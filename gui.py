##GUI Module for AnonVision

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import os
import cv2
import numpy as np
from face_detector import FaceDetector
from utils import validate_image_path, resize_image_for_display
from dnn_detector import DNNFaceDetector


class AnonVisionGUI:
    """Main GUI class for AnonVision application"""
    
    def __init__(self, root):
        """Initialize the GUI"""
        self.root = root
        self.root.title("AnonVision - Face Privacy Protection")
        self.root.geometry("900x700")
        self.root.resizable(True, True)
        
        # Set minimum window size
        self.root.minsize(800, 600)
        
        # Variables
        self.current_image_path = None
        self.current_image = None
        self.display_image = None
        self.detected_faces = []
        self.current_image_array = None
        self.photo_image = None
        
        # Initialize face detector
        try:
            self.face_detector = FaceDetector()
        except Exception as e:
            messagebox.showerror(
                "Initialization Error",
                f"Could not initialize face detector:\n{str(e)}"
            )
            self.face_detector = None
            
        # Initialize DNN detector (optional, more accurate)
        self.dnn_detector = None
        try:
            self.dnn_detector = DNNFaceDetector()
        except Exception as e:
            print(f"DNN detector not available: {e}")
        
        # Detection settings
        self.detection_method = tk.StringVar(value="haar")  # 'haar' or 'dnn'
        self.scale_factor = tk.DoubleVar(value=1.1)
        self.min_neighbors = tk.IntVar(value=5)
        self.min_size = tk.IntVar(value=30)
        self.dnn_confidence = tk.DoubleVar(value=0.5)
        
        # Create GUI elements
        self._create_widgets()
        
        print("GUI initialized successfully")
    
    def _create_widgets(self):
        """Create and layout all GUI widgets"""
        
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights for responsiveness
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Top button frame
        self._create_top_buttons(main_frame)
        
        # Image display area
        self._create_image_display(main_frame)
        
        # Advanced settings panel
        self._create_advanced_settings(main_frame)
        
        # Status bar
        self._create_status_bar(main_frame)
    
    def _create_top_buttons(self, parent):
        """Create top button panel"""
        button_frame = ttk.Frame(parent)
        button_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Select Image button
        self.btn_select = ttk.Button(
            button_frame,
            text="Select Image",
            command=self.select_image
        )
        self.btn_select.pack(side=tk.LEFT, padx=5)
        
        # Save button (disabled initially)
        self.btn_save = ttk.Button(
            button_frame,
            text="Save Protected Image",
            command=self.save_image,
            state=tk.DISABLED
        )
        self.btn_save.pack(side=tk.LEFT, padx=5)
    
    def _create_image_display(self, parent):
        """Create image display area"""
        # Frame for image display
        display_frame = ttk.LabelFrame(parent, text="Image Preview", padding="5")
        display_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        display_frame.columnconfigure(0, weight=1)
        display_frame.rowconfigure(0, weight=1)
        
        # Canvas for image
        self.canvas = tk.Canvas(
            display_frame,
            bg='gray85',
            highlightthickness=1,
            highlightbackground='gray'
        )
        self.canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Bind resize event
        self.canvas.bind('<Configure>', self._on_canvas_configure)
        
        # Add scrollbars
        v_scrollbar = ttk.Scrollbar(display_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        v_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        h_scrollbar = ttk.Scrollbar(display_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        h_scrollbar.grid(row=1, column=0, sticky=(tk.W, tk.E))
        
        self.canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        # Placeholder text
        self.canvas.create_text(
            450, 300,
            text="No image loaded\n\nClick 'Select Image' to begin",
            font=('Arial', 14),
            fill='gray50',
            tags='placeholder'
        )
    
    def _create_status_bar(self, parent):
        """Create status bar at bottom"""
        status_frame = ttk.Frame(parent)
        status_frame.grid(row=2, column=0, sticky=(tk.W, tk.E))
        
        # Status label
        self.status_label = ttk.Label(
            status_frame,
            text="Ready",
            relief=tk.SUNKEN,
            anchor=tk.W
        )
        self.status_label.pack(fill=tk.X, padx=2, pady=2)
        
    def _create_advanced_settings(self, parent):
        """Create collapsible advanced detection settings panel"""
        # Main container
        settings_container = ttk.Frame(parent)
        settings_container.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # Toggle button
        self.show_advanced = tk.BooleanVar(value=False)
        self.btn_toggle_advanced = ttk.Button(
            settings_container,
            text="▶ Show Advanced Detection Settings",
            command=self._toggle_advanced_settings
        )
        self.btn_toggle_advanced.pack(fill=tk.X, pady=(0, 5))
        
        # Settings frame (initially hidden)
        self.settings_frame = ttk.LabelFrame(
            settings_container,
            text="Advanced Detection Settings",
            padding="10"
        )
        
        # Detection method selection
        method_frame = ttk.Frame(self.settings_frame)
        method_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(method_frame, text="Detection Method:").pack(side=tk.LEFT, padx=5)
        
        ttk.Radiobutton(
            method_frame,
            text="Haar Cascade (Fast)",
            variable=self.detection_method,
            value="haar",
            command=self._on_detection_method_change
        ).pack(side=tk.LEFT, padx=5)
        
        if self.dnn_detector:
            ttk.Radiobutton(
                method_frame,
                text="DNN (More Accurate)",
                variable=self.detection_method,
                value="dnn",
                command=self._on_detection_method_change
            ).pack(side=tk.LEFT, padx=5)
        
        # Haar Cascade parameters
        self.haar_params_frame = ttk.Frame(self.settings_frame)
        self.haar_params_frame.pack(fill=tk.X, pady=5)
        
        # Scale Factor
        ttk.Label(self.haar_params_frame, text="Scale Factor:").grid(row=0, column=0, padx=5, sticky=tk.W)
        scale_spinbox = ttk.Spinbox(
            self.haar_params_frame,
            from_=1.01,
            to=2.0,
            increment=0.01,
            textvariable=self.scale_factor,
            width=10
        )
        scale_spinbox.grid(row=0, column=1, padx=5)
        ttk.Label(self.haar_params_frame, text="(1.01-2.0, lower=more faces)").grid(row=0, column=2, padx=5)
        
        # Min Neighbors
        ttk.Label(self.haar_params_frame, text="Min Neighbors:").grid(row=1, column=0, padx=5, sticky=tk.W)
        neighbors_spinbox = ttk.Spinbox(
            self.haar_params_frame,
            from_=1,
            to=10,
            increment=1,
            textvariable=self.min_neighbors,
            width=10
        )
        neighbors_spinbox.grid(row=1, column=1, padx=5)
        ttk.Label(self.haar_params_frame, text="(1-10, higher=fewer false positives)").grid(row=1, column=2, padx=5)
        
        # Min Size
        ttk.Label(self.haar_params_frame, text="Min Size:").grid(row=2, column=0, padx=5, sticky=tk.W)
        size_spinbox = ttk.Spinbox(
            self.haar_params_frame,
            from_=10,
            to=100,
            increment=5,
            textvariable=self.min_size,
            width=10
        )
        size_spinbox.grid(row=2, column=1, padx=5)
        ttk.Label(self.haar_params_frame, text="(10-100 pixels)").grid(row=2, column=2, padx=5)
        
        # DNN parameters
        self.dnn_params_frame = ttk.Frame(self.settings_frame)
        
        ttk.Label(self.dnn_params_frame, text="Confidence Threshold:").grid(row=0, column=0, padx=5, sticky=tk.W)
        conf_spinbox = ttk.Spinbox(
            self.dnn_params_frame,
            from_=0.1,
            to=0.95,
            increment=0.05,
            textvariable=self.dnn_confidence,
            width=10
        )
        conf_spinbox.grid(row=0, column=1, padx=5)
        ttk.Label(self.dnn_params_frame, text="(0.1-0.95, higher=more confident detections)").grid(row=0, column=2, padx=5)
        
        # Re-detect button
        ttk.Button(
            self.settings_frame,
            text="Re-detect Faces",
            command=self._redetect_faces
        ).pack(pady=10)
        
        # Show appropriate parameters
        self._on_detection_method_change()
    
    def _toggle_advanced_settings(self):
        """Toggle visibility of advanced settings"""
        if self.show_advanced.get():
            # Hide settings
            self.settings_frame.pack_forget()
            self.btn_toggle_advanced.config(text="▶ Show Advanced Detection Settings")
            self.show_advanced.set(False)
        else:
            # Show settings
            self.settings_frame.pack(fill=tk.X, pady=5)
            self.btn_toggle_advanced.config(text="▼ Hide Advanced Detection Settings")
            self.show_advanced.set(True)
    
    def _on_detection_method_change(self):
        """Handle detection method change"""
        if self.detection_method.get() == "haar":
            self.haar_params_frame.pack(fill=tk.X, pady=5)
            self.dnn_params_frame.pack_forget()
        else:
            self.haar_params_frame.pack_forget()
            self.dnn_params_frame.pack(fill=tk.X, pady=5)
    
    def _redetect_faces(self):
        """Re-run face detection with current parameters"""
        if not self.current_image_path:
            messagebox.showinfo("No Image", "Please load an image first")
            return
        
        # Re-detect with current settings
        self._detect_faces_in_image(self.current_image_path)
        
        # Refresh display
        self._load_and_display_image(self.current_image_path)
        
        self.update_status(f"Re-detected faces: {len(self.detected_faces)} face(s) found")
    
    def select_image(self):
        """Open file dialog to select an image"""
        # File dialog to select image
        filetypes = [
            ('Image files', '*.jpg *.jpeg *.png *.bmp'),
            ('JPEG files', '*.jpg *.jpeg'),
            ('PNG files', '*.png'),
            ('BMP files', '*.bmp'),
            ('All files', '*.*')
        ]
        
        filepath = filedialog.askopenfilename(
            title="Select an image",
            filetypes=filetypes,
            initialdir=os.getcwd()
        )
        
        if filepath:
            # Validate the selected file
            if not self._validate_image(filepath):
                messagebox.showerror(
                    "Invalid File",
                    "Please select a valid image file (JPG, PNG, or BMP)"
                )
                return
            
            # Load and display the image
            self._load_and_display_image(filepath)
        else:
            self.update_status("No image selected")
    
    def _validate_image(self, filepath):
        """
        Validate if selected file is a valid image
        
        Args:
            filepath (str): Path to file
            
        Returns:
            bool: True if valid image
        """
        if not validate_image_path(filepath):
            return False
        
        # Try to open the image
        try:
            test_image = Image.open(filepath)
            test_image.verify()  # Verify it's actually an image
            return True
        except Exception as e:
            print(f"Error validating image: {e}")
            return False
    
    def _load_and_display_image(self, filepath):
        """
        Load an image and display it on the canvas
        
        Args:
            filepath (str): Path to image file
        """
        try:
            # Load image
            self.current_image_path = filepath
            self.current_image = Image.open(filepath)
            
            # Detect faces
            self._detect_faces_in_image(filepath)
            
            # Get canvas dimensions
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            
            # Use default size if canvas not yet rendered
            if canvas_width <= 1:
                canvas_width = 800
                canvas_height = 500
            
            # Resize image for display
            self.display_image = resize_image_for_display(
                self.current_image,
                max_width=canvas_width - 20,
                max_height=canvas_height - 20
            )
            
            # Convert to PhotoImage
            self.photo_image = ImageTk.PhotoImage(self.display_image)
            
            # Clear canvas
            self.canvas.delete('all')
            
            # Display image
            self.canvas.create_image(
                canvas_width // 2,
                canvas_height // 2,
                image=self.photo_image,
                anchor=tk.CENTER,
                tags='image'
            )
            
            # Update canvas scroll region
            self.canvas.configure(scrollregion=self.canvas.bbox('all'))
            
            # Update status with face count
            filename = os.path.basename(filepath)
            img_width, img_height = self.current_image.size
            face_count = len(self.detected_faces)
            
            status_text = f"Loaded: {filename} ({img_width}x{img_height}) | {face_count} face(s) detected"
            self.update_status(status_text)
            
            print(f"Successfully loaded and displayed: {filepath}")
            
        except Exception as e:
            messagebox.showerror(
                "Error Loading Image",
                f"Could not load image:\n{str(e)}"
            )
            print(f"Error loading image: {e}")
    
    def _detect_faces_in_image(self, filepath):
        """
        Detect faces in the loaded image using selected method
        
        Args:
            filepath (str): Path to image file
        """
        try:
            if self.detection_method.get() == "dnn" and self.dnn_detector:
                # Use DNN detector
                image_array, faces = self.dnn_detector.detect_faces(
                    filepath,
                    confidence_threshold=self.dnn_confidence.get()
                )
                self.detected_faces = faces
                self.current_image_array = image_array
                print(f"DNN detected {len(faces)} face(s)")
            else:
                # Use Haar Cascade with custom parameters
                if not self.face_detector:
                    self.detected_faces = []
                    return
                
                # Read image
                image_array = cv2.imread(filepath)
                gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
                
                # Detect with custom parameters
                faces = self.face_detector.face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=self.scale_factor.get(),
                    minNeighbors=self.min_neighbors.get(),
                    minSize=(self.min_size.get(), self.min_size.get())
                )
                
                self.detected_faces = faces
                self.current_image_array = image_array
                print(f"Haar Cascade detected {len(faces)} face(s)")
            
        except Exception as e:
            print(f"Error detecting faces: {e}")
            self.detected_faces = []
    
    def _on_canvas_configure(self, event):
        """Handle canvas resize events"""
        if self.current_image_path and event.width > 1:
            # Reload image with new size
            self._load_and_display_image(self.current_image_path)
    
    def save_image(self):
        """Placeholder for save image functionality"""
        self.update_status("Save image clicked - functionality to be added")
        print("Save Image button clicked")
    
    def update_status(self, message):
        """Update status bar message"""
        self.status_label.config(text=message)
        self.root.update_idletasks()


def main():
    """Main function to run the application"""
    root = tk.Tk()
    app = AnonVisionGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()