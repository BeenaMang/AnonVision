
##GUI Module for AnonVision



import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import os
import cv2
import numpy as np
from face_detector import FaceDetector
from utils import validate_image_path, resize_image_for_display


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
        Detect faces in the loaded image
        
        Args:
            filepath (str): Path to image file
        """
        if not self.face_detector:
            self.detected_faces = []
            return
        
        try:
            # Detect faces
            image_array, faces = self.face_detector.detect_faces(filepath)
            self.detected_faces = faces
            self.current_image_array = image_array
            
            print(f"Detected {len(faces)} face(s)")
            
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