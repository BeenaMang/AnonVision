### AnonVision - GUI Module



import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import os


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
        """Placeholder for select image functionality"""
        self.update_status("Select image clicked - functionality to be added")
        print("Select Image button clicked")
    
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