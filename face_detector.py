## Face Detection Module for AnonVision

import cv2
import os
import urllib.request

class FaceDetector:
    """Handles face detection using Haar Cascade classifier"""
    
    def __init__(self):
        """Initialize the face detector with Haar Cascade classifier"""
        # Ensure we have the Haar Cascade file
        cascade_path = self._get_cascade_path()
        
        # Load the cascade classifier
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        if self.face_cascade.empty():
            raise IOError("Could not load Haar Cascade classifier")
        
        print("Face detector initialized successfully")
    
    def _get_cascade_path(self):
        """Get or download Haar Cascade file"""
        # Create data directory if needed
        os.makedirs('data', exist_ok=True)
        
        cascade_path = os.path.join('data', 'haarcascade_frontalface_default.xml')
        
        # If file doesn't exist, download it
        if not os.path.exists(cascade_path):
            print("Haar Cascade file not found. Downloading...")
            self._download_cascade(cascade_path)
        
        return cascade_path
    
    def _download_cascade(self, cascade_path):
        """Download Haar Cascade XML file from OpenCV repository"""
        url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
        
        try:
            urllib.request.urlretrieve(url, cascade_path)
            print(f"Successfully downloaded Haar Cascade to {cascade_path}")
        except Exception as e:
            print(f"Error downloading Haar Cascade: {e}")
            # Fall back to OpenCV's built-in data
            return cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    
    def detect_faces(self, image_path):
        """
        Detect faces in an image
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            tuple: (image, faces) where faces is a list of (x, y, w, h) tuples
        """
        # Read the image
        image = cv2.imread(image_path)
        
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Convert to grayscale for detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        print(f"Detected {len(faces)} face(s) in {os.path.basename(image_path)}")
        
        return image, faces
    
    def draw_detection_boxes(self, image, faces, color=(0, 255, 0), thickness=2):
        """
        Draw rectangles around detected faces
        
        Args:
            image: The image array
            faces: List of (x, y, w, h) tuples
            color: BGR color tuple (default: green)
            thickness: Line thickness in pixels
            
        Returns:
            image: Image with detection boxes drawn
        """
        image_copy = image.copy()
        
        for (x, y, w, h) in faces:
            cv2.rectangle(image_copy, (x, y), (x+w, y+h), color, thickness)
        
        return image_copy


if __name__ == "__main__":
    # Test the face detector
    print("Testing face detector...")
    detector = FaceDetector()
    print("Face detector is ready!")