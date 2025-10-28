#DNN-based face detection for AnonVision


import cv2
import os
import urllib.request
import numpy as np


class DNNFaceDetector:
    """DNN-based face detector using OpenCV's pre-trained model"""
    
    def __init__(self):
        """Initialize DNN face detector"""
        self.model_file = None
        self.config_file = None
        self.net = None
        
        # Download and load models
        self._download_models()
        self._load_model()
        
        print("DNN face detector initialized successfully")
    
    def _download_models(self):
        """Download DNN face detection models if not present"""
        # These are the correct working URLs
        models = {
            "res10_300x300_ssd_iter_140000.caffemodel": {
                "url": "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
                "description": "caffe_model"
            },
            "deploy.prototxt": {
                "url": "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
                "description": "config_file"
            }
        }
        
        data_dir = "data"
        os.makedirs(data_dir, exist_ok=True)
        
        for filename, info in models.items():
            filepath = os.path.join(data_dir, filename)
            
            if not os.path.exists(filepath):
                print(f"Downloading {info['description']} for DNN detector...")
                try:
                    # Add headers to avoid being blocked
                    req = urllib.request.Request(
                        info['url'],
                        headers={'User-Agent': 'Mozilla/5.0'}
                    )
                    with urllib.request.urlopen(req) as response, open(filepath, 'wb') as out_file:
                        out_file.write(response.read())
                    print(f"Downloaded: {filepath}")
                except Exception as e:
                    print(f"Error downloading {filename}: {e}")
                    print(f"Tried URL: {info['url']}")
                    raise
            else:
                print(f"DNN model already exists: {filepath}")
            
            # Store file paths
            if filename.endswith('.caffemodel'):
                self.model_file = filepath
            elif filename.endswith('.prototxt'):
                self.config_file = filepath
    
    def _load_model(self):
        """Load the DNN model"""
        try:
            # Use Caffe model instead of TensorFlow
            self.net = cv2.dnn.readNetFromCaffe(self.config_file, self.model_file)
            print("DNN model loaded successfully")
        except Exception as e:
            print(f"Error loading DNN model: {e}")
            raise
    
    def detect_faces(self, image_path, confidence_threshold=0.5):
        """
        Detect faces using DNN model
        
        Args:
            image_path (str): Path to image file
            confidence_threshold (float): Minimum confidence for detection (0.0-1.0)
            
        Returns:
            tuple: (image, faces) where faces is list of (x, y, w, h) tuples
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        h, w = image.shape[:2]
        
        # Create blob from image
        blob = cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)), 
            1.0, 
            (300, 300), 
            (104.0, 177.0, 123.0)
        )
        
        # Set input and run forward pass
        self.net.setInput(blob)
        detections = self.net.forward()
        
        # Extract face coordinates
        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence > confidence_threshold:
                # Get bounding box coordinates
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype("int")
                
                # Convert to (x, y, w, h) format
                face_x = max(0, x1)
                face_y = max(0, y1)
                face_w = min(w - face_x, x2 - x1)
                face_h = min(h - face_y, y2 - y1)
                
                faces.append((face_x, face_y, face_w, face_h))
        
        print(f"DNN detected {len(faces)} face(s) in {os.path.basename(image_path)}")
        
        return image, faces
    
    def draw_detection_boxes(self, image, faces, color=(0, 255, 0), thickness=2, show_label=True):
        """
        Draw rectangles around detected faces
        Same interface as Haar Cascade detector for compatibility
        """
        image_copy = image.copy()
        
        for i, (x, y, w, h) in enumerate(faces):
            # Draw rectangle
            cv2.rectangle(image_copy, (x, y), (x+w, y+h), color, thickness)
            
            # Draw label if requested
            if show_label:
                label = f"Face {i+1}"
                label_y = y - 10 if y - 10 > 10 else y + h + 20
                
                (label_width, label_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                
                cv2.rectangle(
                    image_copy,
                    (x, label_y - label_height - baseline),
                    (x + label_width, label_y),
                    color,
                    -1
                )
                
                cv2.putText(
                    image_copy,
                    label,
                    (x, label_y - baseline),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1
                )
        
        return image_copy


if __name__ == "__main__":
    # Test DNN detector
    print("Testing DNN face detector...")
    detector = DNNFaceDetector()
    print("DNN detector is ready!")