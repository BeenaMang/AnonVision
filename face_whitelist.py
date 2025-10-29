##Face Recognition Module for AnonVision


import os
import pickle
import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
from pathlib import Path

# Try to import face_recognition, fallback to basic OpenCV features if not available
try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
    print("✓ face_recognition library available (high accuracy)")
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False
    print("⚠ face_recognition not available, using OpenCV features (lower accuracy)")
    print("  Install with: pip install face_recognition")


class FaceWhitelist:
    """
    Manages a whitelist of faces that should NOT be anonymized.
    Uses face embeddings for robust matching.
    """
    
    def __init__(self, storage_dir: str = "data/whitelist"):
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)
        
        self.whitelist_file = os.path.join(self.storage_dir, "whitelist.pkl")
        
        # Storage: {name: {encoding: np.array, thumbnail_path: str}}
        self.whitelisted_faces: Dict[str, Dict] = {}
        
        self.use_face_recognition = FACE_RECOGNITION_AVAILABLE
        
        # Load existing whitelist
        self._load_whitelist()
        
        print(f"Face whitelist initialized ({len(self.whitelisted_faces)} protected faces)")
    
    def add_face(self, image_path: str, name: str) -> bool:
        """
        Add a face to the whitelist from an image.
        
        Args:
            image_path: Path to image containing the face
            name: Identifier for this person
            
        Returns:
            True if face was added successfully, False otherwise
        """
        try:
            print(f"Adding face to whitelist: {name}")
            
            # Load image
            if self.use_face_recognition:
                image = face_recognition.load_image_file(image_path)
                
                # Get face encodings
                face_encodings = face_recognition.face_encodings(image)
                
                if len(face_encodings) == 0:
                    print(f"  ✗ No face detected in image")
                    return False
                
                if len(face_encodings) > 1:
                    print(f"  ⚠ Multiple faces detected, using the first one")
                
                encoding = face_encodings[0]
                
            else:
                # Fallback: Use OpenCV ORB features
                image = cv2.imread(image_path)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                
                # Detect face
                face_cascade = cv2.CascadeClassifier(
                    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                )
                faces = face_cascade.detectMultiScale(gray, 1.1, 5)
                
                if len(faces) == 0:
                    print(f"  ✗ No face detected in image")
                    return False
                
                # Use first face
                x, y, w, h = faces[0]
                face_region = gray[y:y+h, x:x+w]
                face_region = cv2.resize(face_region, (100, 100))
                
                # Compute ORB features as "encoding"
                orb = cv2.ORB_create()
                keypoints, descriptors = orb.detectAndCompute(face_region, None)
                
                if descriptors is None:
                    print(f"  ✗ Could not extract features from face")
                    return False
                
                encoding = descriptors.flatten()[:128]  # Use first 128 features
                
                # Pad if needed
                if len(encoding) < 128:
                    encoding = np.pad(encoding, (0, 128 - len(encoding)))
            
            # Save thumbnail
            thumbnail_path = os.path.join(self.storage_dir, f"{name}_thumb.jpg")
            self._save_thumbnail(image_path, thumbnail_path)
            
            # Store in whitelist
            self.whitelisted_faces[name] = {
                'encoding': encoding,
                'thumbnail_path': thumbnail_path,
                'source_image': image_path
            }
            
            # Persist to disk
            self._save_whitelist()
            
            print(f"  ✓ Face added to whitelist: {name}")
            return True
            
        except Exception as e:
            print(f"  ✗ Error adding face: {e}")
            return False
    
    def remove_face(self, name: str) -> bool:
        """Remove a face from the whitelist"""
        if name in self.whitelisted_faces:
            # Clean up thumbnail
            thumb_path = self.whitelisted_faces[name].get('thumbnail_path')
            if thumb_path and os.path.exists(thumb_path):
                try:
                    os.remove(thumb_path)
                except:
                    pass
            
            del self.whitelisted_faces[name]
            self._save_whitelist()
            print(f"✓ Removed {name} from whitelist")
            return True
        return False
    
    def clear_all(self):
        """Remove all faces from whitelist"""
        # Clean up thumbnails
        for face_data in self.whitelisted_faces.values():
            thumb_path = face_data.get('thumbnail_path')
            if thumb_path and os.path.exists(thumb_path):
                try:
                    os.remove(thumb_path)
                except:
                    pass
        
        self.whitelisted_faces.clear()
        self._save_whitelist()
        print("✓ Cleared all faces from whitelist")
    
    def is_whitelisted(
        self, 
        face_image: np.ndarray, 
        threshold: float = 0.6
    ) -> Tuple[bool, Optional[str], float]:
        """
        Check if a detected face matches any whitelisted face.
        
        Args:
            face_image: BGR image of the detected face
            threshold: Similarity threshold (0.6 recommended for face_recognition, 0.7 for ORB)
            
        Returns:
            (is_match, matched_name, confidence)
        """
        if len(self.whitelisted_faces) == 0:
            return False, None, 0.0
        
        try:
            if self.use_face_recognition:
                # Convert BGR to RGB
                rgb_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
                
                # Get encoding for this face
                encodings = face_recognition.face_encodings(rgb_face)
                
                if len(encodings) == 0:
                    return False, None, 0.0
                
                face_encoding = encodings[0]
                
                # Compare with all whitelisted faces
                best_match_name = None
                best_distance = float('inf')
                
                for name, data in self.whitelisted_faces.items():
                    known_encoding = data['encoding']
                    
                    # Compute face distance (lower = more similar)
                    distance = face_recognition.face_distance([known_encoding], face_encoding)[0]
                    
                    if distance < best_distance:
                        best_distance = distance
                        best_match_name = name
                
                # Convert distance to confidence (0-1, higher = more confident)
                confidence = 1.0 - best_distance
                is_match = best_distance < threshold
                
                return is_match, best_match_name, confidence
                
            else:
                # Fallback: ORB feature matching
                gray_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
                gray_face = cv2.resize(gray_face, (100, 100))
                
                orb = cv2.ORB_create()
                kp, desc = orb.detectAndCompute(gray_face, None)
                
                if desc is None:
                    return False, None, 0.0
                
                test_encoding = desc.flatten()[:128]
                if len(test_encoding) < 128:
                    test_encoding = np.pad(test_encoding, (0, 128 - len(test_encoding)))
                
                # Compare with whitelisted faces using cosine similarity
                best_match_name = None
                best_similarity = 0.0
                
                for name, data in self.whitelisted_faces.items():
                    known_encoding = data['encoding']
                    
                    # Cosine similarity
                    similarity = np.dot(test_encoding, known_encoding) / (
                        np.linalg.norm(test_encoding) * np.linalg.norm(known_encoding)
                    )
                    
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match_name = name
                
                is_match = best_similarity > threshold
                return is_match, best_match_name, best_similarity
                
        except Exception as e:
            print(f"Error checking whitelist: {e}")
            return False, None, 0.0
    
    def match_detected_faces(
        self,
        image: np.ndarray,
        detected_faces: List[Tuple[int, int, int, int]],
        threshold: float = 0.6
    ) -> List[Dict]:
        """
        Check which detected faces match whitelisted faces.
        
        Args:
            image: Full BGR image
            detected_faces: List of (x, y, w, h) bounding boxes
            threshold: Matching threshold
            
        Returns:
            List of dicts with keys: bbox, is_whitelisted, matched_name, confidence
        """
        results = []
        
        for bbox in detected_faces:
            x, y, w, h = bbox
            
            # Extract face region
            face_img = image[y:y+h, x:x+w]
            
            # Check if whitelisted
            is_match, name, confidence = self.is_whitelisted(face_img, threshold)
            
            results.append({
                'bbox': bbox,
                'is_whitelisted': is_match,
                'matched_name': name,
                'confidence': confidence
            })
        
        return results
    
    def get_whitelisted_names(self) -> List[str]:
        """Get list of all whitelisted names"""
        return list(self.whitelisted_faces.keys())
    
    def get_thumbnail_path(self, name: str) -> Optional[str]:
        """Get thumbnail path for a whitelisted face"""
        if name in self.whitelisted_faces:
            return self.whitelisted_faces[name].get('thumbnail_path')
        return None
    
    def _save_thumbnail(self, source_path: str, thumbnail_path: str, size: int = 100):
        """Save a thumbnail of the face"""
        try:
            img = cv2.imread(source_path)
            if img is not None:
                h, w = img.shape[:2]
                # Crop to square
                min_dim = min(h, w)
                y_start = (h - min_dim) // 2
                x_start = (w - min_dim) // 2
                img = img[y_start:y_start+min_dim, x_start:x_start+min_dim]
                # Resize
                img = cv2.resize(img, (size, size))
                cv2.imwrite(thumbnail_path, img)
        except Exception as e:
            print(f"Error saving thumbnail: {e}")
    
    def _save_whitelist(self):
        """Persist whitelist to disk"""
        try:
            with open(self.whitelist_file, 'wb') as f:
                pickle.dump(self.whitelisted_faces, f)
        except Exception as e:
            print(f"Error saving whitelist: {e}")
    
    def _load_whitelist(self):
        """Load whitelist from disk"""
        if os.path.exists(self.whitelist_file):
            try:
                with open(self.whitelist_file, 'rb') as f:
                    self.whitelisted_faces = pickle.load(f)
                print(f"Loaded {len(self.whitelisted_faces)} faces from whitelist")
            except Exception as e:
                print(f"Error loading whitelist: {e}")
                self.whitelisted_faces = {}


if __name__ == "__main__":
    print("=" * 60)
    print("Face Whitelist Module Test")
    print("=" * 60)
    
    whitelist = FaceWhitelist()
    print(f"\nWhitelist has {len(whitelist.get_whitelisted_names())} faces")
    print("Protected faces:", whitelist.get_whitelisted_names())
    
    print("\n" + "=" * 60)