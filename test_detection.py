"""
Test script for face detection
Tests the face detector on sample images
"""

import cv2
import os
from face_detector import FaceDetector


def setup_test_environment():
    """Set up test images directory if needed"""
    test_dir = 'test_images'
    
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
        print(f"Created {test_dir}/ directory")
        print("Please add test images (JPG/PNG) to this folder and run again")
        return False
    
    # Create output directory
    os.makedirs('output', exist_ok=True)
    
    return True


def test_face_detection():
    """Test face detection on sample images"""
    
    print("=" * 50)
    print("AnonVision - Face Detection Test")
    print("=" * 50)
    
    # Set up environment
    if not setup_test_environment():
        return
    
    # Initialize detector
    try:
        detector = FaceDetector()
    except Exception as e:
        print(f"Error initializing detector: {e}")
        return
    
    # Get all test images
    test_image_dir = 'test_images'
    image_files = [f for f in os.listdir(test_image_dir) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    if not image_files:
        print("\nNo image files found in test_images/ folder")
        print("Please add some test images (JPG/PNG) and run again")
        return
    
    print(f"\nFound {len(image_files)} test image(s)\n")
    
    # Test each image
    total_faces = 0
    
    for image_file in image_files:
        image_path = os.path.join(test_image_dir, image_file)
        
        try:
            # Detect faces
            image, faces = detector.detect_faces(image_path)
            total_faces += len(faces)
            
            # Draw detection boxes
            image_with_boxes = detector.draw_detection_boxes(image, faces)
            
            # Save output
            output_path = os.path.join('output', f"detected_{image_file}")
            cv2.imwrite(output_path, image_with_boxes)
            print(f"  ✓ Saved result: {output_path}")
            
        except Exception as e:
            print(f"  ✗ Error processing {image_file}: {e}")
    
    print("\n" + "=" * 50)
    print(f"Test Complete! Total faces detected: {total_faces}")
    print("=" * 50)
    print("\nCheck the 'output/' folder to see detection results")


if __name__ == "__main__":
    test_face_detection()