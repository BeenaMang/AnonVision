import cv2
import numpy as np

def create_basic_image():
    """Draw a few simple shapes and add text onto a blank canvas."""
    print("=== Creating Basic Image ===")

    # Start with a blank black image (height=400, width=600, RGB=3 channels)
    img = np.zeros((400, 600, 3), dtype=np.uint8)

    # Draw shapes
    cv2.rectangle(img, (50, 50), (250, 150), (0, 255, 0), 3)         # Green rectangle outline
    cv2.circle(img, (450, 100), 60, (255, 0, 0), -1)                  # Solid blue circle
    cv2.line(img, (50, 200), (550, 200), (0, 0, 255), 2)              # Red line

    # Add text
    cv2.putText(img, "OpenCV Basics", (200, 250),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(img, "WIP1.py Demo", (220, 300),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

    # Save to disk
    cv2.imwrite("opencv_basics.jpg", img)
    print("Saved: opencv_basics.jpg | Shape:", img.shape)

    return img

def create_gradient_image():
    """Generate a grayscale and colorful gradient."""
    print("\n=== Creating Gradient Image ===")

    width, height = 400, 300
    gradient = np.zeros((height, width), dtype=np.uint8)

    # Fill each column with increasing intensity (black → white)
    for x in range(width):
        gradient[:, x] = int(255 * x / width)

    # Apply a rainbow color map
    colored = cv2.applyColorMap(gradient, cv2.COLORMAP_RAINBOW)

    # Save results
    cv2.imwrite("gradient_bw.jpg", gradient)
    cv2.imwrite("gradient_colored.jpg", colored)
    print("Saved: gradient_bw.jpg and gradient_colored.jpg")

    return gradient, colored

def basic_image_operations():
    """Show resizing, blurring, and image property extraction."""
    print("\n=== Basic Image Operations ===")

    # Blank black image
    img = np.zeros((200, 300, 3), dtype=np.uint8)

    # Fill regions with solid colors
    img[50:150, 50:150] = [255, 0, 0]   # Blue block
    img[25:175, 150:250] = [0, 255, 0]  # Green block

    # Display basic info
    h, w, c = img.shape
    print(f"Size: {w}x{h} | Channels: {c} | Dtype: {img.dtype}")

    # Resize
    resized = cv2.resize(img, (150, 100))
    print("Resized to:", resized.shape)

    # Blur
    blurred = cv2.GaussianBlur(img, (21, 21), 0)

    # Save
    cv2.imwrite("original_squares.jpg", img)
    cv2.imwrite("resized_squares.jpg", resized)
    cv2.imwrite("blurred_squares.jpg", blurred)
    print("Saved: original_squares.jpg, resized_squares.jpg, blurred_squares.jpg")

def draw_multiple_shapes():
    """Draw a variety of shapes on a white background."""
    print("\n=== Drawing Multiple Shapes ===")

    # White canvas
    img = np.ones((400, 500, 3), dtype=np.uint8) * 255

    # Shapes
    cv2.rectangle(img, (50, 50), (150, 120), (0, 0, 255), 2)      # Red rectangle outline
    cv2.rectangle(img, (200, 50), (300, 120), (0, 255, 0), -1)    # Green filled rectangle
    cv2.circle(img, (100, 200), 40, (255, 0, 0), 3)                # Blue circle outline
    cv2.circle(img, (250, 200), 40, (128, 0, 128), -1)             # Purple filled circle

    # Triangle (polygon fill)
    triangle_pts = np.array([[100, 300], [150, 350], [50, 350]], np.int32)
    cv2.fillPoly(img, [triangle_pts], (0, 165, 255))               # Orange filled triangle

    # Ellipse
    cv2.ellipse(img, (350, 200), (60, 30), 45, 0, 360,
                (255, 255, 0), 2)                                  # Cyan ellipse

    # Title
    cv2.putText(img, "Shapes Demo", (150, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    # Save
    cv2.imwrite("shapes_demo.jpg", img)
    print("Saved: shapes_demo.jpg")


def main():
    """Run all OpenCV demonstration functions."""
    print("=" * 40)
    print("WIP1.py - OpenCV Basics Demo")
    print("=" * 40)

    try:
        create_basic_image()
        create_gradient_image()
        basic_image_operations()
        draw_multiple_shapes()
        print("\n✅ All demos completed successfully.")
        print("📂 Check your folder for the generated images.")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Tip: Make sure OpenCV is installed → pip install opencv-python")


if __name__ == "__main__":
    main()