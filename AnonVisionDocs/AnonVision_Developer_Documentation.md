# 📗 AnonVision Developer Documentation

## Face Privacy Protection Application — Technical Reference

**Version:** 1.0  
**Last Updated:** December 2025

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Module Documentation](#2-module-documentation)
3. [Detection Pipeline](#3-detection-pipeline)
4. [Anonymization Pipeline](#4-anonymization-pipeline)
5. [UI Event Flow](#5-ui-event-flow)
6. [Error Handling](#6-error-handling)
7. [Extensibility Guide](#7-extensibility-guide)
8. [Performance Considerations](#8-performance-considerations)
9. [API Reference](#9-api-reference)

---

## 1. Architecture Overview

### System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              AnonVision Application                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────┐                                                            │
│  │   main.py   │  ← Entry Point                                             │
│  │  (Launcher) │     • CPU-only enforcement                                 │
│  └──────┬──────┘     • Dependency checking                                  │
│         │            • Configuration summary                                │
│         ▼                                                                   │
│  ┌─────────────────────────────────────────────────────────────────┐       │
│  │                          gui.py                                  │       │
│  │                    (Main Application UI)                         │       │
│  │  ┌──────────────┬──────────────────┬──────────────────┐         │       │
│  │  │   Prepare    │     Preview      │ Anonymize/Export │         │       │
│  │  │    Panel     │      Panel       │      Panel       │         │       │
│  │  └──────────────┴──────────────────┴──────────────────┘         │       │
│  └───────────┬───────────────────────────────┬─────────────────────┘       │
│              │                               │                              │
│              ▼                               ▼                              │
│  ┌─────────────────────────┐    ┌─────────────────────────┐                │
│  │    Detection Layer      │    │   Anonymization Layer   │                │
│  │  ┌───────────────────┐  │    │  • Gaussian Blur        │                │
│  │  │  dnn_detector.py  │  │    │  • Pixelation           │                │
│  │  │  (YuNet - Primary) │  │    │  • Black Box            │                │
│  │  └─────────┬─────────┘  │    │  • Feather Masking      │                │
│  │            │ fallback   │    └─────────────────────────┘                │
│  │            ▼            │                                               │
│  │  ┌───────────────────┐  │                                               │
│  │  │ face_detector.py  │  │                                               │
│  │  │(Haar - Fallback)  │  │                                               │
│  │  └───────────────────┘  │                                               │
│  └─────────────────────────┘                                               │
│                                                                             │
│  ┌─────────────────────────┐    ┌─────────────────────────┐                │
│  │  face_whitelist.py     │    │       utils.py          │                │
│  │  (Protection Database)  │    │  (Helper Functions)     │                │
│  └─────────────────────────┘    └─────────────────────────┘                │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Technology Stack

| Layer | Technology | Purpose |
|-------|------------|---------|
| **UI Framework** | Tkinter | Native cross-platform GUI |
| **Image Processing** | OpenCV (cv2) | Detection, transformation, anonymization |
| **Image Display** | Pillow (PIL) | Tkinter-compatible image rendering |
| **Numerical Operations** | NumPy | Array manipulation, mask generation |
| **Face Encoding** | face_recognition (optional) | Deep learning face matching |
| **Model Format** | ONNX | YuNet neural network model |

### Data Flow Overview

```
Image File → Load → Detection → Face List → Protection Selection → 
Anonymization → Preview Update → Export
```

---

## 2. Module Documentation

### 2.1 main.py — Application Launcher

#### Purpose

The launcher module serves as the application entry point, responsible for environment configuration, dependency management, and GUI initialization.

#### Key Responsibilities

1. **CPU-Only Enforcement:** Disables GPU acceleration for maximum compatibility
2. **Dependency Checking:** Verifies required packages are installed
3. **Auto-Installation:** Installs missing packages when possible
4. **Configuration Summary:** Displays detection method availability
5. **GUI Launch:** Initializes the main application window

#### Environment Configuration

```
Environment Variables Set:
├── CUDA_VISIBLE_DEVICES = ""      # Disable CUDA
├── OPENCV_OPENCL_DEVICE = ""      # Disable OpenCL
└── OPENCV_DNN_BACKEND = "CPU"     # Force CPU backend
```

#### Startup Sequence

```
┌────────────────────────────────────────────────────────┐
│                    main.py Startup                      │
├────────────────────────────────────────────────────────┤
│ 1. Set environment variables (CPU-only)                │
│ 2. Check required packages                             │
│ 3. Check optional packages                             │
│ 4. Attempt auto-install of missing packages            │
│ 5. Verify YuNet model availability                     │
│ 6. Print configuration summary                         │
│ 7. Initialize Tkinter root window                      │
│ 8. Instantiate AnonVisionApp (gui.py)                  │
│ 9. Enter mainloop()                                    │
└────────────────────────────────────────────────────────┘
```

#### Key Functions

| Function | Description |
|----------|-------------|
| `enforce_cpu_mode()` | Sets environment variables to disable GPU |
| `check_required_packages()` | Verifies core dependencies |
| `check_optional_packages()` | Checks for enhanced features |
| `install_package(name)` | Attempts pip install of missing package |
| `print_config_summary()` | Displays startup configuration |
| `main()` | Entry point; orchestrates startup |

---

### 2.2 gui.py — User Interface

#### Purpose

The GUI module implements the complete user interface using Tkinter, managing the three-panel workflow, user interactions, canvas rendering, and coordination between detection and anonymization components.

#### Class: `AnonVisionApp`

The main application class that encapsulates all UI logic and state management.

#### Panel Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                         Main Application Window                        │
├────────────────┬────────────────────────────┬─────────────────────────┤
│  Left Frame    │      Center Frame          │     Right Frame         │
│  (Prepare)     │      (Preview)             │   (Anonymize/Export)    │
│  width: 250px  │      weight: 1 (expand)    │     width: 280px        │
└────────────────┴────────────────────────────┴─────────────────────────┘
```

#### State Management

| State Variable | Type | Description |
|----------------|------|-------------|
| `current_image` | `np.ndarray` | Original loaded image (BGR) |
| `display_image` | `np.ndarray` | Working copy for preview |
| `anonymized_image` | `np.ndarray` | Result after anonymization |
| `detected_faces` | `list[tuple]` | List of (x, y, w, h) bounding boxes |
| `protected_faces` | `set[int]` | Indices of faces to skip |
| `current_method` | `str` | Selected anonymization method |
| `show_boxes` | `BooleanVar` | Toggle bounding box display |

#### Key Methods

| Method | Description |
|--------|-------------|
| `__init__(root)` | Initialize UI components and state |
| `setup_left_panel()` | Build Prepare panel widgets |
| `setup_center_panel()` | Build Preview canvas |
| `setup_right_panel()` | Build Anonymize/Export panel |
| `select_image()` | Handle image file selection |
| `detect_faces()` | Trigger detection pipeline |
| `update_preview()` | Refresh canvas with current state |
| `draw_bounding_boxes()` | Render face boxes on canvas |
| `apply_anonymization()` | Execute anonymization pipeline |
| `save_result()` | Export anonymized image |
| `toggle_face_protection(idx)` | Add/remove face from protected set |

---

### 2.3 dnn_detector.py — YuNet Detection

#### Purpose

Implements the primary face detection system using the YuNet deep neural network model via OpenCV's DNN module.

#### Class: `DNNFaceDetector`

#### Detection Pipeline

```
detect_faces(image, conf_threshold, nms_threshold)
        │
        ▼
┌───────────────────────────────┐
│ Validate input image          │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│ Set input size to image       │
│ dimensions: setInputSize(w,h) │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│ Execute detection:            │
│ detector.detect(image)        │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│ Filter by confidence:         │
│ keep if score >= threshold    │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│ Apply NMS to remove           │
│ overlapping detections        │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│ Convert to (x,y,w,h) format   │
│ Return list of bounding boxes │
└───────────────────────────────┘
```

#### YuNet Output Format

| Index | Field | Description |
|-------|-------|-------------|
| 0 | x | Bounding box left coordinate |
| 1 | y | Bounding box top coordinate |
| 2 | w | Bounding box width |
| 3 | h | Bounding box height |
| 4-13 | landmarks | 5 facial landmark points (x,y pairs) |
| 14 | score | Detection confidence (0.0–1.0) |

#### Threshold Behavior

**Confidence Threshold:** Range 0.0–1.0. Filters detections below this score. Lower = more detections.

**NMS Threshold:** Range 0.0–1.0. Controls IoU threshold for duplicate removal. Lower = more aggressive.

---

### 2.4 face_detector.py — Haar Cascade Detection

#### Purpose

Provides fallback face detection using OpenCV's classical Haar Cascade classifier when the YuNet DNN model is unavailable.

#### Class: `HaarFaceDetector`

#### Detection Pipeline

```
detect_faces(image, scale_factor, min_neighbors, min_size)
        │
        ▼
┌───────────────────────────────┐
│ Convert to grayscale          │
│ cv2.cvtColor(BGR2GRAY)        │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│ Apply histogram equalization  │
│ cv2.equalizeHist()            │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│ Run cascade detection:        │
│ detectMultiScale()            │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│ Return list of (x,y,w,h)      │
└───────────────────────────────┘
```

#### Parameter Effects

| Parameter | Range | Default | Effect |
|-----------|-------|---------|--------|
| `scale_factor` | 1.01–2.0 | 1.1 | Image pyramid scale; lower = slower but thorough |
| `min_neighbors` | 0–10+ | 5 | Minimum detections to confirm; higher = fewer false positives |
| `min_size` | (w, h) | (30, 30) | Minimum face size in pixels |

---

### 2.5 face_whitelist.py — Face Protection Database

#### Purpose

Manages a persistent database of protected faces, enabling recognition-based protection across sessions.

#### Class: `FaceWhitelist`

#### Storage Structure (JSON)

```json
{
  "faces": [
    {
      "id": "uuid-string",
      "name": "Person Name",
      "encoding": [128-dim array],
      "orb_descriptors": "base64",
      "thumbnail": "base64-jpeg",
      "created_at": "ISO-8601"
    }
  ],
  "version": "1.0"
}
```

#### Encoding Methods

| Method | Library | Output |
|--------|---------|--------|
| Deep Learning | face_recognition | 128-dimensional vector |
| ORB Features | OpenCV | Keypoints + Descriptors |

#### Key Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `add_face(image, name)` | str (id) | Add face to whitelist |
| `remove_face(face_id)` | bool | Remove face from whitelist |
| `match_face(image)` | tuple | Find matching face |
| `get_thumbnail(id)` | PIL.Image | Get face preview |

---

### 2.6 utils.py — Utility Functions

#### Purpose

Provides common helper functions for image conversion, resizing, validation, and coordinate handling.

#### Image Conversion Functions

| Function | Input | Output |
|----------|-------|--------|
| `bgr_to_rgb(image)` | BGR ndarray | RGB ndarray |
| `rgb_to_bgr(image)` | RGB ndarray | BGR ndarray |
| `to_pil_image(image)` | BGR ndarray | PIL.Image |
| `from_pil_image(pil)` | PIL.Image | BGR ndarray |

#### Bounding Box Functions

| Function | Description |
|----------|-------------|
| `clamp_bbox(bbox, w, h)` | Ensure box within image bounds |
| `expand_bbox(bbox, factor, w, h)` | Grow box by percentage |
| `scale_bbox(bbox, scale)` | Apply uniform scale |
| `bbox_to_slice(bbox)` | Convert to array slices |

#### Validation Functions

| Function | Returns | Description |
|----------|---------|-------------|
| `is_valid_image(image)` | bool | Check image validity |
| `is_valid_path(path)` | bool | Check file existence |
| `get_image_dimensions(image)` | tuple | Get (height, width) |


---

## 3. Detection Pipeline

### Complete Detection Flow

```
User clicks "Detect Faces"
        │
        ▼
┌─────────────────────────────────────┐
│ gui.py: detect_faces()              │
│ • Validate image loaded             │
│ • Read UI settings                  │
│ • Determine detector method         │
└──────────────────┬──────────────────┘
                   │
        ┌──────────┴──────────┐
        │  Detection Method?  │
        └──────────┬──────────┘
            DNN    │    Haar
        ┌──────────┴──────────┐
        ▼                     ▼
┌───────────────┐    ┌───────────────┐
│dnn_detector   │    │face_detector  │
│• Load YuNet   │    │• Load Haar    │
│• detect()     │    │• Grayscale    │
│• Filter conf  │    │• detectMS()   │
│• Apply NMS    │    │               │
└───────┬───────┘    └───────┬───────┘
        └──────────┬─────────┘
                   ▼
┌─────────────────────────────────────┐
│ Return: list[(x, y, w, h), ...]     │
└──────────────────┬──────────────────┘
                   ▼
┌─────────────────────────────────────┐
│ gui.py: Store in detected_faces     │
│ • Update face count label           │
│ • Populate protection list          │
│ • Trigger update_preview()          │
└─────────────────────────────────────┘
```

### YuNet vs Haar Comparison

| Aspect | YuNet (DNN) | Haar Cascade |
|--------|-------------|--------------|
| **Architecture** | Deep neural network | Boosted cascade classifier |
| **Model Format** | ONNX | XML |
| **Speed** | Moderate | Fast |
| **Accuracy** | High | Moderate |
| **Pose Tolerance** | Good (±45°) | Limited (frontal) |
| **False Positives** | Low | Moderate |

### Non-Maximum Suppression (NMS)

```
Filtered Detections
        │
        ▼
┌───────────────────────────────┐
│ Sort by confidence descending │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│ For each detection:           │
│   Mark as "kept"              │
│   For each remaining:         │
│     Calculate IoU             │
│     If IoU > threshold:       │
│       Suppress (remove)       │
└───────────────────────────────┘
```

**IoU Formula:** `IoU = Area(A ∩ B) / Area(A ∪ B)`

---

## 4. Anonymization Pipeline

### Complete Anonymization Flow

```
User clicks "Apply Anonymization"
        │
        ▼
┌─────────────────────────────────────┐
│ Create working copy of image        │
│ result = current_image.copy()       │
└──────────────────┬──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│ For each face in detected_faces:    │
└──────────────────┬──────────────────┘
                   │
        ┌──────────┴──────────┐
        │ Is face protected? │
        └──────────┬──────────┘
            Yes    │    No
        ┌──────────┴──────────┐
        ▼                     ▼
┌─────────────┐    ┌─────────────────────┐
│   Skip      │    │ 1. Expand bbox      │
│   (next)    │    │ 2. Extract ROI      │
└─────────────┘    │ 3. Apply method     │
                   │ 4. Generate mask    │
                   │ 5. Blend into result│
                   └──────────┬──────────┘
                              │
                              ▼
┌─────────────────────────────────────┐
│ Store result, update preview        │
└─────────────────────────────────────┘
```

### Anonymization Methods

#### Method 1: Multi-Pass Gaussian Blur

```
Input: face_roi, strength, passes
        │
        ▼
┌───────────────────────────────┐
│ kernel_size = strength | 1    │
│ (ensure odd number)           │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│ For i in range(passes):       │
│   roi = GaussianBlur(roi,     │
│         kernel, 0)            │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│ Return blurred ROI            │
└───────────────────────────────┘
```

> **💡 Note**  
> Kernel size must be odd. The expression `strength | 1` (bitwise OR) ensures this.

#### Method 2: Pixelation

```
Input: face_roi, block_size
        │
        ▼
┌───────────────────────────────┐
│ Store original dimensions     │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│ Downscale with INTER_LINEAR:  │
│ small = resize(roi,           │
│   (w/block, h/block))         │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│ Upscale with INTER_NEAREST:   │
│ (preserves blocky look)       │
│ pixelated = resize(small,     │
│   original_size)              │
└───────────────────────────────┘
```

#### Method 3: Black Box

```
Input: face_roi, fill_color
        │
        ▼
┌───────────────────────────────┐
│ roi[:] = fill_color           │
│ (direct array assignment)     │
└───────────────────────────────┘
```

### Feather Mask Generation

The feather mask creates smooth edge transitions using distance transform.

```
Input: roi_shape, feather_pixels
        │
        ▼
┌───────────────────────────────┐
│ Create binary mask (all 1s)   │
│ mask = ones(shape)            │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│ Set border to 0:              │
│ mask[0:feather, :] = 0  (top) │
│ mask[-feather:, :] = 0 (bottom)│
│ mask[:, 0:feather] = 0 (left) │
│ mask[:, -feather:] = 0 (right)│
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│ Apply distance transform:     │
│ dist = distanceTransform(     │
│   mask, DIST_L2, 5)           │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│ Normalize to 0-1 range:       │
│ mask = dist / dist.max()      │
└───────────────────────────────┘
```

**Distance Transform Visualization:**

```
0 0 0 0 0 0 0 0 0   ← Edge (0)
0 1 1 1 1 1 1 1 0
0 1 2 2 2 2 2 1 0
0 1 2 3 3 3 2 1 0   ← Center (max)
0 1 2 2 2 2 2 1 0
0 1 1 1 1 1 1 1 0
0 0 0 0 0 0 0 0 0   ← Edge (0)
```

### Blending Formula

```python
result = anonymized * mask + original * (1 - mask)
```

---

## 5. UI Event Flow

### Event Binding Map

| Widget | Event | Handler | Action |
|--------|-------|---------|--------|
| Select Image Button | Click | `select_image()` | Open file dialog |
| Detect Faces Button | Click | `detect_faces()` | Run detection |
| Apply Button | Click | `apply_anonymization()` | Anonymize faces |
| Save Button | Click | `save_result()` | Export image |
| Canvas | Configure | `on_canvas_resize()` | Redraw on resize |
| Protection Checkbox | Click | `toggle_protection(idx)` | Update set |
| Method Radio | Click | `on_method_change()` | Update UI |
| Sliders | Drag | `on_slider_change()` | Live preview |

### Canvas Rendering Pipeline

```
update_preview() Called
        │
        ▼
┌───────────────────────────────┐
│ Create display copy           │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│ Calculate scaling factor      │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│ Resize maintaining ratio      │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│ If show_boxes: draw boxes     │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│ BGR → RGB → PIL → ImageTk     │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│ canvas.create_image()         │
└───────────────────────────────┘
```

### Status Messages

| Action | Status Message |
|--------|----------------|
| Image loaded | "Loaded: filename.jpg" |
| Detection running | "Detecting faces..." |
| Detection complete | "Found X faces" |
| Anonymization complete | "Anonymized X, Protected Y" |
| Save successful | "Saved: output.jpg" |

---

## 6. Error Handling

### Error Handling Strategy

```
┌────────────────────────────────────────────────────────┐
│                   Error Handling Layers                 │
├────────────────────────────────────────────────────────┤
│  Layer 1: Input Validation                             │
│  ├── File existence checks                             │
│  ├── Image format validation                           │
│  └── Parameter range validation                        │
│                                                         │
│  Layer 2: Operation Guards                             │
│  ├── Null image checks                                 │
│  ├── Empty face list checks                            │
│  └── Model availability checks                         │
│                                                         │
│  Layer 3: Exception Handling                           │
│  ├── FileNotFoundError → User notification            │
│  ├── cv2.error → Graceful degradation                 │
│  ├── MemoryError → Suggest smaller image              │
│  └── Generic Exception → Log and notify               │
└────────────────────────────────────────────────────────┘
```

### Common Error Patterns

**Image Loading:**
```python
try:
    image = cv2.imread(path)
    if image is None:
        raise ValueError("Failed to decode")
except Exception as e:
    self.show_error("Load failed", str(e))
```

**Detection:**
```python
try:
    faces = detector.detect_faces(image)
except cv2.error as e:
    self.update_status("Detection failed")
    faces = []
```

**File Saving:**
```python
try:
    cv2.imwrite(path, image)
except PermissionError:
    self.show_error("Permission denied")
except Exception as e:
    self.show_error("Save failed", str(e))
```

### Error Recovery Strategies

| Error Type | Recovery Action |
|------------|-----------------|
| YuNet unavailable | Fall back to Haar |
| Image too large | Suggest resize |
| No faces found | Suggest lower threshold |
| Save failed | Suggest different location |


---

## 7. Extensibility Guide

### Adding New Anonymization Methods

To add a new anonymization method (e.g., "Mosaic" or "Swirl"):

**Step 1:** Add the method to the UI in `gui.py`:

```python
# In setup_right_panel():
self.methods = ["Blur", "Pixelate", "Black Box", "NewMethod"]
# Add radio button for new method
```

**Step 2:** Implement the anonymization logic:

```python
def apply_new_method(self, roi, param1, param2):
    """
    Apply new anonymization effect to ROI.
    
    Args:
        roi: Face region (BGR ndarray)
        param1: First parameter
        param2: Second parameter
    
    Returns:
        Processed ROI (same shape as input)
    """
    # Your implementation here
    processed = some_operation(roi, param1, param2)
    return processed
```

**Step 3:** Update the method dispatcher in `apply_anonymization()`:

```python
if self.current_method == "Blur":
    result_roi = self.apply_blur(roi, strength, passes)
elif self.current_method == "NewMethod":
    result_roi = self.apply_new_method(roi, param1, param2)
```

**Step 4:** Add UI controls for method-specific parameters.

---

### Adding New Detectors

To add a new face detector (e.g., RetinaFace, MTCNN):

**Step 1:** Create a new detector module:

```python
# new_detector.py
class NewFaceDetector:
    def __init__(self, model_path=None):
        """Initialize the detector."""
        self.model = self._load_model(model_path)
    
    def detect_faces(self, image, **kwargs):
        """
        Detect faces in image.
        
        Args:
            image: BGR ndarray
            **kwargs: Detector-specific parameters
        
        Returns:
            list of (x, y, w, h) tuples
        """
        # Implementation
        return faces
    
    def is_available(self):
        """Check if detector is ready."""
        return self.model is not None
```

**Step 2:** Register in detector factory:

```python
# In gui.py or a detector_factory.py
DETECTORS = {
    "DNN (YuNet)": DNNFaceDetector,
    "Haar Cascade": HaarFaceDetector,
    "New Detector": NewFaceDetector,
}
```

**Step 3:** Update UI to include new detector option.

**Step 4:** Add detector-specific settings to Advanced Settings panel.

---

### Adding New UI Panels

To add a new panel (e.g., "History" or "Batch Processing"):

**Step 1:** Create the panel setup method:

```python
def setup_new_panel(self):
    """Build the new panel UI."""
    self.new_frame = ttk.LabelFrame(
        self.main_container, 
        text="New Panel"
    )
    # Add widgets to self.new_frame
```

**Step 2:** Add to main container layout:

```python
# In __init__():
self.setup_left_panel()
self.setup_center_panel()
self.setup_right_panel()
self.setup_new_panel()  # Add new panel

# Grid configuration
self.new_frame.grid(row=1, column=0, columnspan=3, sticky="ew")
```

**Step 3:** Implement panel-specific functionality and event handlers.

---

### Adding New Export Formats

To support additional export formats:

**Step 1:** Update file dialog filter:

```python
filetypes = [
    ("JPEG", "*.jpg"),
    ("PNG", "*.png"),
    ("WebP", "*.webp"),  # New format
    ("TIFF", "*.tiff"),
]
```

**Step 2:** Handle format-specific encoding:

```python
def save_result(self, path):
    ext = os.path.splitext(path)[1].lower()
    
    if ext == '.webp':
        params = [cv2.IMWRITE_WEBP_QUALITY, 95]
    elif ext == '.jpg':
        params = [cv2.IMWRITE_JPEG_QUALITY, 95]
    else:
        params = []
    
    cv2.imwrite(path, self.result_image, params)
```

---

### Integrating Face Whitelist

To enable automatic protection via whitelist matching:

**Step 1:** Initialize whitelist in GUI:

```python
from face_whitelist import FaceWhitelist

class AnonVisionApp:
    def __init__(self, root):
        self.whitelist = FaceWhitelist("whitelist.json")
```

**Step 2:** Check against whitelist during detection:

```python
def detect_faces(self):
    faces = self.detector.detect_faces(self.image)
    
    for idx, (x, y, w, h) in enumerate(faces):
        face_roi = self.image[y:y+h, x:x+w]
        match = self.whitelist.match_face(face_roi)
        
        if match is not None:
            self.protected_faces.add(idx)
```

**Step 3:** Add UI for whitelist management (add/remove faces).

---

## 8. Performance Considerations

### CPU-Only Execution

AnonVision enforces CPU-only mode for maximum compatibility:

```python
# Environment variables set at startup
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OPENCV_OPENCL_DEVICE"] = ""
os.environ["OPENCV_DNN_BACKEND"] = "CPU"
```

**Impact:**
- Slower than GPU execution
- Consistent behavior across systems
- No driver dependencies

---

### Image Memory Management

#### Color Space Conversions

| Conversion | Cost | When Used |
|------------|------|-----------|
| BGR → RGB | Low | Display in Tkinter |
| BGR → Grayscale | Low | Haar detection |
| RGB → BGR | Low | Save from PIL |

> **⚠️ Warning**  
> Avoid unnecessary conversions. Store images in BGR (OpenCV native) and convert only for display.

#### Image Copying Strategy

| Operation | Copy Required? | Reason |
|-----------|---------------|--------|
| Detection | No | Read-only operation |
| Preview display | Yes | Avoid modifying original |
| Anonymization | Yes | Preserve original for undo |
| Drawing boxes | Yes | Temporary visualization |

```python
# Good: Single copy for anonymization
result = self.current_image.copy()
# Process result...

# Bad: Multiple unnecessary copies
temp1 = self.current_image.copy()
temp2 = temp1.copy()  # Wasteful
```

---

### Canvas Redraw Optimization

**Problem:** Frequent redraws cause UI lag.

**Solutions:**

1. **Debounce resize events:**
```python
def on_resize(self, event):
    if self._resize_timer:
        self.after_cancel(self._resize_timer)
    self._resize_timer = self.after(100, self.update_preview)
```

2. **Cache scaled images:**
```python
# Store last scaled dimensions
if (new_w, new_h) == self._cached_size:
    return self._cached_image
```

3. **Avoid redundant redraws:**
```python
def update_preview(self):
    if not self._needs_redraw:
        return
    # ... render ...
    self._needs_redraw = False
```

---

### Detection Performance

| Factor | Impact | Optimization |
|--------|--------|--------------|
| Image size | O(n) pixels processed | Resize before detection |
| Face count | O(n) for NMS | Use higher NMS threshold |
| Scale factor (Haar) | More scales = slower | Use 1.1–1.2 |
| Confidence threshold | Lower = more processing | Use 0.5+ |

**Recommended workflow for large images:**

```python
# Detect at reduced resolution
scale = 0.5
small = cv2.resize(image, None, fx=scale, fy=scale)
faces = detector.detect_faces(small)

# Scale bounding boxes back
faces = [(int(x/scale), int(y/scale), 
          int(w/scale), int(h/scale)) 
         for x, y, w, h in faces]
```

---

### Memory Usage Estimates

| Image Size | Memory (BGR) | With Copy |
|------------|--------------|-----------|
| 1920×1080 | ~6 MB | ~12 MB |
| 3840×2160 | ~24 MB | ~48 MB |
| 4000×4000 | ~48 MB | ~96 MB |

> **💡 Tip**  
> For systems with limited RAM, recommend images under 4000×4000 pixels.

---

## 9. API Reference

### DNNFaceDetector

```python
class DNNFaceDetector:
    """YuNet-based face detector."""
    
    def __init__(self, model_path: str = None) -> None:
        """
        Initialize detector.
        
        Args:
            model_path: Path to ONNX model file.
                       If None, searches default locations.
        """
    
    def detect_faces(
        self,
        image: np.ndarray,
        conf_threshold: float = 0.5,
        nms_threshold: float = 0.3
    ) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in image.
        
        Args:
            image: BGR image as numpy array
            conf_threshold: Minimum confidence (0.0-1.0)
            nms_threshold: NMS IoU threshold (0.0-1.0)
        
        Returns:
            List of (x, y, width, height) bounding boxes
        """
    
    def is_available(self) -> bool:
        """Check if model is loaded and ready."""
```

### HaarFaceDetector

```python
class HaarFaceDetector:
    """Haar Cascade face detector."""
    
    def __init__(self, cascade_path: str = None) -> None:
        """
        Initialize detector.
        
        Args:
            cascade_path: Path to cascade XML file.
        """
    
    def detect_faces(
        self,
        image: np.ndarray,
        scale_factor: float = 1.1,
        min_neighbors: int = 5,
        min_size: Tuple[int, int] = (30, 30)
    ) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in image.
        
        Args:
            image: BGR image as numpy array
            scale_factor: Image pyramid scale (1.01-2.0)
            min_neighbors: Minimum detections to confirm
            min_size: Minimum face size (width, height)
        
        Returns:
            List of (x, y, width, height) bounding boxes
        """
```

### FaceWhitelist

```python
class FaceWhitelist:
    """Persistent face protection database."""
    
    def __init__(self, db_path: str = "whitelist.json") -> None:
        """
        Initialize whitelist.
        
        Args:
            db_path: Path to JSON storage file.
        """
    
    def add_face(
        self,
        face_image: np.ndarray,
        name: str
    ) -> str:
        """
        Add face to whitelist.
        
        Args:
            face_image: Face ROI as BGR array
            name: Display name for face
        
        Returns:
            UUID string for the new face entry
        """
    
    def match_face(
        self,
        face_image: np.ndarray,
        threshold: float = 0.6
    ) -> Optional[Tuple[str, float]]:
        """
        Find matching face in whitelist.
        
        Args:
            face_image: Face ROI to match
            threshold: Maximum distance for match
        
        Returns:
            (face_id, confidence) tuple or None if no match
        """
    
    def remove_face(self, face_id: str) -> bool:
        """Remove face from whitelist."""
    
    def get_thumbnail(self, face_id: str) -> PIL.Image:
        """Get thumbnail image for face."""
```

### Utility Functions

```python
def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert BGR to RGB color space."""

def rgb_to_bgr(image: np.ndarray) -> np.ndarray:
    """Convert RGB to BGR color space."""

def resize_to_fit(
    image: np.ndarray,
    max_width: int,
    max_height: int
) -> np.ndarray:
    """Resize image to fit within dimensions, maintaining aspect ratio."""

def clamp_bbox(
    bbox: Tuple[int, int, int, int],
    img_width: int,
    img_height: int
) -> Tuple[int, int, int, int]:
    """Clamp bounding box to image boundaries."""

def expand_bbox(
    bbox: Tuple[int, int, int, int],
    expansion_percent: float,
    img_width: int,
    img_height: int
) -> Tuple[int, int, int, int]:
    """Expand bounding box by percentage, clamped to image."""

def is_valid_image(image: Any) -> bool:
    """Check if object is a valid image array."""

def is_valid_path(path: str) -> bool:
    """Check if file path exists and is readable."""
```

---

## Document Information

**Application:** AnonVision – Face Privacy Protection  
**Document Type:** Developer Documentation  
**Version:** 1.0  
**Last Updated:** December 2025

---

*For user-facing documentation, see the AnonVision User Guide.*
