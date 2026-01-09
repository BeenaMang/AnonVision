# AnonVision: Complete Training & Systems Documentation

**A Comprehensive Guide to Understanding, Building, and Extending the Face Privacy Protection System**

---

## Document Information

| Field | Value |
|-------|-------|
| Project | AnonVision |
| Document Type | Training & Architecture Documentation |
| Target Audience | Developers, HCI Researchers, Beginner-to-Intermediate Python Programmers |
| Last Updated | December 2025 |

---

# Part 0: Project Intake Report

## 0.1 Files Received

**Total files received: 6**

| # | Filename | Lines | Primary Purpose |
|---|----------|-------|-----------------|
| 1 | `start_the_app_.py` | ~115 | Application launcher & dependency manager |
| 2 | `gui.py` | 913 | Main GUI application (Tkinter) |
| 3 | `dnn_detector.py` | ~95 | YuNet DNN face detection with Haar fallback |
| 4 | `face_detector.py` | ~115 | Standalone Haar Cascade face detector |
| 5 | `face_whitelist.py` | ~145 | Face recognition & whitelist management |
| 6 | `utils.py` | ~65 | Image manipulation utilities |

## 0.2 Reconstructed Folder Structure

```
AnonVision/
├── start_the_app_.py       # Entry point launcher
├── gui.py                  # Main application GUI
├── dnn_detector.py         # Primary DNN-based face detector
├── face_detector.py        # Secondary Haar Cascade detector
├── face_whitelist.py       # Whitelist face recognition
├── utils.py                # Shared image utilities
├── data/                   # Auto-created at runtime
│   ├── face_detection_yunet_2023mar.onnx   # Downloaded model
│   └── haarcascade_frontalface_default.xml # Downloaded cascade
└── whitelist_db/           # Auto-created for whitelist
    ├── faces.json          # Whitelist metadata
    └── *_thumb.jpg         # Face thumbnails
```

## 0.3 Module Classification

### Entry Point
- **`start_the_app_.py`** — Checks dependencies, configures environment, launches application

### UI Layer
- **`gui.py`** — All user interface components, event handling, display logic

### Core Processing Pipeline
- **`dnn_detector.py`** — Primary face detection (YuNet neural network)
- **`face_detector.py`** — Fallback face detection (Haar Cascade)
- **`face_whitelist.py`** — Face recognition for selective anonymization

### Utilities
- **`utils.py`** — Image format conversion, validation, display helpers

### Configuration & Data Assets
- Auto-downloaded models stored in `data/`
- Whitelist database in `whitelist_db/`

## 0.4 Identified Gaps & Assumptions

### Missing Files
The launcher (`start_the_app_.py`) references `whitelist_tkinter_integration.py` in error messages, but this file was **not provided**. 

**Assumption**: The whitelist integration functionality is either:
1. Planned but not yet implemented
2. Integrated directly into `gui.py` (which does import `face_whitelist.py` though doesn't appear to use it in the provided code)
3. Available in a separate module not included in this submission

**Implication**: The current `gui.py` does not appear to implement whitelist-based selective anonymization. The infrastructure exists (`face_whitelist.py`) but isn't connected to the UI. This documentation will explain how to complete this integration.

### Video Processing
While the system description mentions "video," no video processing code was found. The current implementation handles **still images only**.

---

# Part 1: System Overview

## 1.1 What AnonVision Does

AnonVision is a desktop application that automatically detects human faces in photographs and applies anonymization effects (blur, pixelation, or blackout) to protect individuals' privacy. What makes it **selective** is the ability to choose which faces to anonymize and which to leave visible.

### Core Workflow
1. User loads an image containing people
2. AnonVision detects all faces using computer vision
3. User selects which faces to **protect** (leave visible)
4. Unprotected faces are anonymized using the chosen method
5. User saves the privacy-protected image

## 1.2 The Problem It Solves

In an era of ubiquitous photography and social media sharing, there's a fundamental tension between:
- **Sharing moments** — People want to share photos from events, public spaces, and gatherings
- **Respecting privacy** — Not everyone in a photo consented to have their image published

Traditional solutions fail:
- **Manual blurring** is tedious and inconsistent
- **Automatic blurring** anonymizes everyone, even the subjects who want to be visible
- **Cloud-based tools** require uploading sensitive images to third parties

AnonVision provides **selective, local, automated** face anonymization.

## 1.3 Why Offline-First Architecture

AnonVision processes everything locally on the user's computer. This design choice stems from several important considerations:

### Privacy by Design
The moment you upload a face image to a cloud service, you've already compromised the privacy you're trying to protect. The server could:
- Store the original image
- Use it for training data
- Be breached by attackers
- Be compelled by legal processes to reveal data

By processing locally, the original image never leaves the user's device.

### No Dependencies on External Services
- Works without internet connection
- No API rate limits or costs
- No service discontinuation risk
- Predictable, consistent performance

### User Control
Users maintain complete control over:
- Which images are processed
- What anonymization is applied
- Where outputs are saved
- Whether to keep or delete originals

## 1.4 Why Selective Anonymization Matters

Consider these real scenarios:

**The Group Photo Problem**: You want to share a photo from a birthday party. The birthday person and their family want to be visible, but several guests prefer not to be on social media.

**The Street Photography Problem**: You captured a beautiful moment in a public space, but incidental bystanders didn't consent to being photographed.

**The Journalism Problem**: A photo documents a newsworthy event, but some individuals are vulnerable or at risk if identified.

In each case, blanket anonymization would destroy the image's purpose. Selective anonymization preserves the intended subjects while protecting others.

## 1.5 Target Users

### Primary Users
- **Content creators** sharing images on social media
- **Researchers** publishing studies involving human subjects
- **Journalists** protecting sources and bystanders
- **Organizations** documenting events with mixed consent

### Developer Audience
- **Computer vision learners** studying face detection
- **HCI researchers** interested in privacy tools
- **Python developers** building desktop applications
- **Open source contributors** extending the system

---

# Part 2: Architecture Map

## 2.1 High-Level System Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              AnonVision                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                     START_THE_APP_.PY                            │    │
│  │           (Entry Point / Dependency Manager)                     │    │
│  └──────────────────────────┬──────────────────────────────────────┘    │
│                              │                                           │
│                              ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                          GUI.PY                                  │    │
│  │                  (Application Controller)                        │    │
│  │                                                                   │    │
│  │  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐     │    │
│  │  │  LEFT     │  │  CENTER   │  │  RIGHT    │  │  STATE    │     │    │
│  │  │  PANEL    │  │  PANEL    │  │  PANEL    │  │  MANAGER  │     │    │
│  │  │           │  │           │  │           │  │           │     │    │
│  │  │ • Input   │  │ • Canvas  │  │ • Anon    │  │ • Images  │     │    │
│  │  │ • Method  │  │ • Preview │  │ • Faces   │  │ • Faces   │     │    │
│  │  │ • Settings│  │ • Display │  │ • Export  │  │ • Settings│     │    │
│  │  └───────────┘  └───────────┘  └───────────┘  └───────────┘     │    │
│  └───────────────────────┬─────────────────────────────────────────┘    │
│                          │                                               │
│           ┌──────────────┼──────────────┐                               │
│           ▼              ▼              ▼                               │
│  ┌──────────────┐ ┌─────────────┐ ┌──────────────┐                      │
│  │DNN_DETECTOR  │ │FACE_DETECTOR│ │FACE_WHITELIST│                      │
│  │   .PY        │ │    .PY      │ │     .PY      │                      │
│  │              │ │             │ │              │                      │
│  │ • YuNet     │ │ • Haar      │ │ • Add faces  │                      │
│  │ • ONNX      │ │ • Cascade   │ │ • Match faces│                      │
│  │ • Fallback  │ │ • OpenCV    │ │ • face_rec   │                      │
│  └──────────────┘ └─────────────┘ └──────────────┘                      │
│           │              │              │                               │
│           └──────────────┼──────────────┘                               │
│                          ▼                                               │
│                   ┌─────────────┐                                        │
│                   │  UTILS.PY   │                                        │
│                   │             │                                        │
│                   │ • Image I/O │                                        │
│                   │ • Resize    │                                        │
│                   │ • Convert   │                                        │
│                   └─────────────┘                                        │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## 2.2 Data Flow Diagram

```
INPUT                    DETECTION               DECISION                OUTPUT
─────                    ─────────               ────────                ──────

┌─────────┐             ┌──────────┐            ┌──────────┐           ┌──────────┐
│  Image  │────────────▶│  Face    │───────────▶│  User    │──────────▶│  Anon    │
│  File   │   load      │  Detect  │   boxes    │  Select  │  masks    │  Apply   │
│  (.jpg) │   cv2       │  YuNet/  │   [(x,y,   │  Protect │           │  Blur/   │
│         │   imread    │  Haar    │   w,h),..] │  Faces   │           │  Pixel/  │
└─────────┘             └──────────┘            └──────────┘           │  Black   │
                                                      │                 └────┬─────┘
                                                      │                      │
                                                      │                      ▼
                                                      │                ┌──────────┐
                                                      │                │  Save    │
                                                      │                │  Image   │
                                                      ▼                │  (.png)  │
                                               ┌───────────┐          └──────────┘
                                               │ Protected │
                                               │ Faces     │
                                               │ (indices) │
                                               └───────────┘
```

## 2.3 Control Flow Diagram

```
USER ACTION              GUI HANDLER              PROCESSING              UI FEEDBACK
───────────              ───────────              ──────────              ───────────

[Click Select Image]
        │
        └──────────────▶ select_image()
                              │
                              ├──▶ filedialog.askopenfilename()
                              ├──▶ validate_image_path()
                              ├──▶ cv2.imread()
                              ├──▶ Store: self.img_bgr, self.pil_image
                              └──▶ _refresh_display() ─────────────▶ [Image shown on canvas]

[Click Detect Faces]
        │
        └──────────────▶ detect_faces()
                              │
                              ├──▶ Check: self.detection_method
                              │      │
                              │      ├──▶ "dnn": self.dnn.detect_faces()
                              │      └──▶ "haar": self.haar.detectMultiScale()
                              │
                              ├──▶ Store: self.detected_faces
                              ├──▶ _build_face_checklist() ────────▶ [Face thumbnails appear]
                              └──▶ _refresh_display() ─────────────▶ [Boxes drawn on image]

[Check/Uncheck Face]
        │
        └──────────────▶ self.face_vars[idx].set(True/False)
                              │
                              └──▶ _refresh_display() ─────────────▶ [Box color changes]

[Click Apply Anonymization]
        │
        └──────────────▶ apply_anonymization()
                              │
                              ├──▶ _anonymize_faces()
                              │      │
                              │      ├──▶ For each face: check protected
                              │      ├──▶ If not protected: apply blur/pixel/black
                              │      └──▶ Return modified image
                              │
                              └──▶ _refresh_display() ─────────────▶ [Anonymized image shown]

[Click Save Result]
        │
        └──────────────▶ save_image()
                              │
                              ├──▶ filedialog.asksaveasfilename()
                              ├──▶ cv2.imwrite()
                              └──▶ messagebox.showinfo() ──────────▶ [Confirmation dialog]
```

## 2.4 State Model

At any moment, the AnonVisionGUI class maintains the following state:

```python
# Image State
self.current_image_path     # Path to loaded image file
self.pil_image              # PIL.Image (original, RGB)
self.img_bgr                # numpy array (working copy, BGR)
self.original_bgr           # numpy array (pristine copy, BGR)
self.photo_image            # ImageTk.PhotoImage (display buffer)

# Detection State  
self.detected_faces         # List of (x, y, w, h) tuples
self.face_vars              # List of tk.BooleanVar (protection toggles)
self.face_thumbs            # List of ImageTk.PhotoImage (thumbnail refs)

# Detector Instances
self.dnn                    # DNNFaceDetector or None
self.haar                   # FaceDetector or None

# User Settings (tk.Variables)
self.detection_method       # "dnn" or "haar"
self.dnn_confidence         # 0.0-1.0
self.scale_factor           # 1.01-2.0 (Haar)
self.min_neighbors          # 1-10 (Haar)
self.min_size               # 10-200 (Haar)
self.anon_method            # "blur", "pixelate", "blackout"
self.blur_strength          # 41-99 (odd values)
self.blur_passes            # 3-10
self.pixel_size             # 8-50
self.edge_feather           # 0-50 (percent)
self.region_expansion       # 10-60 (percent)
self.show_boxes             # True/False
```

## 2.5 Why This Architecture?

### Why Modular Detection?

The system separates detection into two modules (`dnn_detector.py` and `face_detector.py`) for several reasons:

1. **Graceful Degradation**: If YuNet fails to load (model not downloaded, OpenCV version issues), the system falls back to Haar Cascade automatically.

2. **User Choice**: Some users prioritize accuracy (DNN), others prioritize speed (Haar). Separate modules let them choose.

3. **Maintainability**: Each detector can be updated, replaced, or extended independently.

4. **Testing**: Detectors can be tested in isolation without the full GUI.

### Why Whitelist-Based Recognition?

The system uses a "whitelist" approach rather than a "blacklist":

**Whitelist (Keep These Visible)**:
- Users add faces of people who should remain visible
- Unknown faces are anonymized by default
- Fails safe: if recognition fails, privacy is protected

**Why Not Blacklist (Hide These Faces)?**:
- Requires knowing who will appear in photos ahead of time
- Fails unsafe: if recognition fails, privacy is compromised
- More complex to maintain

### Why CPU-Only Design?

The code explicitly disables GPU acceleration:

```python
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OPENCV_DNN_DISABLE_OPENCL"] = "1"
cv2.ocl.setUseOpenCL(False)
```

**Reasons**:

1. **Portability**: Works on any computer, regardless of GPU availability
2. **Simplicity**: No CUDA/cuDNN installation headaches
3. **Predictability**: Same behavior on all machines
4. **Sufficient Performance**: Face detection is fast enough on CPU for single images
5. **Reliability**: GPU drivers and OpenCL implementations vary wildly

### Why Tkinter?

Tkinter was chosen as the GUI toolkit for several pragmatic reasons:

1. **Zero Dependencies**: Ships with Python—no pip install needed
2. **Cross-Platform**: Works on Windows, macOS, Linux identically
3. **Simplicity**: Easy to understand for beginners
4. **Stability**: Mature, well-tested, won't break between Python versions
5. **Sufficient Features**: Adequate for this application's needs

**Tradeoffs Accepted**:
- Less modern appearance (mitigated with custom styling)
- Fewer advanced widgets (not needed here)
- Threading complexities for long operations (handled manually)

---

# Part 3: End-to-End Walkthrough

This section narrates exactly what happens when a user performs each major action, identifying the specific code responsible.

## 3.1 Launching AnonVision

### Step 1: User runs `start_the_app_.py`

**What happens**:
1. Environment variables are set to disable GPU acceleration
2. `check_requirements()` verifies core dependencies exist
3. If missing, user is prompted to install
4. `check_optional()` checks for `face_recognition` library
5. Detection configuration is displayed
6. `from gui import main as run_app` imports the GUI
7. `run_app()` is called

**Key code** (`start_the_app_.py`):
```python
def main():
    # ... dependency checks ...
    from gui import main as run_app
    run_app()
```

### Step 2: GUI Initializes

**What happens**:
1. `tk.Tk()` creates the root window
2. `AnonVisionGUI(root)` constructs the application
3. In `__init__`:
   - `_init_style()` configures the dark theme
   - State variables are initialized to None/empty
   - `DNNFaceDetector` and `FaceDetector` are instantiated
   - `_build_ui()` constructs all panels
   - Status is set to "Ready"
4. `root.mainloop()` enters the event loop

**Key code** (`gui.py`):
```python
def main():
    root = tk.Tk()
    app = AnonVisionGUI(root)
    root.mainloop()
```

**What can fail**:
- If `cv2` isn't installed, import fails
- If YuNet model can't download, `self.dnn` is None (Haar fallback activates)
- If Haar cascade can't load, both detectors are None (detection won't work)

**Error handling**:
- Each detector is wrapped in try/except
- Failures are printed to console but don't crash the app
- UI disables DNN option if unavailable

## 3.2 Loading an Image

### User Action: Clicks "Select Image" button

**Chain of events**:

1. **`select_image()` is called** (`gui.py` line 556)
   
2. **File dialog opens**:
   ```python
   path = filedialog.askopenfilename(
       title="Select an image",
       filetypes=ft,
       initialdir=os.getcwd()
   )
   ```

3. **Path validation**:
   ```python
   if not self._validate_image(path):
       messagebox.showerror("Invalid File", "...")
       return
   ```
   - `_validate_image()` checks file exists
   - `validate_image_path()` from utils checks extension
   - `Image.open().verify()` checks file is actually readable

4. **Image loaded into memory**:
   ```python
   self.pil_image = Image.open(path).convert("RGB")
   self.img_bgr = cv2.imread(path)
   self.original_bgr = self.img_bgr.copy()
   ```
   - PIL image for potential processing
   - BGR numpy array for OpenCV operations
   - Pristine copy for reset capability

5. **State reset**:
   ```python
   self.detected_faces = []
   self.face_vars = []
   self.face_thumbs = []
   self._refresh_faces_list()
   ```

6. **Display updated**:
   ```python
   self._set_status(f"Loaded: {Path(path).name} — click Detect")
   self._refresh_display()
   ```

**Data entering**: File path string
**Data exiting**: Image stored in three formats in instance variables

**Why this matters**: The triple storage (PIL, BGR, original) enables:
- PIL for display in Tkinter (via ImageTk)
- BGR for OpenCV processing (detection, anonymization)
- Original for non-destructive editing (can always revert)

## 3.3 Running Face Detection

### User Action: Clicks "Detect Faces" button

**Chain of events**:

1. **`detect_faces()` is called** (`gui.py` line 584)

2. **Pre-condition check**:
   ```python
   if self.img_bgr is None:
       messagebox.showinfo("No image", "Load an image first.")
       return
   ```

3. **Detection method selected**:
   ```python
   method = self.detection_method.get()  # "dnn" or "haar"
   ```

4. **DNN path** (if method == "dnn" and self.dnn exists):
   ```python
   _, faces = self.dnn.detect_faces(
       self.img_bgr, 
       confidence_threshold=float(self.dnn_confidence.get())
   )
   ```
   
   Inside `DNNFaceDetector.detect_faces()`:
   - Image size is read
   - YuNet input size is set to match
   - `detector_yn.detect(img)` runs inference
   - Results are parsed: `[x, y, w, h, score, ...]`
   - Faces above threshold are collected
   - Boxes are clamped to image boundaries

5. **Haar path** (fallback):
   ```python
   gray = cv2.cvtColor(self.img_bgr, cv2.COLOR_BGR2GRAY)
   faces = self.haar.face_cascade.detectMultiScale(
       gray,
       scaleFactor=float(self.scale_factor.get()),
       minNeighbors=int(self.min_neighbors.get()),
       minSize=(int(self.min_size.get()), int(self.min_size.get()))
   )
   ```

6. **Results stored**:
   ```python
   self.detected_faces = [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]
   self.img_bgr = self.original_bgr.copy()  # Reset to original
   ```

7. **Face checklist built**:
   ```python
   self._build_face_checklist()
   ```
   - For each face, crop region from image
   - Resize to 48×48 thumbnail
   - Create ImageTk.PhotoImage
   - Create BooleanVar for protection toggle
   - Add checkbox widget with thumbnail

8. **Display updated**:
   ```python
   self._refresh_display()
   self._set_status(f"Detected: {len(self.detected_faces)} face(s)")
   ```

**Data entering**: BGR image array, detection parameters
**Data exiting**: List of bounding boxes, UI widgets for each face

**Why this sequence matters**:
- Image must be reset to original before redrawing boxes
- Thumbnails must be created from current image state
- BooleanVars must be created fresh (old ones would be stale)

## 3.4 Selecting Faces to Protect

### User Action: Checks/unchecks face checkboxes

**Chain of events**:

1. **Checkbox toggled**:
   - Tkinter updates `self.face_vars[idx]` automatically
   - Checkbox `command` callback triggers: `self._refresh_display`

2. **`_refresh_display()` redraws canvas**:
   ```python
   for idx, (x, y, w, h) in enumerate(self.detected_faces):
       is_protected = (idx < len(self.face_vars)) and bool(self.face_vars[idx].get())
       color = (16, 185, 129) if is_protected else (239, 68, 68)  # green : red
       cv2.rectangle(bgr_to_show, (x, y), (x + w, y + h), color, 2)
   ```

**Visual feedback**:
- Protected faces get **green** boxes
- Unprotected faces get **red** boxes
- Expanded anonymization region shown as gray outline for unprotected faces

**Why immediate visual feedback matters**:
- User confirms their selection is correct before anonymizing
- Reduces anxiety about making mistakes
- Builds trust through transparency

## 3.5 Applying Anonymization

### User Action: Clicks "Apply Anonymization" button

**Chain of events**:

1. **`apply_anonymization()` is called** (`gui.py` line 620)

2. **Pre-condition check**:
   ```python
   if self.original_bgr is None or not self.detected_faces:
       messagebox.showinfo("Nothing to do", "...")
       return
   ```

3. **Anonymization applied to fresh copy**:
   ```python
   self.img_bgr = self._anonymize_faces(self.original_bgr.copy())
   ```

4. **Inside `_anonymize_faces()`** (line 630):

   For each detected face:
   
   a. **Check if protected**:
   ```python
   is_protected = (idx < len(self.face_vars)) and bool(self.face_vars[idx].get())
   if is_protected:
       protected += 1
       continue  # Skip this face entirely
   ```
   
   b. **Calculate expanded region**:
   ```python
   expansion_pct = self.region_expansion.get() / 100.0
   ext_x = max(0, int(x - w * expansion_pct))
   ext_y = max(0, int(y - h * expansion_pct))
   ext_w = min(image.shape[1] - ext_x, int(w * (1 + 2 * expansion_pct)))
   ext_h = min(image.shape[0] - ext_y, int(h * (1 + 2 * expansion_pct)))
   ```
   
   c. **Extract region of interest**:
   ```python
   roi = out[ext_y:ext_y+ext_h, ext_x:ext_x+ext_w].copy()
   ```
   
   d. **Apply selected method**:
   
   **Blur**:
   ```python
   for _ in range(passes):
       anonymized_roi = cv2.GaussianBlur(anonymized_roi, (kernel, kernel), 0)
   ```
   
   **Pixelate**:
   ```python
   temp = cv2.resize(roi, (w_roi // pixel_size, h_roi // pixel_size))
   anonymized_roi = cv2.resize(temp, (w_roi, h_roi), interpolation=cv2.INTER_NEAREST)
   ```
   
   **Blackout**:
   ```python
   anonymized_roi = np.zeros_like(roi)
   ```
   
   e. **Apply feathering** (optional):
   ```python
   mask = self._create_feather_mask(ext_h, ext_w, feather_pct)
   blended = (anonymized_roi * mask_3ch + roi * (1 - mask_3ch)).astype(np.uint8)
   ```
   
   f. **Write back to image**:
   ```python
   out[ext_y:ext_y+ext_h, ext_x:ext_x+ext_w] = anonymized_roi
   ```

5. **Display updated**:
   ```python
   self._refresh_display()
   ```

**Data entering**: Original image, face boxes, protection flags, anonymization settings
**Data exiting**: Modified image with anonymization applied

**Why start from original_bgr.copy()**:
- Allows re-running with different settings without cumulative degradation
- Each application is idempotent: same inputs → same output

## 3.6 Exporting the Result

### User Action: Clicks "Save Result" button

**Chain of events**:

1. **`save_image()` is called** (`gui.py` line 725)

2. **Pre-condition check**:
   ```python
   if self.img_bgr is None:
       messagebox.showinfo("No image", "Nothing to save.")
       return
   ```

3. **File dialog opens**:
   ```python
   path = filedialog.asksaveasfilename(
       title="Save Image",
       defaultextension=".png",
       filetypes=[
           ("PNG (lossless)", "*.png"),
           ("JPEG", "*.jpg"),
           ("All files", "*.*")
       ]
   )
   ```

4. **Image written to disk**:
   ```python
   cv2.imwrite(path, self.img_bgr)
   ```

5. **Confirmation shown**:
   ```python
   self._set_status(f"Saved: {Path(path).name}")
   messagebox.showinfo("Saved", f"Saved to:\n{path}")
   ```

**Data entering**: Modified image array, user-chosen path
**Data exiting**: File written to disk

**Why PNG is default**:
- Lossless compression preserves anonymization quality
- JPEG artifacts could partially reveal anonymized features
- Transparency support if ever needed

---

# Part 4: Deep Class-by-Class Teaching Breakdown

## 4.1 AnonVisionGUI (gui.py)

### Purpose
The main application class that orchestrates the entire user interface and coordinates between detection modules and user actions. This is the "brain" of the application.

### Responsibilities
- Create and manage all UI widgets
- Handle all user interactions
- Coordinate face detection
- Apply anonymization transforms
- Manage application state
- Handle file I/O for images

### Dependencies
- `tkinter`: Window management, widgets, event handling
- `PIL.Image`, `PIL.ImageTk`: Image format conversion for display
- `cv2`: Image I/O and processing
- `numpy`: Array operations for anonymization
- `utils`: Image validation and resizing
- `dnn_detector.DNNFaceDetector`: Primary face detection
- `face_detector.FaceDetector`: Fallback face detection

### Who Depends on It
Nothing—this is the top-level application class.

### Key Attributes

| Attribute | Type | Purpose |
|-----------|------|---------|
| `root` | tk.Tk | The main window |
| `canvas` | tk.Canvas | Where images are displayed |
| `current_image_path` | str | Path to loaded image |
| `img_bgr` | np.ndarray | Working image (BGR format) |
| `original_bgr` | np.ndarray | Pristine original (never modified) |
| `detected_faces` | List[tuple] | Face bounding boxes |
| `face_vars` | List[tk.BooleanVar] | Protection toggles per face |
| `dnn` | DNNFaceDetector | Neural network detector |
| `haar` | FaceDetector | Cascade classifier detector |
| `anon_method` | tk.StringVar | "blur", "pixelate", or "blackout" |
| `colors` | dict | UI color palette |

### Key Methods

| Method | Purpose |
|--------|---------|
| `__init__(root)` | Initialize state, build UI, create detectors |
| `_init_style()` | Configure dark theme for ttk widgets |
| `_build_ui()` | Construct the three-panel layout |
| `_build_left_panel()` | Input controls and detection settings |
| `_build_center_panel()` | Image preview canvas |
| `_build_right_panel()` | Anonymization and export controls |
| `select_image()` | Open file dialog, load and validate image |
| `detect_faces()` | Run detection, store results, update UI |
| `apply_anonymization()` | Process unprotected faces |
| `_anonymize_faces(image)` | Core anonymization logic |
| `save_image()` | Write result to disk |
| `_refresh_display()` | Redraw canvas with current state |
| `_build_face_checklist()` | Create thumbnail+checkbox for each face |

### Control Flow Through This Class

```
User launches app
    │
    ▼
__init__()
    ├── _init_style()
    ├── Initialize state variables
    ├── Create DNNFaceDetector
    ├── Create FaceDetector (Haar)
    └── _build_ui()
            ├── _build_left_panel()
            ├── _build_center_panel()
            └── _build_right_panel()

User clicks "Select Image"
    │
    ▼
select_image()
    ├── filedialog.askopenfilename()
    ├── _validate_image()
    ├── cv2.imread()
    ├── Store images
    └── _refresh_display()

User clicks "Detect Faces"
    │
    ▼
detect_faces()
    ├── Check detection method
    ├── dnn.detect_faces() OR haar.detectMultiScale()
    ├── Store detected_faces
    ├── _build_face_checklist()
    └── _refresh_display()

User toggles face protection
    │
    ▼
BooleanVar updates → _refresh_display()

User clicks "Apply Anonymization"
    │
    ▼
apply_anonymization()
    ├── _anonymize_faces(original_bgr.copy())
    │       ├── For each face:
    │       │   ├── Check protected
    │       │   ├── Calculate expanded region
    │       │   ├── Apply blur/pixelate/blackout
    │       │   └── Optional feathering
    │       └── Return modified image
    └── _refresh_display()

User clicks "Save Result"
    │
    ▼
save_image()
    ├── filedialog.asksaveasfilename()
    ├── cv2.imwrite()
    └── messagebox.showinfo()
```

### Failure Cases & Defensive Design

| Failure | Defense |
|---------|---------|
| No image loaded when detecting | `if self.img_bgr is None: messagebox.showinfo(...)` |
| Invalid image file | `_validate_image()` checks file before loading |
| DNN unavailable | Falls back to `self.haar` |
| Both detectors unavailable | Error message, detection disabled |
| Empty face crop | `if crop_bgr.size == 0: raise ValueError` |
| Index out of bounds for face_vars | `(idx < len(self.face_vars)) and ...` |
| Save path not chosen | `if not path: return` |
| Save fails | `try/except` with `messagebox.showerror` |

### Why This Class Is Structured This Way

1. **Single Responsibility Violation (Intentional)**: GUI classes often handle multiple concerns because separating them adds complexity without proportional benefit for small applications.

2. **State in Instance Variables**: Allows any method to access current image, faces, settings without parameter passing.

3. **Tkinter Variables**: Using `tk.IntVar`, `tk.StringVar`, etc., enables automatic widget binding—when the variable changes, the widget updates.

4. **Original Preservation**: Keeping `original_bgr` separate from `img_bgr` enables non-destructive editing.

### What Would Break If It Were Removed
Everything. This is the main application.

### How to Safely Extend It

1. **Adding new anonymization methods**:
   - Add radio button in `_build_anon_settings()`
   - Add case in `_anonymize_faces()` switch
   - Optionally add method-specific settings

2. **Adding whitelist integration**:
   - Import `FaceWhitelist` at top
   - Create instance in `__init__`
   - Add UI panel for whitelist management
   - Call `whitelist.match_detected_faces()` during detection
   - Set protection based on match results

3. **Adding video support**:
   - Add video state variables
   - Add frame extraction logic
   - Add navigation controls
   - Process frames sequentially

---

## 4.2 DNNFaceDetector (dnn_detector.py)

### Purpose
Provides high-accuracy face detection using the YuNet deep neural network model. Designed to be the primary detector with graceful degradation to Haar Cascade if neural network loading fails.

### Responsibilities
- Download YuNet ONNX model if not present
- Initialize OpenCV's `FaceDetectorYN` API
- Detect faces with configurable confidence threshold
- Clamp detected boxes to image boundaries
- Fall back to Haar Cascade if DNN fails

### Dependencies
- `cv2`: OpenCV for detection API
- `numpy`: Array handling
- `urllib.request`: Model download
- `pathlib.Path`: File path handling

### Who Depends on It
- `AnonVisionGUI` instantiates and calls `detect_faces()`

### Key Attributes

| Attribute | Type | Purpose |
|-----------|------|---------|
| `data_dir` | str | Directory for model storage |
| `model_path` | str | Full path to ONNX model file |
| `score_threshold` | float | Minimum confidence (0.0-1.0) |
| `nms_threshold` | float | Non-max suppression threshold |
| `detector_yn` | cv2.FaceDetectorYN | The neural network detector |
| `cascade` | cv2.CascadeClassifier | Haar fallback (or None) |

### Key Methods

| Method | Signature | Purpose |
|--------|-----------|---------|
| `__init__` | `(data_dir, score_threshold, nms_threshold)` | Initialize detector, download model |
| `set_score_threshold` | `(thr: float)` | Update confidence threshold |
| `detect_faces` | `(bgr_img, confidence_threshold) → (img, boxes)` | Run detection on image |

### Control Flow Through This Class

```
__init__(data_dir="data", score_threshold=0.5, nms_threshold=0.3)
    │
    ├── Create data directory
    ├── Set model path
    │
    ├── Try: Initialize YuNet
    │   ├── If model missing: Download from GitHub
    │   └── cv2.FaceDetectorYN_create(...)
    │       └── If fails: detector_yn = None
    │
    ├── If YuNet failed: Try Haar fallback
    │   └── cv2.CascadeClassifier(haarcascades + ...)
    │
    └── Disable OpenCL (CPU-only mode)

detect_faces(bgr_img, confidence_threshold=None)
    │
    ├── Validate input (not None, not empty)
    │
    ├── If detector_yn exists:
    │   ├── Set input size to image dimensions
    │   ├── Run detector_yn.detect(img)
    │   ├── Parse results: [x, y, w, h, score, ...]
    │   ├── Filter by confidence
    │   ├── Clamp boxes to image bounds
    │   └── Return (img, faces)
    │
    ├── Elif cascade exists (fallback):
    │   ├── Convert to grayscale
    │   ├── Run detectMultiScale
    │   └── Return (img, faces)
    │
    └── Else: Return (img, [])  # No detector available
```

### Failure Cases & Defensive Design

| Failure | Defense |
|---------|---------|
| Model download fails | Silently continues; Haar fallback |
| YuNet initialization fails | `try/except`, sets `detector_yn = None` |
| Haar cascade file missing | Falls back to OpenCV's bundled version |
| Both methods fail | Returns empty face list |
| Input image is None | Returns immediately |
| Box coordinates negative | `max(0, ...)` clamping |
| Box extends past image | `min(w, w-x)` clamping |

### Why This Class Is Structured This Way

1. **Fallback Pattern**: YuNet is better but less reliable. Haar is worse but always works. Having both ensures detection works everywhere.

2. **Model Auto-Download**: Users shouldn't need to manually download models. The code fetches what it needs.

3. **CPU-Only Mode**: Explicitly disabling GPU avoids cryptic errors on machines without proper GPU drivers.

4. **Threshold Configurability**: Different images need different sensitivity. Allowing runtime adjustment enables tuning without code changes.

### What Would Break If It Were Removed
- Primary detection would fail
- `gui.py` would fall back to using `FaceDetector` only
- Detection accuracy would decrease
- User would lose DNN option in UI

### How to Safely Extend It

1. **Supporting additional DNN models**:
   - Add model URL and path
   - Add model selection logic
   - Adjust result parsing for model's output format

2. **Adding GPU support**:
   - Remove `setUseOpenCL(False)`
   - Add backend/target configuration
   - Add fallback if GPU fails

3. **Adding batch processing**:
   - Modify `detect_faces` to accept list of images
   - Batch inference for efficiency

---

## 4.3 FaceDetector (face_detector.py)

### Purpose
A simpler face detection class using OpenCV's Haar Cascade classifier. Designed as a standalone detector and fallback option.

### Responsibilities
- Download Haar Cascade XML if not present
- Provide face detection via `detectMultiScale`
- Draw detection boxes on images (utility function)

### Dependencies
- `cv2`: OpenCV for cascade classifier
- `urllib.request`: Cascade file download
- `os`: File path operations

### Who Depends on It
- `AnonVisionGUI` uses it as fallback detector

### Key Attributes

| Attribute | Type | Purpose |
|-----------|------|---------|
| `face_cascade` | cv2.CascadeClassifier | The Haar classifier |

### Key Methods

| Method | Signature | Purpose |
|--------|-----------|---------|
| `__init__` | `()` | Load Haar cascade |
| `_get_cascade_path` | `()` | Find or create path to XML |
| `_download_cascade` | `(path)` | Download XML from GitHub |
| `detect_faces` | `(image_path) → (image, faces)` | Detect faces in file |
| `draw_detection_boxes` | `(image, faces, ...)` | Draw rectangles on image |

### Code Issue Alert

There's a bug in `draw_detection_boxes()`. The loop doesn't properly indent the label drawing:

```python
for i, (x, y, w, h) in enumerate(faces):
   cv2.rectangle(image_copy, (x, y), (x+w, y+h), color, thickness)
        
# This code runs ONCE after the loop, not for each face:
if show_label:
    label = f"Face {i+1}"  # Uses last value of i
    # ...drawing code...
```

**Should be**:
```python
for i, (x, y, w, h) in enumerate(faces):
    cv2.rectangle(image_copy, (x, y), (x+w, y+h), color, thickness)
        
    if show_label:
        label = f"Face {i+1}"
        # ...drawing code...
```

### Why This Class Exists Alongside DNNFaceDetector

1. **Historical**: May have been the original detector before DNN was added
2. **Standalone Testing**: Can be used independently of the full app
3. **File-Based API**: Takes image path, useful for batch processing
4. **Drawing Utility**: Provides box drawing that DNN detector doesn't

### Relationship to DNNFaceDetector

```
             ┌─────────────────────┐
             │  AnonVisionGUI      │
             └─────────┬───────────┘
                       │
         ┌─────────────┴─────────────┐
         │                           │
         ▼                           ▼
┌─────────────────┐        ┌─────────────────┐
│ DNNFaceDetector │        │  FaceDetector   │
├─────────────────┤        ├─────────────────┤
│ Primary         │        │ Fallback        │
│ YuNet DNN       │        │ Haar Cascade    │
│ Takes array     │        │ Takes file path │
│ More accurate   │        │ Faster          │
│ Needs model DL  │        │ Built into CV   │
└────────┬────────┘        └────────┬────────┘
         │                          │
         │ Also has Haar            │
         │ fallback internally      │
         └──────────────────────────┘
```

---

## 4.4 FaceWhitelist (face_whitelist.py)

### Purpose
Manages a database of known faces that should remain visible (not anonymized). Provides face recognition/matching to automatically protect known individuals.

### Responsibilities
- Store face images with associated names
- Generate and store face thumbnails
- Match detected faces against stored faces
- Support both `face_recognition` library (accurate) and ORB feature matching (fallback)

### Dependencies
- `cv2`: Image I/O, ORB features
- `numpy`: Array operations
- `json`: Database persistence
- `face_recognition` (optional): Deep learning face matching

### Who Depends on It
- **Currently**: Nothing in provided code
- **Intended**: `AnonVisionGUI` should import and use this

### Key Attributes

| Attribute | Type | Purpose |
|-----------|------|---------|
| `db` | Dict[str, Dict] | Maps name → {image path, thumbnail path} |

### Key Methods

| Method | Signature | Purpose |
|--------|-----------|---------|
| `__init__` | `()` | Load database from JSON |
| `save` | `()` | Write database to JSON |
| `get_whitelisted_names` | `() → List[str]` | List all known names |
| `get_thumbnail_path` | `(name) → str` | Get thumbnail for name |
| `add_face` | `(image_path, name) → bool` | Register new face |
| `remove_face` | `(name)` | Delete face from database |
| `match_detected_faces` | `(bgr_img, bboxes, threshold) → List[dict]` | Match faces against whitelist |

### Match Result Format

```python
{
    'bbox': (x, y, w, h),           # Face location
    'is_whitelisted': True/False,   # Should be protected?
    'matched_name': "Alice" or None, # Who was matched
    'confidence': 0.85              # Match confidence
}
```

### Recognition Strategy

The class uses a tiered approach:

**Tier 1: face_recognition library** (if installed)
- Uses deep learning face embeddings
- Computes 128-dimensional face encoding
- Compares via cosine similarity
- High accuracy (>95%)

**Tier 2: ORB feature matching** (fallback)
- Uses OpenCV's ORB keypoint detector
- Matches keypoints between face patches
- Computes similarity from match distances
- Lower accuracy (~60-70%)

```
match_detected_faces(image, bboxes, threshold=0.60)
    │
    ├── If whitelist empty → All faces unmatched
    │
    ├── Prepare whitelist encodings:
    │   ├── If face_recognition available:
    │   │   └── Compute 128-d encoding for each stored face
    │   └── Else:
    │       └── Store grayscale patches for ORB matching
    │
    ├── For each detected face:
    │   ├── If face_recognition available:
    │   │   ├── Encode detected face
    │   │   └── Compare to all whitelist encodings (cosine sim)
    │   └── Else:
    │   │   ├── Convert to grayscale
    │   │   └── ORB match against whitelist patches
    │   │
    │   ├── Find best match
    │   └── If confidence >= threshold → Whitelisted
    │
    └── Return list of match results
```

### Integration Guide

To connect `FaceWhitelist` to the GUI:

```python
# In gui.py __init__():
from face_whitelist import FaceWhitelist
self.whitelist = FaceWhitelist()

# In detect_faces(), after detection:
match_results = self.whitelist.match_detected_faces(
    self.img_bgr, 
    self.detected_faces
)
for idx, result in enumerate(match_results):
    if result['is_whitelisted']:
        self.face_vars[idx].set(True)  # Auto-protect matched faces
```

---

## 4.5 utils.py

### Purpose
Shared utility functions for image manipulation used across modules.

### Key Functions

| Function | Signature | Purpose |
|----------|-----------|---------|
| `_to_pil` | `(image) → PIL.Image` | Convert numpy to PIL |
| `_to_numpy_rgb` | `(image) → np.ndarray` | Convert PIL to numpy RGB |
| `resize_image_for_display` | `(image, max_width, max_height) → PIL.Image` | Aspect-fit resize |
| `validate_image_path` | `(path) → bool` | Check if path is valid image |
| `hex_to_bgr` | `(hex_color) → (b, g, r)` | Convert hex color to BGR |
| `clamp_box` | `(x, y, w, h, img_w, img_h) → tuple` | Keep box within image bounds |

### Why These Functions Are Centralized

1. **DRY Principle**: Multiple modules need image format conversion
2. **Consistency**: Same conversion logic everywhere
3. **Testing**: Can test utilities independently
4. **Maintainability**: Fix once, fixed everywhere

---

# Part 5: Computer Vision Logic (The "Why")

## 5.1 Why Face Detection Is Separate From Recognition

These are fundamentally different problems:

**Detection**: "Where are faces in this image?"
- Input: Image
- Output: Bounding boxes (x, y, width, height)
- Doesn't know *who* each face is
- Relatively fast
- Works on unknown faces

**Recognition**: "Whose face is this?"
- Input: Face image
- Output: Identity or "unknown"
- Requires database of known faces
- More computationally expensive
- Fails gracefully for unknown faces

**Why separate them in code?**

1. **Different algorithms**: Detection uses sliding windows or neural networks. Recognition uses embeddings and similarity.

2. **Different update cycles**: Detection models improve independently from recognition.

3. **Optional recognition**: Many use cases only need detection (count faces, blur all faces).

4. **Pipeline efficiency**: Detect once, then only recognize the detected regions.

## 5.2 Why YuNet / Haar / Ensemble Approaches

### YuNet (Deep Neural Network)

**How it works**: A convolutional neural network trained on millions of face images. Learns hierarchical features (edges → textures → face parts → full faces).

**Pros**:
- High accuracy (~95%+ on standard benchmarks)
- Handles varied poses, lighting, occlusions
- Detects small faces
- Modern and actively maintained

**Cons**:
- Requires model file download
- Slower than classical methods (though still fast)
- May fail on unusual OpenCV builds

### Haar Cascade

**How it works**: A cascade of simple classifiers using Haar-like features (contrast patterns). Each stage quickly rejects non-face regions.

**Pros**:
- Very fast (designed for real-time video)
- Built into OpenCV (no download)
- Works everywhere
- Well-understood behavior

**Cons**:
- Lower accuracy (~80%)
- Struggles with non-frontal faces
- Many false positives/negatives
- Older technology (2001!)

### Why Provide Both?

```
                      ┌─────────────────┐
                      │   User's Image  │
                      └────────┬────────┘
                               │
                      ┌────────▼────────┐
                      │  Try YuNet DNN  │
                      └────────┬────────┘
                               │
               ┌───────────────┼───────────────┐
               │ Success       │               │ Failure
               ▼               │               ▼
        ┌──────────────┐       │        ┌──────────────┐
        │ High Accuracy│       │        │ Use Haar     │
        │ Detection    │       │        │ Fallback     │
        └──────────────┘       │        └──────────────┘
                               │               │
                               │               ▼
                               │        ┌──────────────┐
                               │        │ Lower But    │
                               │        │ Working Det. │
                               │        └──────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  User Never Sees    │
                    │  Complete Failure   │
                    └─────────────────────┘
```

**Design principle**: Graceful degradation over catastrophic failure.

## 5.3 How Confidence Thresholds Work

Every face detector outputs not just locations but also **confidence scores**—how certain the model is that a region contains a face.

```
Detection Output:
┌──────┬──────┬───────┬────────┬────────────┐
│  x   │  y   │   w   │   h    │ confidence │
├──────┼──────┼───────┼────────┼────────────┤
│ 100  │ 50   │  80   │  100   │   0.95     │  ← Almost certainly a face
│ 300  │ 200  │  60   │   70   │   0.72     │  ← Probably a face
│ 450  │ 30   │  40   │   45   │   0.35     │  ← Maybe a face?
│ 600  │ 400  │  25   │   30   │   0.12     │  ← Probably not a face
└──────┴──────┴───────┴────────┴────────────┘
```

The threshold setting determines the cutoff:

```python
threshold = 0.50  # Only accept detections with confidence >= 50%

# With threshold 0.50:
# ✓ 0.95 → Accepted
# ✓ 0.72 → Accepted  
# ✗ 0.35 → Rejected
# ✗ 0.12 → Rejected
```

### Threshold Tradeoffs

| Low Threshold (0.3) | High Threshold (0.8) |
|---------------------|----------------------|
| Catches more faces | Misses some faces |
| More false positives | Fewer false positives |
| Better recall | Better precision |
| Good for: "Don't miss anyone" | Good for: "Only sure faces" |

## 5.4 Why False Positives Happen

False positives are regions detected as faces that aren't actually faces. Common causes:

1. **Face-like patterns**: Electrical outlets, car grills, tree knots, round objects with "eyes"

2. **Trained bias**: Models overfit to training data patterns

3. **Low thresholds**: Accepting uncertain detections

4. **Texture similarity**: Patterns resembling skin tones and facial geometry

**Mitigation strategies used in AnonVision**:
- Configurable confidence threshold
- User review before anonymization
- Protection toggling to correct mistakes

## 5.5 Why Whitelist Recognition Is Safer Than Blacklist

### Blacklist Approach: "Hide These Faces"
```
Input: Image + List of faces to hide
If face matches blacklist → Blur it
Else → Leave visible

Failure mode: If match fails, face is visible
Risk: Privacy breach for intended targets
```

### Whitelist Approach: "Keep These Faces Visible"
```
Input: Image + List of faces to keep
If face matches whitelist → Keep visible
Else → Blur it

Failure mode: If match fails, face is blurred
Risk: Over-anonymization (intended subjects hidden)
```

**Why whitelist is safer for privacy**:

1. **Fails closed**: Unknown faces are protected by default
2. **Conservative**: When in doubt, protect privacy
3. **Proportional**: The few known people are identifiable; the many unknown are protected
4. **Consent-based**: Only explicitly approved faces remain visible

## 5.6 Accuracy vs. Speed vs. Privacy Tradeoffs

```
                    FAST ◄────────────────────► ACCURATE
                      │                              │
    Haar Cascade     ●│                              │● Deep CNN
    (Real-time)       │                              │  (Batch)
                      │                              │
                      │         ●                    │
                      │    YuNet                     │
                      │   (Balanced)                 │
                      │                              │
    ──────────────────┼──────────────────────────────┼──────────────
                      │                              │
              Higher  │                              │  Higher
              False   │                              │  Compute
              Negatives│                             │  Cost
```

**AnonVision's choice**: YuNet provides the best balance—accurate enough for high-stakes privacy protection, fast enough for interactive use.

---

# Part 6: UI Logic & Human Factors

## 6.1 Panel Structure Philosophy

The UI uses a three-panel layout that mirrors the user's mental model of the workflow:

```
┌───────────────┬─────────────────────┬───────────────────┐
│  1. PREPARE   │     2. PREVIEW      │  3. ANONYMIZE &   │
│               │                     │     EXPORT        │
│  • Load image │                     │                   │
│  • Configure  │    [Image Canvas]   │  • Settings       │
│  • Detect     │                     │  • Protect faces  │
│               │                     │  • Save           │
└───────────────┴─────────────────────┴───────────────────┘
```

**Why this layout works**:

1. **Left-to-right flow**: Matches natural reading direction (in LTR languages)
2. **Progressive disclosure**: Each panel represents a step in the process
3. **Central focus**: The image (most important element) gets the most space
4. **Minimal decisions**: Each panel has a clear purpose

## 6.2 Event Flow: Button → Handler → Processing → Update

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│    Button    │────▶│   Handler    │────▶│  Processing  │────▶│  UI Update   │
│   Clicked    │     │   Called     │     │   Runs       │     │   Triggers   │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
       │                    │                    │                    │
       │              detect_faces()        cv2 detection       _refresh_display()
       │                    │                    │                    │
       │                    │                    │              Canvas redrawn
       │                    │                    │              Status updated
       │                    │                    │              Widgets created
       │                    │                    │
     User              Application            OpenCV              Tkinter
    Action              Logic                Library            Framework
```

**Separation of concerns**:
- User only sees buttons and results
- Handler coordinates but doesn't do CV work
- Processing is isolated in detector classes
- UI updates are centralized in `_refresh_display()`

## 6.3 Why Blocking Dialogs Were Minimized

The application avoids blocking dialogs (modal popups that halt everything) except when absolutely necessary:

**Used for**:
- Error messages (user must acknowledge)
- Save confirmations (user must confirm path)
- Critical warnings

**Not used for**:
- Progress feedback (status bar instead)
- Informational messages (status bar)
- Settings changes (immediate visual feedback)

**Why this matters**:
- Dialogs break flow
- Users develop "dialog blindness"
- Status bar is less intrusive
- Immediate canvas feedback is more intuitive

## 6.4 Progress and Feedback Communication

The application provides feedback through multiple channels:

### Status Bar (Bottom Right)
```
"Ready — load an image to begin"
"Loaded: photo.jpg — click Detect"
"Detecting faces (dnn)..."
"Detected: 5 face(s)"
"✓ Anonymized: 3 | Protected: 2 | Method: Heavy Blur"
"Saved: result.png"
```

**Design**: Single line, always visible, minimal but informative.

### Visual Feedback on Canvas
- Green boxes = Protected (will stay visible)
- Red boxes = Not protected (will be anonymized)
- Gray outline = Expanded anonymization region

**Design**: Color coding provides instant understanding without reading.

### Face Thumbnails
- 48×48 previews of each detected face
- Checkbox next to each for protection toggle
- Scroll area for many faces

**Design**: Users see exactly what was detected, can verify correctness.

## 6.5 Trust and Transparency

Privacy tools require user trust. AnonVision builds trust through:

### Showing Before Applying
Users see detection boxes before anonymization runs. They can verify:
- All faces were found
- No faces were missed
- Correct faces are protected

### Non-Destructive Editing
- Original image is preserved
- Can re-run with different settings
- Can change protection and re-apply
- Only final save is permanent

### Visible Settings
- All anonymization parameters are exposed
- Users can see exactly what will happen
- No hidden "magic"

### Preview of Anonymization Region
- Gray outline shows the expanded area
- Users understand what "region expansion" means
- No surprises about coverage

---

# Part 7: Rebuild-From-Scratch Tutorial

This section guides you through rebuilding AnonVision step by step.

## 7.1 Environment Setup

### Step 1: Python Environment

```bash
# Create project directory
mkdir anonvision
cd anonvision

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (macOS/Linux)
source venv/bin/activate
```

### Step 2: Install Dependencies

```bash
# Core dependencies
pip install opencv-contrib-python pillow numpy

# Optional (for high-accuracy whitelist matching)
pip install face_recognition
```

### Step 3: Verify Installation

```python
# test_setup.py
import cv2
import numpy as np
from PIL import Image
import tkinter as tk

print(f"OpenCV version: {cv2.__version__}")
print(f"NumPy version: {np.__version__}")
print("Tkinter available: Yes")
print("Setup complete!")
```

**Success looks like**:
```
OpenCV version: 4.8.0
NumPy version: 1.24.3
Tkinter available: Yes
Setup complete!
```

**Common mistakes**:
- Installing `opencv-python` instead of `opencv-contrib-python` (missing YuNet support)
- Forgetting to activate virtual environment
- Python 2 instead of Python 3

## 7.2 Project Structure

Create this folder structure:

```
anonvision/
├── start_the_app.py      # Entry point
├── gui.py                # Main application
├── dnn_detector.py       # YuNet detector
├── face_detector.py      # Haar detector
├── face_whitelist.py     # Recognition (optional)
├── utils.py              # Utilities
├── data/                 # (auto-created)
└── whitelist_db/         # (auto-created)
```

## 7.3 Minimal UI Shell

Start with a working window:

```python
# gui.py - Phase 1: Window
import tkinter as tk
from tkinter import ttk

class AnonVisionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("AnonVision")
        self.root.geometry("1200x700")
        
        # Simple placeholder
        label = ttk.Label(root, text="AnonVision - Coming Soon")
        label.pack(expand=True)

def main():
    root = tk.Tk()
    app = AnonVisionGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
```

**What success looks like**: A window appears with text.

**Why this matters**: Verify tkinter works before adding complexity.

## 7.4 Add Three-Panel Layout

```python
# gui.py - Phase 2: Layout
import tkinter as tk
from tkinter import ttk

class AnonVisionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("AnonVision")
        self.root.geometry("1200x700")
        
        # Main container
        main = tk.Frame(root, bg="#1a1a1e")
        main.pack(fill="both", expand=True)
        main.columnconfigure(1, weight=1)  # Center expands
        main.rowconfigure(0, weight=1)
        
        # Left panel
        left = tk.Frame(main, bg="#232328", width=250)
        left.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        ttk.Label(left, text="1. Prepare").pack(pady=10)
        
        # Center panel
        center = tk.Frame(main, bg="#1e1e22")
        center.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        self.canvas = tk.Canvas(center, bg="#1e1e22")
        self.canvas.pack(fill="both", expand=True)
        
        # Right panel
        right = tk.Frame(main, bg="#232328", width=280)
        right.grid(row=0, column=2, sticky="nsew", padx=5, pady=5)
        ttk.Label(right, text="3. Anonymize").pack(pady=10)

def main():
    root = tk.Tk()
    app = AnonVisionGUI(root)
    root.mainloop()
```

**What success looks like**: Three distinct panels, center is largest.

## 7.5 Add Image Loading

```python
# Add to gui.py
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2

class AnonVisionGUI:
    def __init__(self, root):
        # ... previous code ...
        
        # State
        self.img_bgr = None
        self.photo_image = None
        
        # Add load button in left panel
        load_btn = ttk.Button(left, text="Load Image", command=self.load_image)
        load_btn.pack(pady=5, padx=10, fill="x")
    
    def load_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp")]
        )
        if not path:
            return
            
        # Load with OpenCV (BGR format)
        self.img_bgr = cv2.imread(path)
        if self.img_bgr is None:
            messagebox.showerror("Error", "Could not load image")
            return
        
        # Convert for display
        rgb = cv2.cvtColor(self.img_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        
        # Resize to fit canvas
        canvas_w = self.canvas.winfo_width() or 800
        canvas_h = self.canvas.winfo_height() or 600
        pil.thumbnail((canvas_w - 20, canvas_h - 20))
        
        # Display
        self.photo_image = ImageTk.PhotoImage(pil)
        self.canvas.delete("all")
        self.canvas.create_image(
            canvas_w // 2, canvas_h // 2,
            image=self.photo_image,
            anchor="center"
        )
```

**What success looks like**: Button opens file dialog, image displays in center.

**Common mistakes**:
- Forgetting to keep reference to `PhotoImage` (garbage collected → blank canvas)
- Using RGB instead of BGR with OpenCV (colors wrong)

## 7.6 Add Face Detection

```python
# dnn_detector.py - Minimal version
import cv2
import numpy as np
from pathlib import Path
import urllib.request

class DNNFaceDetector:
    MODEL_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
    
    def __init__(self):
        Path("data").mkdir(exist_ok=True)
        model_path = "data/face_detection_yunet_2023mar.onnx"
        
        # Download if needed
        if not Path(model_path).exists():
            print("Downloading YuNet model...")
            urllib.request.urlretrieve(self.MODEL_URL, model_path)
        
        # Create detector
        self.detector = cv2.FaceDetectorYN_create(
            model_path, "", (320, 320),
            score_threshold=0.5,
            nms_threshold=0.3
        )
    
    def detect(self, image):
        h, w = image.shape[:2]
        self.detector.setInputSize((w, h))
        
        _, faces = self.detector.detect(image)
        
        if faces is None:
            return []
        
        # Extract bounding boxes
        boxes = []
        for face in faces:
            x, y, w, h = int(face[0]), int(face[1]), int(face[2]), int(face[3])
            boxes.append((x, y, w, h))
        
        return boxes
```

**Integration in gui.py**:
```python
from dnn_detector import DNNFaceDetector

class AnonVisionGUI:
    def __init__(self, root):
        # ... previous code ...
        
        self.detector = DNNFaceDetector()
        self.detected_faces = []
        
        # Add detect button
        detect_btn = ttk.Button(left, text="Detect Faces", command=self.detect_faces)
        detect_btn.pack(pady=5, padx=10, fill="x")
    
    def detect_faces(self):
        if self.img_bgr is None:
            messagebox.showinfo("Info", "Load an image first")
            return
        
        self.detected_faces = self.detector.detect(self.img_bgr)
        self.refresh_display()
        print(f"Found {len(self.detected_faces)} faces")
    
    def refresh_display(self):
        if self.img_bgr is None:
            return
        
        # Draw boxes on copy
        display_img = self.img_bgr.copy()
        for (x, y, w, h) in self.detected_faces:
            cv2.rectangle(display_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Convert and display
        rgb = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        
        canvas_w = self.canvas.winfo_width() or 800
        canvas_h = self.canvas.winfo_height() or 600
        pil.thumbnail((canvas_w - 20, canvas_h - 20))
        
        self.photo_image = ImageTk.PhotoImage(pil)
        self.canvas.delete("all")
        self.canvas.create_image(
            canvas_w // 2, canvas_h // 2,
            image=self.photo_image,
            anchor="center"
        )
```

## 7.7 Add Anonymization

```python
# Add to AnonVisionGUI class

def __init__(self, root):
    # ... previous code ...
    
    self.face_protected = []  # Track which faces to protect
    
    # Add anonymize button in right panel
    anon_btn = ttk.Button(right, text="Apply Blur", command=self.apply_anonymization)
    anon_btn.pack(pady=5, padx=10, fill="x")

def apply_anonymization(self):
    if self.img_bgr is None or not self.detected_faces:
        return
    
    result = self.img_bgr.copy()
    
    for idx, (x, y, w, h) in enumerate(self.detected_faces):
        # Skip protected faces
        if idx < len(self.face_protected) and self.face_protected[idx]:
            continue
        
        # Extract region
        roi = result[y:y+h, x:x+w]
        
        # Apply heavy blur
        for _ in range(5):
            roi = cv2.GaussianBlur(roi, (51, 51), 0)
        
        # Replace region
        result[y:y+h, x:x+w] = roi
    
    self.img_bgr = result
    self.refresh_display()
```

## 7.8 Add Face Protection UI

```python
# Add to AnonVisionGUI

def __init__(self, root):
    # ... previous code ...
    
    # Faces list in right panel
    self.faces_frame = ttk.Frame(right)
    self.faces_frame.pack(fill="both", expand=True, padx=10, pady=10)
    self.face_vars = []  # BooleanVars for checkboxes

def detect_faces(self):
    # ... detection code ...
    
    self.build_face_list()

def build_face_list(self):
    # Clear old widgets
    for widget in self.faces_frame.winfo_children():
        widget.destroy()
    
    self.face_vars = []
    
    for idx, (x, y, w, h) in enumerate(self.detected_faces):
        var = tk.BooleanVar(value=False)
        self.face_vars.append(var)
        
        cb = ttk.Checkbutton(
            self.faces_frame,
            text=f"Protect Face {idx + 1}",
            variable=var,
            command=self.refresh_display
        )
        cb.pack(anchor="w", pady=2)

def apply_anonymization(self):
    # ... (modify to check self.face_vars[idx].get()) ...
    for idx, (x, y, w, h) in enumerate(self.detected_faces):
        if idx < len(self.face_vars) and self.face_vars[idx].get():
            continue  # Skip protected
        # ... blur logic ...
```

## 7.9 Add Save Functionality

```python
def __init__(self, root):
    # ... previous code ...
    
    save_btn = ttk.Button(right, text="Save Image", command=self.save_image)
    save_btn.pack(pady=5, padx=10, fill="x")

def save_image(self):
    if self.img_bgr is None:
        messagebox.showinfo("Info", "Nothing to save")
        return
    
    path = filedialog.asksaveasfilename(
        defaultextension=".png",
        filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")]
    )
    
    if path:
        cv2.imwrite(path, self.img_bgr)
        messagebox.showinfo("Saved", f"Saved to {path}")
```

## 7.10 Validation Checklist

Before considering the rebuild complete, verify:

- [ ] Application launches without errors
- [ ] Image loads and displays correctly
- [ ] Face detection finds faces in test images
- [ ] Detection boxes appear in correct positions
- [ ] Checkboxes can be toggled
- [ ] Protected faces are NOT anonymized
- [ ] Unprotected faces ARE anonymized
- [ ] Save dialog works
- [ ] Saved image has anonymization applied
- [ ] Can load a new image and start over
- [ ] Works with images of different sizes
- [ ] Handles images with no faces gracefully
- [ ] Handles images with many faces (10+)

---

# Part 8: Debugging & Failure Playbook

## 8.1 No Faces Detected

### Symptoms
- "Detected: 0 face(s)" on images clearly containing faces
- No boxes appear on canvas

### Root Causes

| Cause | Diagnosis | Fix |
|-------|-----------|-----|
| Confidence too high | Try lowering threshold | Set DNN confidence to 0.3 |
| Faces too small | Check face size in pixels | Lower min_size to 20 |
| Faces not frontal | Haar struggles with profiles | Use DNN method |
| Model not loaded | Check console for errors | Redownload model |
| Image not loaded | `self.img_bgr` is None | Verify file dialog |

### Debugging Steps

```python
# Add to detect_faces() for debugging
print(f"Image shape: {self.img_bgr.shape}")
print(f"Detector method: {self.detection_method.get()}")
print(f"DNN confidence: {self.dnn_confidence.get()}")
print(f"DNN detector exists: {self.dnn is not None}")
```

### Prevention
- Add validation before detection
- Show helpful error messages
- Suggest parameter adjustments in UI

## 8.2 Wrong Faces Anonymized

### Symptoms
- Protected face was blurred
- Unprotected face was left visible
- Some faces treated incorrectly

### Root Causes

| Cause | Diagnosis | Fix |
|-------|-----------|-----|
| Index mismatch | `face_vars` out of sync | Rebuild checklist after detection |
| Stale state | Re-ran detection without reset | Clear `face_vars` before detection |
| Off-by-one error | Check loop indices | Verify `enumerate()` usage |

### Debugging Steps

```python
# Add to apply_anonymization()
for idx, (x, y, w, h) in enumerate(self.detected_faces):
    protected = self.face_vars[idx].get() if idx < len(self.face_vars) else False
    print(f"Face {idx}: protected={protected}, bbox={x},{y},{w},{h}")
```

### Prevention
- Always use `idx < len(self.face_vars)` guard
- Reset state on new detection
- Visual indicators (green/red boxes) before applying

## 8.3 Whitelist Not Working

### Symptoms
- Known faces not automatically protected
- Match confidence always low
- No matches despite having whitelist entries

### Root Causes

| Cause | Diagnosis | Fix |
|-------|-----------|-----|
| `face_recognition` not installed | Check import | `pip install face_recognition` |
| Whitelist image poor quality | Check stored images | Use clear, frontal photos |
| Threshold too high | Check match confidence | Lower to 0.45 |
| ORB fallback inaccurate | No `face_recognition` | Install the library |

### Debugging Steps

```python
# In match_detected_faces()
print(f"Using face_recognition: {_HAS_FR}")
for name, enc in whitelist_encs.items():
    print(f"Whitelist: {name}, encoding shape: {enc.shape}")

for bb in bboxes:
    print(f"Detected face: {bb}, best_conf: {best_conf}, match: {best_name}")
```

## 8.4 Slow Performance

### Symptoms
- UI freezes during detection
- Long pause when applying anonymization
- Application feels unresponsive

### Root Causes

| Cause | Diagnosis | Fix |
|-------|-----------|-----|
| Large image | Check `img.shape` | Resize before processing |
| GPU disabled | By design | Consider enabling for speed |
| Many faces | Check face count | Limit processing or batch |
| Blur passes too high | Check settings | Reduce to 3-4 passes |

### Optimization Strategies

```python
# Resize large images for detection
MAX_DIM = 1920
h, w = image.shape[:2]
if max(h, w) > MAX_DIM:
    scale = MAX_DIM / max(h, w)
    small = cv2.resize(image, None, fx=scale, fy=scale)
    faces = detector.detect(small)
    # Scale boxes back up
    faces = [(int(x/scale), int(y/scale), int(w/scale), int(h/scale)) 
             for (x, y, w, h) in faces]
```

## 8.5 UI Freezing

### Symptoms
- Window becomes unresponsive
- "Not Responding" in title bar
- Buttons don't react during processing

### Root Cause
Long-running operations block the main thread. Tkinter can only update UI when `mainloop()` gets control.

### Solutions

**Option 1: Update during processing**
```python
def detect_faces(self):
    self._set_status("Detecting...")
    self.root.update()  # Force UI update
    # ... long operation ...
```

**Option 2: Threading (more complex)**
```python
import threading

def detect_faces(self):
    def do_detection():
        faces = self.detector.detect(self.img_bgr)
        self.root.after(0, lambda: self._detection_done(faces))
    
    thread = threading.Thread(target=do_detection)
    thread.start()

def _detection_done(self, faces):
    self.detected_faces = faces
    self._refresh_display()
```

## 8.6 Model Loading Errors

### Symptoms
- "YuNet/DNN unavailable" message
- Falls back to Haar unexpectedly
- Errors about ONNX or model file

### Root Causes

| Cause | Diagnosis | Fix |
|-------|-----------|-----|
| Model not downloaded | Check `data/` folder | Delete and re-run (auto-download) |
| OpenCV version old | `cv2.__version__` | Upgrade OpenCV |
| Corrupted download | Check file size | Delete and re-download |
| No internet | Offline machine | Manually place model file |

### Manual Model Installation

```bash
# Download manually
wget https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx

# Move to data folder
mkdir -p data
mv face_detection_yunet_2023mar.onnx data/
```

---

# Part 9: Testing, Ethics & Privacy

## 9.1 Manual Test Cases

### Core Functionality Tests

| Test | Input | Expected Output |
|------|-------|-----------------|
| Load valid image | JPG file | Image displays in canvas |
| Load invalid file | Text file | Error dialog shown |
| Detect single face | Portrait photo | 1 face detected, box shown |
| Detect multiple faces | Group photo | All faces detected |
| Detect no faces | Landscape photo | 0 faces, no boxes |
| Protect one face | Check one box | Green box, red others |
| Blur unprotected | Apply with mix | Checked=visible, unchecked=blurred |
| Save result | Export PNG | File created, blur persisted |

### Edge Case Tests

| Test | Input | Expected Behavior |
|------|-------|-------------------|
| Very small faces | Crowd photo | May miss small faces |
| Profile/side faces | Non-frontal photo | DNN better than Haar |
| Partial faces | Cropped at edge | Detect partial if enough visible |
| Blurry photo | Low quality image | Lower detection rate |
| Very large image | 8000×6000px | Should still work (slower) |
| No image loaded | Click Detect | Shows info dialog |
| Re-detect | New detection on same image | Clears previous boxes |

## 9.2 Bias and Fairness Considerations

Face detection systems can exhibit bias based on their training data. Important considerations:

### Known Biases in Face Detection

1. **Skin tone**: Some detectors perform worse on darker skin tones
2. **Age**: May struggle with very young or elderly faces
3. **Gender**: Potential performance differences
4. **Cultural features**: Trained primarily on Western faces

### Mitigation Strategies

1. **Multiple detection methods**: DNN and Haar may have different biases
2. **Adjustable thresholds**: Users can tune sensitivity
3. **Manual override**: Users review and correct before applying
4. **Transparency**: Users see all detections before anonymization

### Responsible Deployment

- Test across diverse populations
- Document known limitations
- Provide manual correction tools
- Default to over-protecting (anonymize on uncertainty)

## 9.3 Ethical Implications

### Legitimate Uses
- Protecting bystander privacy in shared photos
- Anonymizing research participants
- Protecting vulnerable individuals in documentation
- Compliance with privacy regulations (GDPR)

### Potential Misuse
- Enabling sharing of images without consent
- Hiding evidence of wrongdoing
- Circumventing identification for malicious purposes

### Design Choices That Minimize Harm

1. **Local processing**: Images never leave the user's device
2. **Manual review**: User must confirm before applying
3. **Non-automatic**: Doesn't work on videos in background
4. **Whitelist approach**: Protects by default

## 9.4 Privacy by Design Principles

AnonVision embodies several privacy-by-design principles:

| Principle | Implementation |
|-----------|----------------|
| Data minimization | No data collected, no telemetry |
| Local processing | Everything runs on user's machine |
| User control | User chooses what to anonymize |
| Transparency | Detection results shown before anonymizing |
| Security by default | Unknown faces are anonymized by default |

## 9.5 Limitations and Responsible Use

### Technical Limitations
- May miss faces in challenging conditions
- False positives on face-like objects
- Anonymization can potentially be reversed with AI
- Does not anonymize body, clothing, or context

### Responsible Use Guidelines

1. **Get consent when possible**: Anonymization is a fallback, not a substitute for consent
2. **Verify results**: Always review before publishing
3. **Consider context**: Sometimes the whole image reveals identity
4. **Stay updated**: Anonymization methods may become bypassable

---

# Part 10: Design Philosophy & Tradeoffs

## 10.1 What Was Prioritized

### Reliability Over Features
- Works on all platforms
- Graceful degradation
- No external dependencies at runtime

### Transparency Over Magic
- User sees all detections
- Manual review before applying
- Visible settings

### Privacy Over Convenience
- Local processing only
- Whitelist (protect by default)
- No cloud features

### Simplicity Over Power
- Single image workflow
- Basic anonymization methods
- Minimal configuration

## 10.2 What Was Intentionally Not Included

| Feature | Why Excluded |
|---------|--------------|
| Video processing | Complexity, different workflow |
| Cloud processing | Privacy concerns |
| Automatic mode | User review is essential |
| Real-time camera | Focus on post-capture |
| Body detection | Scope limitation |
| AI upscaling | Could de-anonymize |
| Batch processing | Focus on careful review |

## 10.3 What Would Change in a Cloud Version

If AnonVision were redesigned as a cloud service:

| Local Version | Cloud Version |
|---------------|---------------|
| CPU-only | GPU acceleration |
| Download models | Server-side models |
| Local storage | Cloud storage |
| Single user | Multi-user |
| No accounts | Authentication |
| No telemetry | Usage analytics |
| Unlimited use | Rate limits |

**Privacy tradeoffs**: Would require user trust, data policies, encryption, compliance.

## 10.4 What Would Change for Production SaaS

### Additional Requirements
- User authentication
- Payment processing
- Usage quotas
- Admin dashboard
- Monitoring and logging
- Error tracking
- Performance optimization
- Mobile apps
- API for integrations

### Architecture Changes
- Microservices
- Queue-based processing
- CDN for delivery
- Database for metadata
- Caching layer

## 10.5 What Makes AnonVision Explainable

### Design for Understanding

1. **Visible state**: All detections shown as boxes
2. **Clear actions**: Three steps (Prepare → Preview → Export)
3. **Immediate feedback**: Actions update display instantly
4. **No hidden processing**: What you see is what you get
5. **Configurable parameters**: Users can understand each setting

### Learnability Features
- Status bar explains what happened
- Color coding (green = safe, red = will change)
- Tooltips on settings
- Preview before applying

---

# Part 11: Appendices

## Appendix A: Folder Tree

```
anonvision/
├── start_the_app_.py       # Entry point, dependency checker
├── gui.py                  # Main application (913 lines)
│   └── AnonVisionGUI       # Main class
├── dnn_detector.py         # YuNet detector (~95 lines)
│   └── DNNFaceDetector     # DNN detection class
├── face_detector.py        # Haar detector (~115 lines)
│   └── FaceDetector        # Cascade detection class
├── face_whitelist.py       # Recognition (~145 lines)
│   └── FaceWhitelist       # Whitelist management class
├── utils.py                # Utilities (~65 lines)
│   ├── resize_image_for_display()
│   ├── validate_image_path()
│   ├── hex_to_bgr()
│   └── clamp_box()
├── data/                   # Auto-created
│   ├── face_detection_yunet_2023mar.onnx
│   └── haarcascade_frontalface_default.xml
└── whitelist_db/           # Auto-created
    ├── faces.json          # Whitelist metadata
    └── *_thumb.jpg         # Thumbnails
```

## Appendix B: Class Interaction Map

```
                          start_the_app_.py
                                 │
                                 │ imports
                                 ▼
                         ┌──────────────┐
                         │    gui.py    │
                         │              │
                         │ AnonVisionGUI│
                         └───────┬──────┘
                                 │
            ┌────────────────────┼────────────────────┐
            │                    │                    │
            ▼                    ▼                    ▼
   ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
   │ dnn_detector.py │  │ face_detector.py│  │face_whitelist.py│
   │                 │  │                 │  │                 │
   │ DNNFaceDetector │  │   FaceDetector  │  │  FaceWhitelist  │
   └────────┬────────┘  └────────┬────────┘  └────────┬────────┘
            │                    │                    │
            └────────────────────┼────────────────────┘
                                 │
                                 ▼
                          ┌──────────────┐
                          │   utils.py   │
                          │              │
                          │ Utilities    │
                          └──────────────┘

Legend:
────► imports/uses
┌───┐ module
│   │ primary class
└───┘
```

## Appendix C: Configuration Parameters Reference

### Detection Parameters

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `detection_method` | str | "dnn", "haar" | "dnn" | Which detector to use |
| `dnn_confidence` | float | 0.1-0.95 | 0.50 | Minimum detection confidence |
| `scale_factor` | float | 1.01-2.0 | 1.10 | Haar pyramid scaling |
| `min_neighbors` | int | 1-10 | 5 | Haar minimum detection count |
| `min_size` | int | 10-200 | 30 | Minimum face size (pixels) |

### Anonymization Parameters

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `anon_method` | str | "blur", "pixelate", "blackout" | "blur" | Anonymization type |
| `blur_strength` | int | 41-99 (odd) | 71 | Gaussian kernel size |
| `blur_passes` | int | 3-10 | 6 | Number of blur iterations |
| `pixel_size` | int | 8-50 | 20 | Pixelation block size |
| `edge_feather` | int | 0-50 | 25 | Edge softness (percent) |
| `region_expansion` | int | 10-60 | 35 | Expand region (percent) |

### UI Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `show_boxes` | bool | True | Draw detection rectangles |

## Appendix D: Glossary

### Computer Vision Terms

| Term | Definition |
|------|------------|
| **Bounding box** | Rectangle defined by (x, y, width, height) that encloses a detected object |
| **Cascade classifier** | Machine learning model that uses a cascade of simple classifiers |
| **Confidence** | Probability score (0-1) indicating detection certainty |
| **DNN** | Deep Neural Network; multi-layer machine learning model |
| **False positive** | Detection where no face actually exists |
| **False negative** | Missed detection where a face does exist |
| **Haar features** | Simple rectangular contrast patterns used in cascade classifiers |
| **NMS** | Non-Maximum Suppression; removes overlapping detections |
| **ONNX** | Open Neural Network Exchange; model format |
| **ROI** | Region of Interest; image subregion being processed |
| **YuNet** | A specific lightweight face detection neural network |

### Python/Programming Terms

| Term | Definition |
|------|------------|
| **BGR** | Blue-Green-Red color order (OpenCV's default) |
| **Callback** | Function passed to another function to be called later |
| **Instance variable** | Variable belonging to a specific object (`self.x`) |
| **Lambda** | Anonymous inline function |
| **NumPy array** | Efficient multi-dimensional array for numerical computing |
| **PIL** | Python Imaging Library (Pillow) |
| **Tkinter** | Python's built-in GUI toolkit |
| **Widget** | GUI element (button, label, canvas, etc.) |

### UX Terms

| Term | Definition |
|------|------------|
| **Affordance** | Visual cue suggesting how to interact |
| **Modal dialog** | Popup that blocks interaction with main window |
| **Progressive disclosure** | Revealing complexity gradually |
| **Status bar** | Area displaying current state/feedback |
| **Thumbnail** | Small preview image |

## Appendix E: Extension Ideas

### Safe Extensions (Low Risk)

1. **Add more blur methods**: Motion blur, box blur, median blur
2. **Export format options**: WebP, TIFF with transparency
3. **Zoom and pan**: Navigate large images
4. **Keyboard shortcuts**: Ctrl+O open, Ctrl+S save
5. **Recent files list**: Quick access to previous images
6. **Undo/redo**: Track state history

### Moderate Extensions (Medium Risk)

1. **Batch processing**: Process multiple images with same settings
2. **Whitelist integration**: Connect existing `face_whitelist.py` to GUI
3. **Detection statistics**: Show confidence scores, performance metrics
4. **Custom color themes**: User-selectable UI colors
5. **Export presets**: Save/load anonymization configurations

### Advanced Extensions (Higher Risk)

1. **Video support**: Frame-by-frame processing, tracking
2. **Body detection**: Extend beyond faces
3. **GPU acceleration**: Optional CUDA support
4. **Mobile app**: Separate iOS/Android implementation
5. **Plugin system**: Allow third-party anonymization methods

---

# Conclusion

AnonVision represents a thoughtful approach to privacy-preserving image processing. Its design prioritizes:

- **Reliability** through graceful degradation
- **Privacy** through local processing
- **Transparency** through visible state
- **Usability** through progressive workflow

Understanding this system provides a foundation for building computer vision applications that respect both technical constraints and human values.

The modular architecture allows for extension without disruption. The defensive programming protects against failures. The user-centered design builds trust through transparency.

Whether you're learning computer vision, building privacy tools, or studying desktop application development, AnonVision provides concrete examples of theory in practice.

---

*Document generated for AnonVision Training Program*
*Based on analysis of 6 source files comprising approximately 1,500 lines of Python code*
