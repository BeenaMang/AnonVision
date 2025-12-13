#AnonVision Launcher 

import os

# ---- HARD DISABLE ACCELERATORS & SILENCE LOGS ----
os.environ["CUDA_VISIBLE_DEVICES"] = ""          # hide CUDA
os.environ["OPENCV_DNN_DISABLE_OPENCL"] = "1"    # disable OpenCL for OpenCV DNN
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"         # silence OpenCV loader chatter
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_PLATFORM_NAME"] = "cpu"
# --------------------------------------------------

import sys
import subprocess
import importlib.util

# Core dependencies for YuNet + Haar Cascade + Whitelist
CORE_DEPS = {
    "cv2": "opencv-contrib-python",      # YuNet + Haar Cascade + image processing
    "PIL": "pillow",                     # Image handling for Tkinter
    "numpy": "numpy",                    # Array operations
}

# Optional but highly recommended for whitelist face matching
OPTIONAL_DEPS = {
    "face_recognition": "face_recognition",  # High-accuracy face matching
}

def _module_exists(name: str) -> bool:
    """Check if module exists without importing it"""
    return importlib.util.find_spec(name) is not None

def check_requirements():
    """Check for required packages"""
    missing = []
    for mod, pkg in CORE_DEPS.items():
        if not _module_exists(mod):
            missing.append(pkg or mod)
    
    # tkinter check (safe to import)
    try:
        import tkinter  # noqa: F401
    except Exception:
        missing.append('python3-tk')
    
    return missing

def check_optional():
    """Check for optional packages"""
    missing_optional = []
    for mod, pkg in OPTIONAL_DEPS.items():
        if not _module_exists(mod):
            missing_optional.append(pkg or mod)
    return missing_optional

def maybe_install(packages):
    """Install packages via pip"""
    for pkg in packages:
        try:
            print(f"  * Installing {pkg} ...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
        except Exception as e:
            print(f"    WARNING: Failed to install {pkg}: {e}")

def main():
    print("=" * 70)
    print("AnonVision - Face Privacy Protection System")
    print("Detection: YuNet (Primary) + Haar Cascade (Fallback)")
    print("=" * 70)
    
    print("\nChecking requirements...")
    missing_core = check_requirements()
    
    if missing_core:
        print("\nERROR: Missing REQUIRED packages:")
        for pkg in missing_core:
            print(f"   X {pkg}")
        print("\nTo install, run:")
        print(f"   pip install {' '.join(missing_core)}")
        
        resp = input("\nInstall required packages now? (y/N): ").strip().lower()
        if resp == "y":
            print("\nInstalling required packages...")
            maybe_install(missing_core)
            
            # Re-check
            mc = check_requirements()
            if mc:
                print("\nERROR: Some required packages are still missing:")
                for pkg in mc:
                    print(f"   X {pkg}")
                print("\nPlease install manually and re-run.")
                sys.exit(1)
            else:
                print("SUCCESS: Required packages installed successfully!")
        else:
            print("\nWARNING: Cannot continue without required packages. Exiting.")
            sys.exit(1)
    else:
        print("OK: Core requirements satisfied.")
    
    # Check optional dependencies
    print("\nChecking optional packages...")
    missing_optional = check_optional()
    
    if missing_optional:
        print("\nWARNING: Optional packages not found (whitelist accuracy will be lower):")
        for pkg in missing_optional:
            print(f"   * {pkg} - Recommended for high-accuracy face matching")
        print("\nTo install, run:")
        print(f"   pip install {' '.join(missing_optional)}")
        print("\n   Without face_recognition: Whitelist will use OpenCV features (lower accuracy)")
        print("   With face_recognition: Whitelist will use deep learning (high accuracy)")
        
        resp = input("\nInstall optional packages now? (y/N): ").strip().lower()
        if resp == "y":
            print("\nInstalling optional packages...")
            maybe_install(missing_optional)
            # Don't exit if these fail - they're optional
            mo = check_optional()
            if not mo:
                print("SUCCESS: Optional packages installed successfully!")
            else:
                print("WARNING: Some optional packages failed to install. Continuing anyway...")
    else:
        print("OK: Optional packages satisfied (face_recognition available).")
    
    # Show detection configuration
    print("\n" + "=" * 70)
    print("Detection Configuration:")
    print("   Primary Detector:  YuNet (DNN-based, high accuracy)")
    print("   Fallback Detector: Haar Cascade (fast, works on all images)")
    print("   Whitelist Matching:", end=" ")
    if not missing_optional:
        print("face_recognition (high accuracy)")
    else:
        print("OpenCV features (basic accuracy)")
    print("   Hardware Mode:     CPU-only (CUDA/OpenCL disabled)")
    print("=" * 70)
    
    print("\nLaunching AnonVision GUI...")
    print("-" * 70)
    
    try:
        from gui import main as run_app
        run_app()
    except ImportError as e:
        print("\nERROR: Could not import gui.py")
        print(f"Details: {e}")
        print("\nMake sure gui.py is in the same directory as this launcher.")
        print("Required files:")
        print("   * gui.py")
        print("   * face_whitelist.py")
        print("   * whitelist_tkinter_integration.py")
        print("   * dnn_detector.py")
        print("   * face_detector.py (Haar Cascade)")
        print("   * utils.py")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: Error launching application: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()