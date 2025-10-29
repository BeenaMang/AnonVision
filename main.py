#!/usr/bin/env python3
"""
AnonVision Launcher (YuNet-only)
- Forces CPU (no CUDA/OpenCL)
- Avoids importing cv2 during requirement checks
- Silences OpenCV loader warnings
- Launches gui.main()
"""
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

CORE_DEPS = {
    "cv2": "opencv-contrib-python",
    "PIL": "pillow",
    "numpy": "numpy",
}

def _module_exists(name: str) -> bool:
    # Use find_spec to avoid import side-effects (especially cv2)
    return importlib.util.find_spec(name) is not None

def check_requirements():
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

def maybe_install(packages):
    for pkg in packages:
        try:
            print(f"  • Installing {pkg} ...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
        except Exception as e:
            print(f"    ⚠ Failed to install {pkg}: {e}")

def main():
    print("=" * 60)
    print("AnonVision Integrated - Face Privacy Protection System (YuNet-only)")
    print("=" * 60)

    print("\n📋 Checking requirements...")
    missing_core = check_requirements()

    if missing_core:
        print("\n❗ Missing REQUIRED packages:")
        print("   " + " ".join(missing_core))
        print("\nRun:")
        print("   pip install " + " ".join(missing_core))
        resp = input("\nInstall required packages now? (y/N): ").strip().lower()
        if resp == "y":
            print("\n🔧 Installing required packages...")
            maybe_install(missing_core)
            # Re-check
            mc = check_requirements()
            if mc:
                print("\n❌ Some required packages are still missing. Exiting.")
                sys.exit(1)
        else:
            print("\n⚠ Please install the required packages and re-run.")
            sys.exit(1)
    else:
        print("✅ Core requirements satisfied.")

    print("\n🚀 Launching AnonVision...")
    print("-" * 60)
    try:
        from gui import main as run_app
        run_app()
    except ImportError as e:
        print("\n❌ Error: Could not import gui.py")
        print(f"Details: {e}")
        print("\nMake sure gui.py is in the same directory as this launcher.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error launching application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
