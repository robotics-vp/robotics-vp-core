"""
Smoke test for third-party dependencies.

Run with: python -m third_party.smoke
"""
from __future__ import annotations

import sys
from typing import Dict, List, Tuple


def check_import(module_name: str) -> Tuple[bool, str]:
    """Check if a module can be imported."""
    try:
        __import__(module_name)
        return True, "OK"
    except ImportError as e:
        return False, str(e)


def check_torch() -> Tuple[bool, str, Dict[str, str]]:
    """Check PyTorch installation and CUDA availability."""
    try:
        import torch
        info = {
            "version": torch.__version__,
            "cuda": str(torch.cuda.is_available()),
            "cuda_version": torch.version.cuda or "N/A",
        }
        return True, "OK", info
    except ImportError as e:
        return False, str(e), {}


def check_wrapper(wrapper_name: str, class_name: str) -> Tuple[bool, str]:
    """Check if a wrapper can be instantiated."""
    try:
        module = __import__(f"third_party.{wrapper_name}", fromlist=[class_name])
        cls = getattr(module, class_name)
        # Try to instantiate with fallback mode
        instance = cls(use_fallback=True)
        return True, f"{class_name} instantiated"
    except ImportError as e:
        return False, f"Import error: {e}"
    except AttributeError as e:
        return False, f"Class not found: {e}"
    except Exception as e:
        return False, f"Error: {e}"


def run_smoke_tests() -> bool:
    """Run all smoke tests."""
    print("=" * 60)
    print("Third-Party Dependencies Smoke Test")
    print("=" * 60)
    
    all_passed = True
    
    # 1. Check PyTorch
    print("\n[1] PyTorch Installation")
    ok, msg, info = check_torch()
    if ok:
        print(f"  ✓ PyTorch {info['version']}")
        print(f"    CUDA available: {info['cuda']}")
        if info['cuda'] == "True":
            print(f"    CUDA version: {info['cuda_version']}")
    else:
        print(f"  ✗ PyTorch: {msg}")
        all_passed = False
    
    # 2. Check core dependencies
    print("\n[2] Core Dependencies")
    deps = ["numpy", "scipy", "PIL", "cv2"]
    for dep in deps:
        if dep == "cv2":
            try:
                import cv2
                print(f"  ✓ {dep} (OpenCV {cv2.__version__})")
            except ImportError:
                print(f"  ✗ {dep} not installed")
        elif dep == "PIL":
            try:
                from PIL import Image
                import PIL
                print(f"  ✓ {dep} (Pillow {PIL.__version__})")
            except ImportError:
                print(f"  ✗ {dep} not installed")
        else:
            ok, msg = check_import(dep)
            print(f"  {'✓' if ok else '✗'} {dep}: {msg}")
            if not ok:
                all_passed = False
    
    # 3. Check wrappers
    print("\n[3] Wrapper Modules")
    wrappers = [
        ("sam3d_objects_wrapper", "SAM3DObjectsInference"),
        ("sam3d_body_wrapper", "SAM3DBodyInference"),
        ("lpips_wrapper", "LPIPSLoss"),
    ]
    for wrapper, cls in wrappers:
        ok, msg = check_wrapper(wrapper, cls)
        print(f"  {'✓' if ok else '○'} {wrapper}: {msg}")
        # Don't fail on missing wrappers - they're optional
    
    # 4. Check upstream repos (optional)
    print("\n[4] Upstream Repositories (optional)")
    upstream = [
        ("third_party.sam3d_objects", "SAM3D-Objects"),
        ("third_party.sam3d_body", "SAM3D-Body"),
        ("third_party.inrtracker", "INRTracker"),
    ]
    for module, name in upstream:
        ok, msg = check_import(module)
        print(f"  {'✓' if ok else '○'} {name}: {'installed' if ok else 'not installed (optional)'}")
    
    # Summary
    print("\n" + "=" * 60)
    if all_passed:
        print("All required dependencies OK!")
        print("Note: Upstream repos and weights are optional for stub mode.")
    else:
        print("Some required dependencies missing!")
        print("Install with: pip install -r requirements.txt")
    print("=" * 60)
    
    return all_passed


if __name__ == "__main__":
    success = run_smoke_tests()
    sys.exit(0 if success else 1)
