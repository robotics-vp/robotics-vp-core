#!/usr/bin/env python3
"""
Print recommended Python packages/versions for OpenVLA + Meta vision stack.
"""
REQUIREMENTS = [
    "torch>=2.2 (cpu or appropriate CUDA wheel, e.g. cu121)",
    "transformers",
    "timm",
    "tokenizers",
    "pillow",
    "accelerate",
    "flash-attn (optional, --no-build-isolation)",
]


def main():
    print("Recommended packages for OpenVLA/vision:")
    for req in REQUIREMENTS:
        print(" -", req)


if __name__ == "__main__":
    main()
