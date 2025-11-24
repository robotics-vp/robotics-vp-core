#!/usr/bin/env python3
"""
Export demo frames to MP4 video.
Requires ffmpeg installed.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

def export_video(args):
    input_dir = Path(args.input_dir)
    output_file = Path(args.output_file)
    
    if not input_dir.exists():
        print(f"Input directory {input_dir} does not exist.")
        sys.exit(1)
        
    print(f"[ExportVideo] Stitching frames from {input_dir} to {output_file}...")
    
    # FFmpeg command
    # Assumes frames are named frame_0000.png, etc. or similar pattern
    # We'll use a glob pattern if possible, or just %04d.png
    
    pattern = str(input_dir / f"{args.prefix}%d.png")
    
    cmd = [
        "ffmpeg",
        "-y", # Overwrite
        "-framerate", str(args.fps),
        "-i", pattern,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        str(output_file)
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"[ExportVideo] Video saved to {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"[ExportVideo] FFmpeg failed: {e}")
        print("Ensure ffmpeg is installed and frames exist with correct naming pattern.")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, required=True, help="Directory containing frames")
    parser.add_argument("--output-file", type=str, default="demo.mp4", help="Output MP4 file")
    parser.add_argument("--fps", type=int, default=30, help="Frame rate")
    parser.add_argument("--prefix", type=str, default="frame_", help="Frame filename prefix")
    args = parser.parse_args()
    export_video(args)
