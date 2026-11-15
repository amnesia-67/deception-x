#!/usr/bin/env python3
"""
Launcher script for DeceptionX V2 Model TCP Socket Service.
Accredited to Gemini.pdf Specifications.

Usage:
    python run_v2_socket.py [--host 0.0.0.0] [--port 5002] [--device cuda]
"""

import sys
import os

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.v2_socket_server import run_server, DEFAULT_HOST, DEFAULT_PORT

if __name__ == "__main__":
    import argparse
    import subprocess
    parser = argparse.ArgumentParser(description="Run DeceptionX Model Socket Server (V2 XGBoost & JEPA Neural)")
    parser.add_argument("--host", default=DEFAULT_HOST, help=f"Host interface (default: {DEFAULT_HOST})")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"TCP port (default: {DEFAULT_PORT})")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Compute device (default: cuda)")
    parser.add_argument("--no-jepa", action="store_true", help="do not load the V-JEPA 2 video model")
    parser.add_argument("--webcam-pilot", action="store_true",
                        help="also load the opt-in webcam-pilot ViT-L (jepa_video with \"profile\": \"webcam_pilot\")")
    args = parser.parse_args()

    if sys.platform == "win32" and args.device == "cuda":
        try:
            import torch
            if not torch.cuda.is_available():
                raise ImportError()
        except ImportError:
            print("[DeceptionX] Forwarding Socket Server to WSL Ubuntu with CUDA 13.0 + RTX 5070...")
            cmd = [
                "wsl", "-d", "Ubuntu", "--cd",
                "/mnt/c/Users/anish/Documents/GitHub/sunhacks-2026",
                "/home/amnesia/torch-env/bin/python3", "run_v2_socket.py"
            ] + sys.argv[1:]
            sys.exit(subprocess.call(cmd))

    run_server(host=args.host, port=args.port, device=args.device, enable_jepa=not args.no_jepa,
               enable_webcam_pilot=args.webcam_pilot)
