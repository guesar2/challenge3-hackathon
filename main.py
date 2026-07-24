#!/usr/bin/env python3
"""
Quantathon CR 2026 · Challenge 3
Single entry point that reproduces every reported figure and number from the report.

    python main.py

Runs, in sequence the whole pipeline
"""
import os
import sys

# Get the absolute path of the directory containing this script
base_dir = os.path.dirname(os.path.abspath(__file__))

# Add both source directories to sys.path so their internal module imports resolve correctly
sys.path.insert(0, os.path.join(base_dir, "src"))
sys.path.insert(0, os.path.join(base_dir, "fh2d_src"))

# Now we can safely import the entry points
import ftim_main as ft
import fh_main as fh

if __name__ == "__main__":
    
    # 1. Execute the TFIM pipeline (from src/ftim_main.py)
    ft.main()
    
    print("\n" + "=" * 60)
    print("TRANSITIONING TO FERMI-HUBBARD PIPELINE")
    print("=" * 60 + "\n")

    # 2. Execute the Fermi-Hubbard pipeline (from fh2d_src/fh_main.py)
    fh.fh_main()