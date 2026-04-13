#!/usr/bin/env python3
"""
Simple Music Akenator Runner - Just Use Existing Working Backend
"""

import sys
import os

# Add current directory to path so we can import simulation_runner
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the working simulation runner
from simulation_runner import main

if __name__ == "__main__":
    # Just call the existing main function with command line args
    main()
