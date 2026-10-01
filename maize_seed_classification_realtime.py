#!/usr/bin/env python3
"""Compatibility entry point for the single-seed sorter in main.py.

Use the model and metadata exported by the updated notebook. The old binary
classifier preprocessing and watershed detection are no longer used.
"""
from main import main

if __name__ == '__main__':
    raise SystemExit(main())
