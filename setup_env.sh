#!/bin/bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
echo "Babchuk Code environment ready."
echo "Run: python scripts/babchuk_example.py"
