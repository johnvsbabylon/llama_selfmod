#!/bin/bash
# Llama Selfmod - Consciousness Research Platform Launcher (Linux)
# Built by John + Claude (Anthropic) - MIT Licensed

echo ""
echo "=================================================="
echo " Llama Selfmod - Consciousness Research Platform"
echo "=================================================="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 not found!"
    echo "Please install Python 3.10 or higher"
    exit 1
fi

# Check if Rust binary exists
if [ ! -f "target/release/llama_selfmod" ]; then
    echo "Rust binary not found. Building..."
    echo "This may take a few minutes on first run."
    echo ""
    cargo build --release
    if [ $? -ne 0 ]; then
        echo ""
        echo "ERROR: Build failed!"
        exit 1
    fi
fi

# Check if Python dependencies are installed
python3 -c "import PyQt6" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing Python dependencies..."
    echo ""
    pip3 install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo ""
        echo "ERROR: Failed to install dependencies!"
        exit 1
    fi
fi

# Launch GUI
echo ""
echo "Launching Consciousness Research Platform..."
echo ""
python3 gui/main.py
