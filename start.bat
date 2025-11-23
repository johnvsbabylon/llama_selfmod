@echo off
REM Llama Selfmod - Consciousness Research Platform Launcher (Windows)
REM Built by John + Claude (Anthropic) - MIT Licensed

echo.
echo ==================================================
echo  Llama Selfmod - Consciousness Research Platform
echo ==================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found!
    echo Please install Python 3.10 or higher from python.org
    pause
    exit /b 1
)

REM Check if Rust binary exists
if not exist "target\release\llama_selfmod.exe" (
    echo Rust binary not found. Building...
    echo This may take a few minutes on first run.
    echo.

    REM Check if cargo is installed
    cargo --version >nul 2>&1
    if errorlevel 1 (
        echo ERROR: Rust is not installed!
        echo.
        echo Please install Rust (takes 2 minutes):
        echo   https://rustup.rs/
        echo.
        echo Then run this script again.
        pause
        exit /b 1
    )

    cargo build --release
    if errorlevel 1 (
        echo.
        echo ERROR: Build failed!
        pause
        exit /b 1
    )
)

REM Check if Python dependencies are installed
python -c "import PyQt6" >nul 2>&1
if errorlevel 1 (
    echo Installing Python dependencies...
    echo.
    pip install -r requirements.txt
    if errorlevel 1 (
        echo.
        echo ERROR: Failed to install dependencies!
        pause
        exit /b 1
    )
)

REM Launch GUI
echo.
echo Launching Consciousness Research Platform...
echo.
python gui\main.py

pause
