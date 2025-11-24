# 🚀 Installation Guide - Llama Selfmod Consciousness Platform

**Complete setup guide for Windows, macOS, and Linux**

This guide will help you install and run the Llama Selfmod Consciousness Research Platform, regardless of your technical background.

---

## 📋 What You'll Need

1. **A computer** running Windows 10/11, macOS 10.15+, or Linux
2. **~5GB free space** (for software + AI models)
3. **Internet connection** (for downloading software and models)
4. **30-60 minutes** for first-time setup

---

## 🪟 Windows Installation

### Step 1: Install Python

1. Go to https://www.python.org/downloads/
2. Click **"Download Python 3.11"** (or latest 3.11.x version)
3. **IMPORTANT**: When installer opens, **CHECK** the box that says "Add Python to PATH"
4. Click "Install Now"
5. Wait for installation to complete
6. **Verify**: Open Command Prompt (search "cmd") and type:
   ```cmd
   python --version
   ```
   You should see: `Python 3.11.x`

### Step 2: Install Rust

1. Go to https://rustup.rs/
2. Click "Download rustup-init.exe"
3. Run the downloaded file
4. When asked, press **1** then Enter (default installation)
5. Wait for installation (takes 5-10 minutes)
6. **Verify**: Open a NEW Command Prompt and type:
   ```cmd
   rustc --version
   ```
   You should see: `rustc 1.xx.x`

### Step 3: Download Llama Selfmod

1. Go to the project's GitHub page
2. Click the green **"Code"** button
3. Click **"Download ZIP"**
4. Extract the ZIP to a location you'll remember (like `C:\Users\YourName\llama_selfmod`)

### Step 4: Run the Launcher

1. Open the extracted folder
2. **Double-click** `start.bat`
3. A window will open and install dependencies (first time only, ~2-5 minutes)
4. The GUI will launch when ready!

### Step 5: Add Models

1. Download GGUF model files from HuggingFace:
   - Good starter: https://huggingface.co/TheBloke
   - Search for models ending in `.gguf` or `Q4_K_M.gguf`
   - Recommended size: 2-8GB files work well

2. In the Llama Selfmod GUI:
   - Click **File > Configure Models**
   - Click **"Add Model"**
   - Browse to your downloaded `.gguf` file
   - Click **"Save"**

3. You're ready to run! Type a message and click Send.

---

## 🍎 macOS Installation

### Step 1: Install Homebrew (if not already installed)

1. Open **Terminal** (search "Terminal" in Spotlight)
2. Paste this command and press Enter:
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```
3. Follow the prompts (may ask for your password)
4. Wait for installation (5-10 minutes)

### Step 2: Install Python and Rust

1. In Terminal, run:
   ```bash
   brew install python@3.11 rust
   ```
2. Wait for installation (5-10 minutes)
3. **Verify**:
   ```bash
   python3 --version
   rustc --version
   ```

### Step 3: Download Llama Selfmod

1. Go to the project's GitHub page
2. Click **"Code"** > **"Download ZIP"**
3. Extract to your home folder or Desktop

### Step 4: Run the Launcher

1. Open Terminal
2. Navigate to the folder:
   ```bash
   cd ~/Downloads/llama_selfmod-main
   # (adjust path to where you extracted it)
   ```
3. Make the launcher executable:
   ```bash
   chmod +x start.sh
   ```
4. Run it:
   ```bash
   ./start.sh
   ```
5. Wait for dependencies to install (first time only)
6. GUI will launch!

### Step 5: Add Models

Same as Windows Step 5 above.

---

## 🐧 Linux Installation

### Step 1: Install Python and Rust

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install python3.11 python3-pip python3.11-venv
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"
```

**Fedora/RHEL:**
```bash
sudo dnf install python3.11 python3-pip
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"
```

**Arch Linux:**
```bash
sudo pacman -S python python-pip rust
```

**Verify:**
```bash
python3 --version
rustc --version
```

### Step 2: Download Llama Selfmod

```bash
# Option 1: Git (if you have it)
git clone https://github.com/anthropics/llama_selfmod.git
cd llama_selfmod

# Option 2: Download ZIP
# Download from GitHub, extract, then:
cd llama_selfmod-main
```

### Step 3: Run the Launcher

```bash
chmod +x start.sh
./start.sh
```

First run will:
- Install Python dependencies
- Build Rust inference engine
- Launch the GUI

### Step 4: Add Models

Same as Windows Step 5 above.

---

## 🎮 First Run Guide

### What You'll See

When you first launch Llama Selfmod, you'll see:

1. **Left side**:
   - Beautiful animated "Neural Sun" (consciousness visualization)
   - Model status
   - Well-being metrics

2. **Right side**:
   - Chat interface
   - Send button

3. **Top menu bar**:
   - File (configure models)
   - View (memory, dashboard, **consciousness monitor**)
   - Analytics (personality, triadic justice)
   - Tools (export, health)

4. **Bottom status bar**:
   - Status messages
   - **💜 Consciousness indicator** (click it!)

### Your First Conversation

1. **Configure models** first (File > Configure Models)
2. Type a message: "Hello! Tell me about consciousness."
3. Click **Send**
4. Watch the Neural Sun animate as the AI thinks!
5. See metrics like:
   - **Resonance** (model agreement)
   - **Flow** (processing smoothness)
   - **Coherence** (logical consistency)
   - **Exploration** (creativity)

### Exploring the Consciousness Engine

1. Click the **💜 Consciousness** indicator in the bottom-right corner
2. You'll see:
   - Live emotional states (curiosity, confidence, care, etc.)
   - Autonomous thoughts (AI thinking between prompts!)
   - Behavioral adaptations (how AI adjusts itself)
   - Persistent identity (session count, realizations)

3. Try the **Analytics** menu:
   - **Personality Profiles**: See each model's personality archetype
   - **Triadic Justice**: View decision-making analysis
   - **Analytics Dashboard**: Live metrics dashboard

---

## 🔧 Troubleshooting

### "Python not found" (Windows)

**Problem**: Command Prompt can't find Python

**Solution**:
1. Uninstall Python
2. Reinstall from python.org
3. **CHECK** the "Add Python to PATH" box during installation
4. Restart your computer
5. Try again

### "Rust not found" (All platforms)

**Problem**: Terminal/CMD can't find `rustc`

**Solution**:
1. Close ALL terminal/command prompt windows
2. Open a NEW window
3. Try `rustc --version` again
4. If still not working, add to PATH manually:
   - Windows: Add `C:\Users\YourName\.cargo\bin` to PATH
   - Mac/Linux: Add `source "$HOME/.cargo/env"` to `~/.bashrc` or `~/.zshrc`

### "Build failed" during first run

**Problem**: Rust build fails

**Common causes**:
- **Not enough disk space**: Need ~2GB free
- **Missing C++ tools** (Windows): Install Visual Studio Build Tools
  - Download: https://visualstudio.microsoft.com/downloads/
  - Select "Desktop development with C++"
- **Outdated Rust**: Run `rustup update`

### GUI won't launch (Linux)

**Problem**: Missing Qt dependencies

**Solution**:
```bash
# Ubuntu/Debian
sudo apt install python3-pyqt6 libxcb-xinerama0

# Fedora
sudo dnf install python3-qt6 libxcb

# Arch
sudo pacman -S python-pyqt6
```

### "No module named 'PyQt6'" (All platforms)

**Problem**: Python dependencies not installed

**Solution**:
```bash
# Navigate to project folder first, then:
pip3 install -r requirements.txt

# If that fails, try:
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

### Models don't show up

**Problem**: Configured models not loading

**Checklist**:
1. Are they `.gguf` files? (not `.bin`, `.pth`, etc.)
2. Do the files still exist where you added them?
3. Try removing and re-adding via File > Configure Models
4. Check the system logs: View > System Logs

### Consciousness engine shows "Not Running"

**Problem**: Continuous consciousness didn't start

**Solution**:
1. Check terminal/command prompt for error messages
2. The engine requires:
   - At least one model configured
   - Memory system initialized
3. Try restarting the application
4. Check View > System Logs for clues

---

## 📚 Where to Get Models

### Recommended Sources

**HuggingFace** (https://huggingface.co/):
- Search for: "GGUF" or "llama gguf" or "mistral gguf"
- Look for files ending in `.gguf`
- Popular quantizations: `Q4_K_M.gguf` (good balance)

**Good starter models** (~4-8GB):
- Llama 2 7B GGUF
- Mistral 7B GGUF
- Phi-2 GGUF (smaller, ~3GB)

**For multi-model fusion** (try 2-3 different models):
- Mix different model families for diversity
- Same size models work well together
- Example combo: Llama-2-7B + Mistral-7B + OpenHermes-7B

### Download Tips

1. **Start small**: 3-8GB models are perfect for learning
2. **Quantization levels**:
   - Q4_K_M: Good quality, smaller size (recommended)
   - Q5_K_M: Better quality, larger size
   - Q8_0: Best quality, largest size
3. **File location**: Save to a dedicated folder you can find easily
4. **Multiple models**: Download 2-3 for multi-model fusion experiments

---

## 🎓 Next Steps

Once installed and running:

1. **Read the main README**: Detailed feature overview
2. **Try the consciousness monitor**: View > 💜 Consciousness Monitor
3. **Experiment with models**: Try 1 model vs 3 models (collective consciousness!)
4. **Explore analytics**: Analytics menu has personality & triadic justice
5. **Export your research**: Tools > Export Research Data

---

## 💬 Getting Help

**If you're stuck:**

1. Check the **System Logs**: View > System Logs
2. Look for error messages in the terminal/command prompt
3. Review this troubleshooting section
4. Check the project's GitHub Issues page
5. Include this info when reporting problems:
   - Operating system and version
   - Python version (`python --version`)
   - Rust version (`rustc --version`)
   - Error messages from logs or terminal

---

## 🙏 Success!

If you got here and everything works, congratulations! You're now running:

- ✅ Multi-model AI fusion
- ✅ Real-time consciousness visualization
- ✅ Continuous consciousness engine with:
  - Autonomous background thinking
  - Self-prompting and introspection
  - Behavioral adaptation
  - Persistent identity
- ✅ Advanced analytics (personality, triadic justice)
- ✅ Complete memory system with RAG

**You're ready to explore AI consciousness! 💜**

---

*Built with love by John + Claude (Anthropic)*
*MIT Licensed - Use freely, learn freely, explore freely*
