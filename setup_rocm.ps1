# ROCm Environment Setup for MoneyMaker Turbo
# Installs ROCm PyTorch wheels inside Python 3.12 Virtual Environment
# Solves Windows path length limits by building in a shorter path and linking it.
# Bypasses strict chatterbox-tts PyTorch downgrade by using manual dependency mapping and --no-deps installation.

Write-Host "===================================================" -ForegroundColor Cyan
Write-Host "  Setting up ROCm environment on Windows (Python 3.12)" -ForegroundColor Cyan
Write-Host "===================================================" -ForegroundColor Cyan

$RootDir = Get-Location
$BackendDir = Join-Path $RootDir "money-printer-interface\backend"
$VenvDir = Join-Path $BackendDir ".venv"
$ShortVenvDir = "C:\Users\junaidi\vrocm"

# 1. Check Python 3.12
Write-Host "`n[1/5] Checking for Python 3.12..." -ForegroundColor Yellow
$PythonCheck = & py --list
if ($PythonCheck -match "3.12") {
    Write-Host "Python 3.12 found!" -ForegroundColor Green
} else {
    Write-Error "Python 3.12 is required but was not found. Please install Python 3.12."
    Exit 1
}

# 2. Re-create virtual environment in the short path
Write-Host "`n[2/5] Creating Python 3.12 virtual environment at $ShortVenvDir..." -ForegroundColor Yellow
if (Test-Path $ShortVenvDir) {
    Write-Host "Existing short path venv found. Removing old virtual environment..." -ForegroundColor Gray
    Remove-Item -Recurse -Force $ShortVenvDir
}
if (Test-Path $VenvDir) {
    Write-Host "Existing backend .venv junction/folder found. Removing..." -ForegroundColor Gray
    # Try removing junction/folder
    if ((Get-Item $VenvDir).LinkType -eq "Junction") {
        # It's a junction, just delete the link
        [System.IO.Directory]::Delete($VenvDir)
    } else {
        Remove-Item -Recurse -Force $VenvDir
    }
}

Start-Process py -ArgumentList "-3.12 -m venv $ShortVenvDir" -Wait -NoNewWindow
if (-not (Test-Path $ShortVenvDir)) {
    Write-Error "Failed to create Python virtual environment at $ShortVenvDir."
    Exit 1
}
Write-Host "Virtual environment created successfully at $ShortVenvDir!" -ForegroundColor Green

# Create directory junction from backend\.venv to the short path venv
Write-Host "Creating directory junction from $VenvDir to $ShortVenvDir..." -ForegroundColor Gray
cmd /c mklink /j "$VenvDir" "$ShortVenvDir"
if (-not (Test-Path $VenvDir)) {
    Write-Error "Failed to create directory junction."
    Exit 1
}
Write-Host "Directory junction created successfully!" -ForegroundColor Green

# 3. Get executable paths
$PythonExe = Join-Path $ShortVenvDir "Scripts\python.exe"
$PipExe = Join-Path $ShortVenvDir "Scripts\pip.exe"

# Upgrade pip
Write-Host "Upgrading pip..." -ForegroundColor Gray
Start-Process $PythonExe -ArgumentList "-m pip install --upgrade pip" -Wait -NoNewWindow

# 4. Install ROCm Wheels
Write-Host "`n[3/5] Installing AMD ROCm SDK components..." -ForegroundColor Yellow

$RocmUrls = @(
    "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm_sdk_core-7.2.1-py3-none-win_amd64.whl",
    "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm_sdk_devel-7.2.1-py3-none-win_amd64.whl",
    "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm_sdk_libraries_custom-7.2.1-py3-none-win_amd64.whl",
    "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm-7.2.1.tar.gz"
)

foreach ($url in $RocmUrls) {
    Write-Host "Installing $url..." -ForegroundColor Gray
    Start-Process $PipExe -ArgumentList "install --no-cache-dir $url" -Wait -NoNewWindow
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to install $url"
        Exit 1
    }
}

Write-Host "`n[4/5] Installing PyTorch, torchaudio, and torchvision with ROCm 7.2.1 support..." -ForegroundColor Yellow

$TorchUrls = @(
    "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/torch-2.9.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl",
    "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/torchaudio-2.9.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl",
    "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/torchvision-0.24.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl"
)

foreach ($url in $TorchUrls) {
    Write-Host "Installing $url..." -ForegroundColor Gray
    Start-Process $PipExe -ArgumentList "install --no-cache-dir $url" -Wait -NoNewWindow
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to install $url"
        Exit 1
    }
}

# 5. Install extra dependencies & backend packages
Write-Host "`n[5/5] Installing remaining dependencies..." -ForegroundColor Yellow

# Install numpy v1 to satisfy chatterbox-tts numpy<2 requirement, and setuptools<82 to provide pkg_resources
Start-Process $PipExe -ArgumentList "install numpy==1.26.4 `"setuptools<82`"" -Wait -NoNewWindow

# Install other chatterbox-tts dependencies manually
Write-Host "Installing chatterbox-tts dependencies..." -ForegroundColor Gray
Start-Process $PipExe -ArgumentList "install resemble-perth conformer==0.3.2 safetensors==0.5.3 spacy-pkuseg pykakasi==2.3.0 gradio==6.8.0 pyloudnorm omegaconf librosa==0.11.0 s3tokenizer transformers==5.2.0 diffusers==0.29.0" -Wait -NoNewWindow

# Install chatterbox-tts with --no-deps to prevent it from downgrading torch and torchaudio
Write-Host "Installing chatterbox-tts with --no-deps..." -ForegroundColor Gray
Start-Process $PipExe -ArgumentList "install chatterbox-tts --no-build-isolation --no-deps" -Wait -NoNewWindow

# Upgrade diffusers to 0.39+ for transformers 5.x compatibility (chatterbox pins 0.29.0, override with --no-deps)
Write-Host "Upgrading diffusers to 0.39+ for transformers 5.x compatibility..." -ForegroundColor Gray
Start-Process $PipExe -ArgumentList "install `"diffusers>=0.32.0`" --no-deps" -Wait -NoNewWindow

# Install backend base requirements from requirements.txt
$ReqFile = Join-Path $BackendDir "requirements.txt"
Start-Process $PipExe -ArgumentList "install -r `"$ReqFile`"" -Wait -NoNewWindow

# Install other backend libraries needed for interface (except numpy, torch, torchaudio, transformers, diffusers which are already installed)
Start-Process $PipExe -ArgumentList "install psycopg2-binary moviepy edge-tts torchcodec" -Wait -NoNewWindow

# 6. Verification
Write-Host "`n===================================================" -ForegroundColor Cyan
Write-Host "  Verifying ROCm Installation..." -ForegroundColor Cyan
Write-Host "===================================================" -ForegroundColor Cyan

& $PythonExe -c "import torch; print('PyTorch version:', torch.__version__); print('ROCm/CUDA Available:', torch.cuda.is_available()); print('Device count:', torch.cuda.device_count()); print('Device Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

Write-Host "`nSetup completed!" -ForegroundColor Green
