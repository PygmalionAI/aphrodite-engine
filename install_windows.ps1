# Installing Aphrodite engine on Windows
# Copyright (c) 2024 PygmalionAI

$RequiredVer = "0.6.3"

# Check if Python is installed and version >= 3.8
try {
    $pythonVersion = (python --version 2>&1).ToString().Split(" ")[1]
    $major, $minor = $pythonVersion.Split(".")[0,1]
    
    if ([int]$major -lt 3 -or ([int]$major -eq 3 -and [int]$minor -lt 8)) {
        Write-Error "Python version must be 3.8 or above."
        exit 1
    }
} catch {
    Write-Error "Python is not installed. Please install Python 3.8 or above."
    exit 1
}

# Get Python version for wheel files
$pyVer = python -c "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}-cp{sys.version_info.major}{sys.version_info.minor}')"

# Check if running in venv
$venvStatus = python -c "import sys; print('INVENV' if (hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)) else 'NOVENV')"

if ($venvStatus -eq "NOVENV") {
    Write-Host "Creating new virtual environment..."
    python -m venv venv
    . .\venv\Scripts\Activate.ps1
    Write-Host "Virtual environment created and activated."
} elseif ($venvStatus -eq "INVENV") {
    Write-Host "Already running in virtual environment, continuing..."
} else {
    Write-Error "Failed to check virtual environment status."
    exit 1
}

# Check for -Reinstall parameter
$forceReinstall = $args -contains "-Reinstall"

# Check current aphrodite version if installed
if (-not $forceReinstall) {
    try {
        $installedVer = (pip show aphrodite-engine | Select-String "Version").ToString().Split(" ")[1]
        if ($installedVer -eq $RequiredVer) {
            Write-Host "Aphrodite engine is already at required version $RequiredVer"
            exit 0
        }
    } catch {}
}

# Install/Reinstall packages
Write-Host "Installing PyTorch and xformers..."
pip install "https://download.pytorch.org/whl/cu124/torch-2.4.1%2Bcu124-$pyVer-win_amd64.whl"
pip install "https://downloads.pygmalion.chat/whl/windows/xformers/xformers-0.0.28-$pyVer-win_amd64.whl" --no-deps

Write-Host "Installing Aphrodite engine..."
pip install "https://github.com/PygmalionAI/aphrodite-engine/releases/download/v$RequiredVer/aphrodite_engine-$RequiredVer-cp38-abi3-win_amd64.whl"

Write-Host "Installation complete!"