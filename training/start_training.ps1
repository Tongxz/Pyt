# Hairnet Detection Model Training Script (PowerShell Version)

# 设置项目根目录 (脚本现在在training子目录中，需要回到上级目录)
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $ProjectRoot

param(
    [int]$Epochs = 100,
    [int]$BatchSize = 16,
    [int]$ImgSize = 640,
    [string]$Weights = "models/yolo/yolov8m.pt",
    [string]$Device = "",
    [switch]$Help
)

# Show help information
if ($Help) {
    Write-Host "Usage: .\start_training.ps1 [Options]"
    Write-Host "Options:"
    Write-Host "  -Epochs N        Number of training epochs, default is 100"
    Write-Host "  -BatchSize N     Batch size, default is 16"
    Write-Host "  -ImgSize N       Image size, default is 640"
    Write-Host "  -Weights FILE    Initial weights, default is models/yolo/yolov8m.pt"
    Write-Host "  -Device STR      Training device, e.g. cuda:0 or cpu"
    Write-Host "  -Help            Show this help information"
    exit 0
}

# Check Python environment
try {
    $pythonPath = & "C:\Users\Lenovo\.conda\envs\pyt-env\python.exe" -c "import sys; print(sys.executable)" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Error: Cannot find pyt-env Python environment" -ForegroundColor Red
        exit 1
    }
    Write-Host "Using Python environment: $pythonPath" -ForegroundColor Green
} catch {
    Write-Host "Error: Cannot access Python environment" -ForegroundColor Red
    exit 1
}

# Check if ultralytics is installed
Write-Host "Checking ultralytics installation..." -ForegroundColor Yellow
$ultralyticsCheck = & "C:\Users\Lenovo\.conda\envs\pyt-env\python.exe" -c "import ultralytics; print('installed')" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Installing ultralytics..." -ForegroundColor Yellow
    & "C:\Users\Lenovo\.conda\envs\pyt-env\python.exe" -m pip install ultralytics
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Error: ultralytics installation failed" -ForegroundColor Red
        exit 1
    }
}

# Check dataset directories
if (!(Test-Path "datasets\hairnet\train\images") -or !(Test-Path "datasets\hairnet\valid\images")) {
    Write-Host "Error: Dataset directories incomplete, please prepare dataset first" -ForegroundColor Red
    Write-Host "Use the following command to prepare dataset:" -ForegroundColor Yellow
    Write-Host "python prepare_roboflow_dataset.py --input /path/to/roboflow_dataset.zip --output datasets/hairnet"
    exit 1
}

# Check dataset configuration file
if (!(Test-Path "datasets\hairnet\data.yaml")) {
    Write-Host "Error: Dataset configuration file does not exist, please prepare dataset first" -ForegroundColor Red
    exit 1
}

# Create model save directory
if (!(Test-Path "models")) {
    New-Item -ItemType Directory -Path "models" -Force | Out-Null
}

# Build training command arguments
$args = @(
    "train_hairnet_model.py",
    "--data", "datasets/hairnet/data.yaml",
    "--epochs", $Epochs,
    "--batch-size", $BatchSize,
    "--img-size", $ImgSize,
    "--weights", $Weights,
    "--pretrained"
)

# Add device parameter if specified
if ($Device -ne "") {
    $args += "--device", $Device
}

# Display training information
Write-Host "=== Hairnet Detection Model Training ===" -ForegroundColor Cyan
Write-Host "Dataset: datasets/hairnet"
Write-Host "Training epochs: $Epochs"
Write-Host "Batch size: $BatchSize"
Write-Host "Image size: $ImgSize"
Write-Host "Initial weights: $Weights"
if ($Device -ne "") {
    Write-Host "Training device: $Device"
}
Write-Host ""

# Confirm to start training
$confirm = Read-Host "Start training? [y/N]"
if ($confirm -notmatch "^[yY]([eE][sS])?$") {
    Write-Host "Training cancelled" -ForegroundColor Yellow
    exit 0
}

# Start training
Write-Host "Starting training..." -ForegroundColor Green
Write-Host "Executing command: python $($args -join ' ')" -ForegroundColor Gray
Write-Host ""

# Execute training command
& "C:\Users\Lenovo\.conda\envs\pyt-env\python.exe" @args

# Check training results
if ($LASTEXITCODE -eq 0) {
    Write-Host "Training completed!" -ForegroundColor Green
    Write-Host "Model saved at: models/hairnet_detection/models/hairnet_detection/hairnet_detection.pt"
    Write-Host "Use the following command to test the model:" -ForegroundColor Yellow
    Write-Host "python test_hairnet_model.py --weights models/hairnet_detection/models/hairnet_detection/hairnet_detection.pt --source path/to/test/image.jpg --view-img"
} else {
    Write-Host "Training failed, please check error messages" -ForegroundColor Red
    exit 1
}
