# Pyt 项目环境激活脚本
# 使用方法：在项目根目录运行 .\activate_env.ps1

Write-Host "=== Pyt 项目环境激活脚本 ===" -ForegroundColor Cyan

# 检查是否在正确的项目目录
if (-not (Test-Path "src\api\app.py")) {
    Write-Host "❌ 错误：请在 Pyt 项目根目录运行此脚本" -ForegroundColor Red
    exit 1
}

# 检查 conda 是否可用
if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    Write-Host "❌ 错误：未找到 conda 命令，请确保 Anaconda/Miniconda 已安装并添加到 PATH" -ForegroundColor Red
    exit 1
}

# 激活 pyt-env 环境
Write-Host "🔄 正在激活 pyt-env 环境..." -ForegroundColor Yellow
try {
    conda activate pyt-env
    Write-Host "✅ 成功激活 pyt-env 环境" -ForegroundColor Green
} catch {
    Write-Host "❌ 激活环境失败：$($_.Exception.Message)" -ForegroundColor Red
    Write-Host "💡 提示：请确保 pyt-env 环境已创建" -ForegroundColor Yellow
    exit 1
}

# 设置 PYTHONPATH
$env:PYTHONPATH = "$PWD\src;$env:PYTHONPATH"
Write-Host "✅ 已设置 PYTHONPATH：$env:PYTHONPATH" -ForegroundColor Green

# 验证环境
Write-Host "\n=== 环境验证 ===" -ForegroundColor Cyan
Write-Host "当前环境：$env:CONDA_DEFAULT_ENV" -ForegroundColor White
Write-Host "Python 版本：$(python --version)" -ForegroundColor White
Write-Host "工作目录：$PWD" -ForegroundColor White

# 提供快捷命令提示
Write-Host "\n=== 可用命令 ===" -ForegroundColor Cyan
Write-Host "启动后端服务：python -m uvicorn src.api.app:app --reload --port 8000" -ForegroundColor White
Write-Host "启动前端服务：cd frontend && python -m http.server 3000" -ForegroundColor White
Write-Host "运行测试：pytest tests/" -ForegroundColor White

Write-Host "\n🎉 环境激活完成！" -ForegroundColor Green