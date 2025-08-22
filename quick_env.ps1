# Quick environment activation script
# Activate pyt-env environment and set PYTHONPATH

conda activate pyt-env
$env:PYTHONPATH = "$PWD\src;$env:PYTHONPATH"
Write-Host "Environment pyt-env activated and PYTHONPATH set" -ForegroundColor Green