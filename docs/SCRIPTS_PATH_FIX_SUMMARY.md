# 脚本路径修复总结

## 问题概述

在将脚本文件从项目根目录移动到功能分类的子目录后，发现了多个脚本中的路径引用问题。这些脚本原本使用相对路径引用项目文件，移动后路径失效。

## 发现的问题

### 1. Development 目录脚本

**文件**: `development/setup_dev_env.sh`
- **问题**: 脚本中的 `PROJECT_ROOT` 计算错误，仍然使用当前脚本目录作为项目根目录
- **影响**: 无法正确找到项目文件，如 `.python-version`、`requirements.txt` 等
- **修复**: 将 `PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"` 修改为 `PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"`

**文件**: `development/start_dev.sh`
- **问题**: 缺少项目根目录设置，直接使用相对路径 `cd frontend`
- **影响**: 无法找到 `frontend` 目录
- **修复**: 在脚本开头添加项目根目录设置和工作目录切换

### 2. Training 目录脚本

**文件**: `training/start_training.sh`
- **问题**: 使用相对路径引用 `datasets/` 和 `models/` 目录
- **影响**: 无法找到数据集和模型文件
- **修复**: 在脚本开头添加项目根目录设置

**文件**: `training/start_training.ps1`
- **问题**: 使用相对路径引用 `datasets\` 目录
- **影响**: 无法找到数据集文件
- **修复**: 在脚本开头添加 PowerShell 版本的项目根目录设置

### 3. Testing 目录脚本

**文件**: `testing/start_testing.sh`
- **问题**: 使用相对路径引用 `models/` 目录
- **影响**: 无法找到模型文件
- **修复**: 在脚本开头添加项目根目录设置

**文件**: `testing/test_api_curl.sh`
- **问题**: 使用相对路径引用 `tests/fixtures/` 目录
- **影响**: 无法找到测试图片文件
- **修复**: 在脚本开头添加项目根目录设置

### 4. Deployment 目录脚本

**文件**: `deployment/deploy_win.bat`
- **问题**: 使用相对路径引用 `requirements.txt`、`check_ultralytics.py` 等文件
- **影响**: 无法找到依赖文件和检查脚本
- **修复**: 在脚本开头添加 Windows 批处理版本的项目根目录设置

## 修复方案

### Bash 脚本修复模式

```bash
# 设置项目根目录 (脚本现在在子目录中，需要回到上级目录)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
```

### PowerShell 脚本修复模式

```powershell
# 设置项目根目录 (脚本现在在子目录中，需要回到上级目录)
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $ProjectRoot
```

### Windows 批处理脚本修复模式

```batch
REM 设置项目根目录 (脚本现在在子目录中，需要回到上级目录)
cd /d "%~dp0.."
```

## 修复验证

通过测试验证，修复后的脚本能够正确识别项目根目录：

```bash
# 测试命令
cd /Users/zhou/Code/python/Pyt/development
bash -c 'PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; echo "项目根目录: $PROJECT_ROOT"'

# 输出结果
项目根目录: /Users/zhou/Code/python/Pyt
```

## 修复效果

1. **路径引用正确**: 所有脚本现在都能正确找到项目文件和目录
2. **功能完整**: 脚本的原有功能得到保持
3. **向后兼容**: 修复不影响脚本的其他功能
4. **跨平台支持**: 针对不同平台（Linux/macOS、Windows）提供了相应的修复方案

## 注意事项

1. **执行位置**: 修复后的脚本可以从任何位置执行，都会自动切换到正确的项目根目录
2. **相对路径**: 脚本内部的所有相对路径引用现在都基于项目根目录
3. **环境变量**: `PROJECT_ROOT` 变量在脚本中可用，可用于其他路径计算

## 相关文档

- [脚本组织总结](./SCRIPTS_ORGANIZATION_SUMMARY.md)
- [项目清理总结](./PROJECT_CLEANUP_SUMMARY.md)
- [主项目文档](../README.md)
