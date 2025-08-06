# 脚本文件归档整理总结

## 整理概述

本次整理将根目录下的脚本文件按功能分类，创建了专门的目录进行归档管理，提高了项目的组织性和可维护性。

## 归档分类

### 1. 部署脚本 (`deployment/`)
**用途**: 生产环境部署相关脚本

| 脚本文件 | 功能描述 | 适用平台 |
|---------|---------|----------|
| `deploy_win.bat` | Windows一键部署脚本 | Windows 11 + GPU |

**特点**:
- 自动检测conda/venv环境
- GPU加速配置
- 依赖自动安装
- 需要管理员权限

### 2. 开发环境脚本 (`development/`)
**用途**: 开发环境配置和启动

| 脚本文件 | 功能描述 | 适用平台 |
|---------|---------|----------|
| `setup_dev_env.sh` | 开发环境自动配置 | macOS/Linux |
| `start_dev.sh` | 开发服务启动 | macOS/Linux |

**特点**:
- Python版本检查
- 虚拟环境管理
- 依赖验证和安装
- 详细的启动日志

### 3. 训练脚本 (`training/`)
**用途**: 模型训练相关脚本

| 脚本文件 | 功能描述 | 适用平台 |
|---------|---------|----------|
| `start_training.sh` | Linux/macOS训练脚本 | macOS/Linux |
| `start_training.ps1` | Windows训练脚本 | Windows |

**特点**:
- 数据集完整性检查
- 训练参数配置
- 自动创建输出目录
- 支持GPU/CPU训练

### 4. 测试脚本 (`testing/`)
**用途**: 模型测试和API测试

| 脚本文件 | 功能描述 | 适用平台 |
|---------|---------|----------|
| `start_testing.sh` | 模型推理测试 | macOS/Linux |
| `test_api_curl.sh` | API接口测试 | 跨平台 |

**特点**:
- 多种输入源支持
- 可配置检测参数
- 结果可视化
- API功能验证

## 使用指南

### 快速开始

1. **首次环境配置**:
   ```bash
   # macOS/Linux
   ./development/setup_dev_env.sh

   # Windows
   deployment\deploy_win.bat
   ```

2. **日常开发启动**:
   ```bash
   ./development/start_dev.sh
   ```

3. **模型训练**:
   ```bash
   # macOS/Linux
   ./training/start_training.sh

   # Windows
   .\training\start_training.ps1
   ```

4. **模型测试**:
   ```bash
   ./testing/start_testing.sh --source test_image.jpg
   ```

5. **API测试**:
   ```bash
   ./testing/test_api_curl.sh
   ```

### 权限设置

确保脚本有执行权限：
```bash
chmod +x development/*.sh
chmod +x training/*.sh
chmod +x testing/*.sh
```

## 整理效果

### 前后对比

**整理前**:
- 根目录文件数量: 53个
- 脚本文件散布在根目录
- 功能分类不明确

**整理后**:
- 根目录文件数量: 47个（减少6个）
- 脚本按功能分类归档
- 每个目录有详细的README说明

### 优势

1. **结构清晰**: 脚本按功能分类，易于查找
2. **文档完善**: 每个目录都有详细的使用说明
3. **维护便利**: 相关脚本集中管理
4. **新手友好**: 清晰的使用指南和示例

## 目录结构

```
├── deployment/          # 部署脚本
│   ├── README.md       # 部署说明文档
│   └── deploy_win.bat  # Windows部署脚本
├── development/         # 开发环境脚本
│   ├── README.md       # 开发环境说明
│   ├── setup_dev_env.sh # 环境配置脚本
│   └── start_dev.sh    # 开发启动脚本
├── training/            # 训练脚本
│   ├── README.md       # 训练说明文档
│   ├── start_training.sh # Linux/macOS训练脚本
│   └── start_training.ps1 # Windows训练脚本
└── testing/             # 测试脚本
    ├── README.md        # 测试说明文档
    ├── start_testing.sh # 模型测试脚本
    └── test_api_curl.sh # API测试脚本
```

## 相关文档

- [项目主文档](../README.md)
- [项目清理总结](PROJECT_CLEANUP_SUMMARY.md)
- [发网检测文档](README_HAIRNET_DETECTION.md)
- [开发脚本说明](../README_DEV_SCRIPTS.md)

---

*整理完成时间: 2025年1月*
*整理范围: 根目录脚本文件*
*状态: 已完成*
