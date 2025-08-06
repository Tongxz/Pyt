# 开发环境脚本目录

本目录包含开发环境配置和启动相关的脚本文件。

## 脚本说明

### setup_dev_env.sh
- **功能**: 开发环境自动配置脚本
- **主要功能**:
  - 检查Python版本兼容性
  - 自动创建和激活虚拟环境
  - 验证和安装项目依赖
  - 检测缺失依赖并自动安装
  - 环境健康检查
- **使用方法**:
  ```bash
  ./setup_dev_env.sh
  ```

### start_dev.sh
- **功能**: 开发环境一键启动脚本
- **主要功能**:
  - 自动检测和激活Python环境
  - 验证依赖完整性
  - 启动开发服务器
  - 提供详细的启动日志
  - 支持多种启动模式
- **使用方法**:
  ```bash
  ./start_dev.sh
  ```

## 环境要求

- Python 3.10+
- macOS/Linux系统
- 足够的磁盘空间用于依赖安装

## 使用流程

1. 首次使用先运行环境配置脚本:
   ```bash
   ./development/setup_dev_env.sh
   ```

2. 日常开发使用启动脚本:
   ```bash
   ./development/start_dev.sh
   ```

## 故障排除

- 如果遇到权限问题，请确保脚本有执行权限:
  ```bash
  chmod +x development/*.sh
  ```

- 如果Python版本不匹配，请安装正确版本或更新.python-version文件

## 相关文档

- [项目主文档](../README.md)
- [部署脚本](../deployment/README.md)
- [开发指南](../docs/README_DEV_SCRIPTS.md)
