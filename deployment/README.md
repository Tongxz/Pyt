# 部署脚本目录

本目录包含项目部署相关的脚本文件。

## 脚本说明

### deploy_win.bat
- **功能**: Windows系统一键部署脚本
- **适用环境**: Windows 11 + GPU (4090)
- **主要功能**:
  - 自动检测并配置conda或venv环境
  - 安装项目依赖
  - 配置GPU加速（CUDA）
  - 启动服务
- **使用方法**: 以管理员身份运行
  ```cmd
  deploy_win.bat
  ```

## 使用注意事项

1. Windows部署脚本需要管理员权限
2. 确保已安装Python 3.10+
3. GPU版本需要CUDA支持
4. 首次运行可能需要较长时间安装依赖

## 相关文档

- [项目主文档](../README.md)
- [开发环境配置](../development/README.md)
