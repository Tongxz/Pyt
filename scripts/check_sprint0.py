#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sprint 0 任务完成度检查脚本
Sprint 0 Task Completion Check Script

检查Sprint 0的所有任务是否已完成
Check if all Sprint 0 tasks are completed
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# 颜色定义
class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    PURPLE = '\033[0;35m'
    CYAN = '\033[0;36m'
    WHITE = '\033[1;37m'
    NC = '\033[0m'  # No Color

def log_info(message: str) -> None:
    """输出信息日志"""
    print(f"{Colors.BLUE}[INFO]{Colors.NC} {message}")

def log_success(message: str) -> None:
    """输出成功日志"""
    print(f"{Colors.GREEN}[✓]{Colors.NC} {message}")

def log_warning(message: str) -> None:
    """输出警告日志"""
    print(f"{Colors.YELLOW}[!]{Colors.NC} {message}")

def log_error(message: str) -> None:
    """输出错误日志"""
    print(f"{Colors.RED}[✗]{Colors.NC} {message}")

def check_file_exists(file_path: str) -> bool:
    """检查文件是否存在"""
    return Path(file_path).exists()

def check_directory_exists(dir_path: str) -> bool:
    """检查目录是否存在"""
    return Path(dir_path).is_dir()

def check_project_structure() -> Tuple[int, int]:
    """检查项目结构"""
    log_info("检查项目结构...")
    
    required_files = [
        "README.md",
        "requirements.txt",
        "setup.py",
        "main.py",
        "config/default.yaml",
        "src/__init__.py",
        "src/core/__init__.py",
        "src/detection/__init__.py",
        "src/api/__init__.py",
        "src/utils/__init__.py",
        "tests/__init__.py",
        "tests/unit/__init__.py",
        "tests/integration/__init__.py",
    ]
    
    required_dirs = [
        "src",
        "src/core",
        "src/detection",
        "src/api",
        "src/utils",
        "tests",
        "tests/unit",
        "tests/integration",
        "config",
        "docs",
        "scripts",
    ]
    
    passed = 0
    total = len(required_files) + len(required_dirs)
    
    # 检查文件
    for file_path in required_files:
        if check_file_exists(file_path):
            log_success(f"文件存在: {file_path}")
            passed += 1
        else:
            log_error(f"文件缺失: {file_path}")
    
    # 检查目录
    for dir_path in required_dirs:
        if check_directory_exists(dir_path):
            log_success(f"目录存在: {dir_path}")
            passed += 1
        else:
            log_error(f"目录缺失: {dir_path}")
    
    return passed, total

def check_docker_configuration() -> Tuple[int, int]:
    """检查Docker配置"""
    log_info("检查Docker配置...")
    
    docker_files = [
        "Dockerfile",
        "Dockerfile.dev",
        "docker-compose.yml",
        ".dockerignore",
    ]
    
    passed = 0
    total = len(docker_files)
    
    for file_path in docker_files:
        if check_file_exists(file_path):
            log_success(f"Docker文件存在: {file_path}")
            passed += 1
        else:
            log_error(f"Docker文件缺失: {file_path}")
    
    return passed, total

def check_cicd_configuration() -> Tuple[int, int]:
    """检查CI/CD配置"""
    log_info("检查CI/CD配置...")
    
    cicd_files = [
        ".github/workflows/ci.yml",
        ".pre-commit-config.yaml",
        "Makefile",
    ]
    
    passed = 0
    total = len(cicd_files)
    
    for file_path in cicd_files:
        if check_file_exists(file_path):
            log_success(f"CI/CD文件存在: {file_path}")
            passed += 1
        else:
            log_error(f"CI/CD文件缺失: {file_path}")
    
    return passed, total

def check_team_collaboration() -> Tuple[int, int]:
    """检查团队协作工具配置"""
    log_info("检查团队协作工具配置...")
    
    collaboration_files = [
        ".vscode/settings.json",
        ".vscode/extensions.json",
        ".vscode/launch.json",
        ".vscode/tasks.json",
        "CONTRIBUTING.md",
    ]
    
    passed = 0
    total = len(collaboration_files)
    
    for file_path in collaboration_files:
        if check_file_exists(file_path):
            log_success(f"协作配置文件存在: {file_path}")
            passed += 1
        else:
            log_error(f"协作配置文件缺失: {file_path}")
    
    return passed, total

def check_development_scripts() -> Tuple[int, int]:
    """检查开发脚本"""
    log_info("检查开发脚本...")
    
    script_files = [
        "scripts/setup_dev.sh",
        "scripts/init_db.sql",
        "scripts/check_sprint0.py",
    ]
    
    passed = 0
    total = len(script_files)
    
    for file_path in script_files:
        if check_file_exists(file_path):
            log_success(f"脚本文件存在: {file_path}")
            passed += 1
            # 检查脚本是否可执行
            if file_path.endswith('.sh') and os.access(file_path, os.X_OK):
                log_success(f"脚本可执行: {file_path}")
            elif file_path.endswith('.sh'):
                log_warning(f"脚本不可执行: {file_path}")
        else:
            log_error(f"脚本文件缺失: {file_path}")
    
    return passed, total

def check_project_management() -> Tuple[int, int]:
    """检查项目管理文件"""
    log_info("检查项目管理文件...")
    
    management_files = [
        "敏捷迭代执行方案.md",
        ".gitignore",
        "LICENSE",
    ]
    
    passed = 0
    total = len(management_files)
    
    for file_path in management_files:
        if check_file_exists(file_path):
            log_success(f"项目管理文件存在: {file_path}")
            passed += 1
        else:
            log_error(f"项目管理文件缺失: {file_path}")
    
    return passed, total

def check_code_quality() -> Tuple[int, int]:
    """检查代码质量配置"""
    log_info("检查代码质量配置...")
    
    quality_files = [
        "pyproject.toml",
        ".flake8",
        "mypy.ini",
    ]
    
    passed = 0
    total = len(quality_files)
    
    for file_path in quality_files:
        if check_file_exists(file_path):
            log_success(f"代码质量配置存在: {file_path}")
            passed += 1
        else:
            log_warning(f"代码质量配置缺失: {file_path}")
    
    return passed, total

def generate_report(results: Dict[str, Tuple[int, int]]) -> None:
    """生成检查报告"""
    print("\n" + "=" * 60)
    print(f"{Colors.WHITE}Sprint 0 任务完成度报告{Colors.NC}")
    print("=" * 60)
    
    total_passed = 0
    total_tasks = 0
    
    for category, (passed, total) in results.items():
        percentage = (passed / total * 100) if total > 0 else 0
        status_color = Colors.GREEN if percentage >= 90 else Colors.YELLOW if percentage >= 70 else Colors.RED
        
        print(f"{Colors.CYAN}{category}:{Colors.NC}")
        print(f"  完成: {status_color}{passed}/{total}{Colors.NC} ({percentage:.1f}%)")
        
        total_passed += passed
        total_tasks += total
    
    print("\n" + "-" * 60)
    overall_percentage = (total_passed / total_tasks * 100) if total_tasks > 0 else 0
    overall_color = Colors.GREEN if overall_percentage >= 90 else Colors.YELLOW if overall_percentage >= 70 else Colors.RED
    
    print(f"{Colors.WHITE}总体完成度: {overall_color}{total_passed}/{total_tasks}{Colors.NC} ({overall_percentage:.1f}%)")
    
    if overall_percentage >= 90:
        print(f"\n{Colors.GREEN}🎉 Sprint 0 任务基本完成！可以开始 Sprint 1 开发。{Colors.NC}")
    elif overall_percentage >= 70:
        print(f"\n{Colors.YELLOW}⚠️  Sprint 0 任务大部分完成，建议补充完成剩余任务。{Colors.NC}")
    else:
        print(f"\n{Colors.RED}❌ Sprint 0 任务完成度较低，需要继续完善基础设施。{Colors.NC}")
    
    print("\n建议下一步:")
    if overall_percentage < 100:
        print("1. 补充完成缺失的文件和配置")
        print("2. 运行开发环境设置脚本: ./scripts/setup_dev.sh")
        print("3. 验证开发环境是否正常工作")
    print("4. 开始 Sprint 1 核心检测MVP开发")
    print("5. 定期运行此脚本检查项目状态")

def main() -> None:
    """主函数"""
    print(f"{Colors.PURPLE}" + "=" * 60)
    print("  Sprint 0 任务完成度检查")
    print("  Sprint 0 Task Completion Check")
    print("=" * 60 + f"{Colors.NC}\n")
    
    # 检查各个方面
    results = {
        "项目结构": check_project_structure(),
        "Docker配置": check_docker_configuration(),
        "CI/CD配置": check_cicd_configuration(),
        "团队协作工具": check_team_collaboration(),
        "开发脚本": check_development_scripts(),
        "项目管理文件": check_project_management(),
        "代码质量配置": check_code_quality(),
    }
    
    # 生成报告
    generate_report(results)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}检查被用户中断{Colors.NC}")
        sys.exit(1)
    except Exception as e:
        print(f"\n{Colors.RED}检查过程中发生错误: {e}{Colors.NC}")
        sys.exit(1)