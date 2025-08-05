#!/usr/bin/env python3
"""
Sprint 全面检查脚本
Comprehensive Sprint Check Script

检查Sprint 0, Sprint 1, Sprint 2的实现情况
Check implementation status of Sprint 0, Sprint 1, Sprint 2
"""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple


# 颜色定义
class Colors:
    RED = "\033[0;31m"
    GREEN = "\033[0;32m"
    YELLOW = "\033[1;33m"
    BLUE = "\033[0;34m"
    PURPLE = "\033[0;35m"
    CYAN = "\033[0;36m"
    WHITE = "\033[1;37m"
    NC = "\033[0m"  # No Color


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


def check_module_import(module_path: str, module_name: str) -> bool:
    """检查模块是否可以导入"""
    try:
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            return False
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return True
    except Exception as e:
        log_error(f"模块导入失败 {module_name}: {e}")
        return False


def check_sprint0() -> Tuple[int, int]:
    """检查Sprint 0实现情况"""
    log_info("检查Sprint 0: 项目启动与环境搭建...")

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
        "Dockerfile",
        "docker-compose.yml",
        ".github/workflows/ci.yml",
        "scripts/setup_dev.sh",
    ]

    passed = 0
    total = len(required_files)

    for file_path in required_files:
        if check_file_exists(file_path):
            log_success(f"Sprint 0 - 文件存在: {file_path}")
            passed += 1
        else:
            log_error(f"Sprint 0 - 文件缺失: {file_path}")

    return passed, total


def check_sprint1() -> Tuple[int, int]:
    """检查Sprint 1: 核心检测MVP实现情况"""
    log_info("检查Sprint 1: 核心检测MVP...")

    # 核心模块文件
    core_files = [
        "src/core/detector.py",
        "src/core/hairnet_detector.py",
        "src/core/data_manager.py",
        "src/api/app.py",
        "frontend/index.html",
        "frontend/app.js",
    ]

    # API端点检查
    api_endpoints = [
        "/health",
        "/api/v1/detect/image",
        "/api/v1/detect/hairnet",
        "/api/statistics/realtime",
    ]

    passed = 0
    total = (
        len(core_files) + len(api_endpoints) + 3
    )  # +3 for module imports and functionality

    # 检查核心文件
    for file_path in core_files:
        if check_file_exists(file_path):
            log_success(f"Sprint 1 - 核心文件存在: {file_path}")
            passed += 1
        else:
            log_error(f"Sprint 1 - 核心文件缺失: {file_path}")

    # 检查模块导入
    if check_file_exists("src/core/detector.py"):
        if check_module_import("src/core/detector.py", "detector"):
            log_success("Sprint 1 - HumanDetector模块可导入")
            passed += 1
        else:
            log_error("Sprint 1 - HumanDetector模块导入失败")
    else:
        log_error("Sprint 1 - HumanDetector模块文件不存在")

    if check_file_exists("src/core/hairnet_detector.py"):
        if check_module_import("src/core/hairnet_detector.py", "hairnet_detector"):
            log_success("Sprint 1 - HairnetDetector模块可导入")
            passed += 1
        else:
            log_error("Sprint 1 - HairnetDetector模块导入失败")
    else:
        log_error("Sprint 1 - HairnetDetector模块文件不存在")

    if check_file_exists("src/core/data_manager.py"):
        if check_module_import("src/core/data_manager.py", "data_manager"):
            log_success("Sprint 1 - DataManager模块可导入")
            passed += 1
        else:
            log_error("Sprint 1 - DataManager模块导入失败")
    else:
        log_error("Sprint 1 - DataManager模块文件不存在")

    # 检查API功能（通过代码分析）
    if check_file_exists("src/api/app.py"):
        with open("src/api/app.py", "r", encoding="utf-8") as f:
            api_content = f.read()

        # 改进的API端点检测逻辑
        endpoint_patterns = {
            "/health": '@app.get("/health")',
            "/api/v1/detect/image": '@app.post("/api/v1/detect/image")',
            "/api/v1/detect/hairnet": '@app.post("/api/v1/detect/hairnet")',
            "/api/statistics/realtime": '@app.get("/api/statistics/realtime")',
        }

        for endpoint, pattern in endpoint_patterns.items():
            if pattern in api_content:
                log_success(f"Sprint 1 - API端点实现: {endpoint}")
                passed += 1
            else:
                log_error(f"Sprint 1 - API端点缺失: {endpoint}")

    return passed, total


def check_sprint2() -> Tuple[int, int]:
    """检查Sprint 2: 发网检测功能实现情况"""
    log_info("检查Sprint 2: 发网检测功能...")

    # Sprint 2 特定功能
    sprint2_features = [
        ("发网检测算法", "HairnetCNN", "src/core/hairnet_detector.py"),
        ("发网检测流水线", "HairnetDetectionPipeline", "src/core/hairnet_detector.py"),
        ("数据库存储", "save_detection_result", "src/core/data_manager.py"),
        ("统计功能", "get_realtime_statistics", "src/core/data_manager.py"),
        ("历史记录", "get_detection_history", "src/core/data_manager.py"),
        ("发网检测API", "/api/v1/detect/hairnet", "src/api/app.py"),
        ("统计API", "/api/statistics", "src/api/app.py"),
        ("前端发网显示", "displayDetectionResult", "frontend/app.js"),
    ]

    passed = 0
    total = len(sprint2_features)

    for feature_name, feature_key, file_path in sprint2_features:
        if check_file_exists(file_path):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                if feature_key in content:
                    log_success(f"Sprint 2 - {feature_name}功能实现")
                    passed += 1
                else:
                    log_error(f"Sprint 2 - {feature_name}功能缺失")
            except Exception as e:
                log_error(f"Sprint 2 - 检查{feature_name}时出错: {e}")
        else:
            log_error(f"Sprint 2 - {feature_name}相关文件不存在: {file_path}")

    return passed, total


def check_additional_features() -> Tuple[int, int]:
    """检查额外功能实现情况"""
    log_info("检查额外功能...")

    additional_features = [
        ("WebSocket实时通信", "websocket_endpoint", "src/api/app.py"),
        ("视频处理", "detect_hairnet_video", "src/api/app.py"),
        ("统计图表", "statistics.html", "frontend/statistics.html"),
        ("配置管理", "default.yaml", "config/default.yaml"),
        ("Docker支持", "Dockerfile", "Dockerfile"),
        ("CI/CD流水线", "ci.yml", ".github/workflows/ci.yml"),
    ]

    passed = 0
    total = len(additional_features)

    for feature_name, feature_key, file_path in additional_features:
        if check_file_exists(file_path):
            log_success(f"额外功能 - {feature_name}已实现")
            passed += 1
        else:
            log_warning(f"额外功能 - {feature_name}未实现")

    return passed, total


def generate_comprehensive_report(results: Dict[str, Tuple[int, int]]) -> None:
    """生成综合报告"""
    print(f"\n{Colors.CYAN}{'='*80}{Colors.NC}")
    print(f"{Colors.WHITE}Sprint 全面实现情况报告{Colors.NC}")
    print(f"{Colors.CYAN}{'='*80}{Colors.NC}")

    total_passed = 0
    total_tasks = 0

    for sprint_name, (passed, total) in results.items():
        percentage = (passed / total * 100) if total > 0 else 0
        color = (
            Colors.GREEN
            if percentage >= 80
            else Colors.YELLOW
            if percentage >= 60
            else Colors.RED
        )

        print(f"{sprint_name}:")
        print(f"  完成: {color}{passed}/{total} ({percentage:.1f}%){Colors.NC}")

        total_passed += passed
        total_tasks += total

    print(f"{Colors.CYAN}{'-'*80}{Colors.NC}")
    overall_percentage = (total_passed / total_tasks * 100) if total_tasks > 0 else 0
    overall_color = (
        Colors.GREEN
        if overall_percentage >= 80
        else Colors.YELLOW
        if overall_percentage >= 60
        else Colors.RED
    )

    print(
        f"总体完成度: {overall_color}{total_passed}/{total_tasks} ({overall_percentage:.1f}%){Colors.NC}"
    )

    # 给出建议
    print(f"\n{Colors.WHITE}建议下一步:{Colors.NC}")

    if overall_percentage >= 90:
        print(f"{Colors.GREEN}🎉 项目实现度很高！可以考虑:{Colors.NC}")
        print("1. 进行全面测试")
        print("2. 优化性能")
        print("3. 准备生产部署")
        print("4. 开始Sprint 3开发")
    elif overall_percentage >= 70:
        print(f"{Colors.YELLOW}📋 项目基本完成，需要:{Colors.NC}")
        print("1. 完善缺失的核心功能")
        print("2. 增加测试覆盖率")
        print("3. 修复已知问题")
        print("4. 完善文档")
    else:
        print(f"{Colors.RED}⚠️ 项目需要重点关注:{Colors.NC}")
        print("1. 优先完成Sprint 1核心功能")
        print("2. 确保基础架构稳定")
        print("3. 解决环境配置问题")
        print("4. 重新评估开发计划")


def main() -> None:
    """主函数"""
    print(f"{Colors.PURPLE}Sprint 全面检查开始...{Colors.NC}\n")

    # 检查各个Sprint
    results = {
        "Sprint 0 (项目启动)": check_sprint0(),
        "Sprint 1 (核心MVP)": check_sprint1(),
        "Sprint 2 (发网检测)": check_sprint2(),
        "额外功能": check_additional_features(),
    }

    # 生成报告
    generate_comprehensive_report(results)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}检查被用户中断{Colors.NC}")
        sys.exit(1)
    except Exception as e:
        print(f"\n{Colors.RED}检查过程中发生错误: {e}{Colors.NC}")
        sys.exit(1)
