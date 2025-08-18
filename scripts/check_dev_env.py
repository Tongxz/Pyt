#!/usr/bin/env python3
"""
开发环境检测脚本
自动检测Python版本、依赖包版本，并提供详细的环境报告
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pkg_resources


# 颜色定义
class Colors:
    RED = "\033[0;31m"
    GREEN = "\033[0;32m"
    YELLOW = "\033[1;33m"
    BLUE = "\033[0;34m"
    PURPLE = "\033[0;35m"
    CYAN = "\033[0;36m"
    NC = "\033[0m"  # No Color


def log_info(message: str):
    print(f"{Colors.BLUE}[INFO]{Colors.NC} {message}")


def log_success(message: str):
    print(f"{Colors.GREEN}[SUCCESS]{Colors.NC} {message}")


def log_warning(message: str):
    print(f"{Colors.YELLOW}[WARNING]{Colors.NC} {message}")


def log_error(message: str):
    print(f"{Colors.RED}[ERROR]{Colors.NC} {message}")


def log_header(message: str):
    print(f"\n{Colors.PURPLE}{'='*60}{Colors.NC}")
    print(f"{Colors.PURPLE}{message.center(60)}{Colors.NC}")
    print(f"{Colors.PURPLE}{'='*60}{Colors.NC}\n")


class DevEnvironmentChecker:
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.required_python_version = self._get_required_python_version()
        self.requirements = self._parse_requirements()

    def _get_required_python_version(self) -> str:
        """获取项目要求的Python版本"""
        python_version_file = self.project_root / ".python-version"
        if python_version_file.exists():
            return python_version_file.read_text().strip()

        # 从pyproject.toml中获取
        pyproject_file = self.project_root / "pyproject.toml"
        if pyproject_file.exists():
            content = pyproject_file.read_text()
            for line in content.split("\n"):
                if "requires-python" in line:
                    # 提取版本号
                    import re

                    match = re.search(r">=([0-9.]+)", line)
                    if match:
                        return match.group(1)

        return "3.8.0"  # 默认版本

    def _parse_requirements(self) -> Dict[str, str]:
        """解析requirements文件"""
        requirements = {}

        # 尝试不同的requirements文件
        req_files = ["requirements.txt", "requirements.dev.txt", "requirements-dev.txt"]

        for req_file in req_files:
            req_path = self.project_root / req_file
            if req_path.exists():
                log_info(f"解析依赖文件: {req_file}")
                with open(req_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#"):
                            # 解析包名和版本
                            if ">=" in line:
                                package, version = line.split(">=", 1)
                                requirements[package.strip()] = version.split(",")[
                                    0
                                ].strip()
                            elif "==" in line:
                                package, version = line.split("==", 1)
                                requirements[package.strip()] = version.strip()
                            elif ">" in line:
                                package, version = line.split(">", 1)
                                requirements[package.strip()] = version.split(",")[
                                    0
                                ].strip()
                            else:
                                # 没有版本要求
                                requirements[line.strip()] = "any"
                break

        return requirements

    def check_python_version(self) -> bool:
        """检查Python版本"""
        log_header("Python版本检查")

        current_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        required_version = self.required_python_version

        log_info(f"当前Python版本: {current_version}")
        log_info(f"要求Python版本: >= {required_version}")
        log_info(f"Python可执行文件: {sys.executable}")
        log_info(f"Python路径: {sys.path[0]}")

        # 版本比较
        current_parts = [int(x) for x in current_version.split(".")]
        required_parts = [int(x) for x in required_version.split(".")]

        if current_parts >= required_parts:
            log_success("Python版本符合要求")
            return True
        else:
            log_error(f"Python版本不符合要求，需要 >= {required_version}")
            return False

    def check_virtual_environment(self) -> bool:
        """检查虚拟环境"""
        log_header("虚拟环境检查")

        # 检查是否在虚拟环境中
        in_venv = hasattr(sys, "real_prefix") or (
            hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
        )

        if in_venv:
            log_success("当前在虚拟环境中")
            log_info(f"虚拟环境路径: {sys.prefix}")
            return True
        else:
            log_warning("当前不在虚拟环境中")

            # 检查项目目录下是否有虚拟环境
            venv_dirs = ["venv", ".venv", "env", ".env"]
            for venv_dir in venv_dirs:
                venv_path = self.project_root / venv_dir
                if venv_path.exists():
                    log_info(f"发现虚拟环境目录: {venv_path}")
                    log_info(f"激活命令: source {venv_path}/bin/activate")
                    return False

            log_warning("未发现虚拟环境目录")
            log_info("建议创建虚拟环境: python3 -m venv venv")
            return False

    def get_installed_packages(self) -> Dict[str, str]:
        """获取已安装的包列表"""
        installed = {}
        try:
            for dist in pkg_resources.working_set:
                installed[dist.project_name.lower()] = dist.version
        except Exception as e:
            log_warning(f"获取已安装包列表失败: {e}")

        return installed

    def check_dependencies(self) -> Tuple[List[str], List[str], List[str]]:
        """检查依赖包"""
        log_header("依赖包检查")

        installed_packages = self.get_installed_packages()

        satisfied = []
        missing = []
        outdated = []

        # 关键包列表
        key_packages = [
            "torch",
            "torchvision",
            "torchaudio",
            "ultralytics",
            "opencv-python",
            "fastapi",
            "numpy",
            "pillow",
            "pyyaml",
            "uvicorn",
        ]

        log_info("检查关键依赖包:")
        print("-" * 60)

        for package in key_packages:
            package_lower = package.lower().replace("-", "_")
            alt_package = package.replace("-", "_")

            # 检查包是否安装（考虑不同的命名方式）
            installed_version = None
            for pkg_name in [package, package_lower, alt_package]:
                if pkg_name in installed_packages:
                    installed_version = installed_packages[pkg_name]
                    break

            if installed_version:
                required_version = self.requirements.get(package, "any")
                status = f"{Colors.GREEN}✓{Colors.NC}"

                if required_version != "any":
                    # 简单版本比较
                    try:
                        if self._compare_versions(installed_version, required_version):
                            satisfied.append(package)
                            print(
                                f"{status} {package:<20} {installed_version:<15} (>= {required_version})"
                            )
                        else:
                            outdated.append(package)
                            status = f"{Colors.YELLOW}⚠{Colors.NC}"
                            print(
                                f"{status} {package:<20} {installed_version:<15} (需要 >= {required_version})"
                            )
                    except:
                        satisfied.append(package)
                        print(
                            f"{status} {package:<20} {installed_version:<15} (版本检查失败)"
                        )
                else:
                    satisfied.append(package)
                    print(f"{status} {package:<20} {installed_version:<15} (任意版本)")
            else:
                missing.append(package)
                status = f"{Colors.RED}✗{Colors.NC}"
                required_version = self.requirements.get(package, "latest")
                print(f"{status} {package:<20} {'未安装':<15} (需要 {required_version})")

        print("-" * 60)

        return satisfied, missing, outdated

    def _compare_versions(self, installed: str, required: str) -> bool:
        """简单的版本比较"""
        try:
            installed_parts = [int(x) for x in installed.split(".")[:3]]
            required_parts = [int(x) for x in required.split(".")[:3]]

            # 补齐版本号
            while len(installed_parts) < 3:
                installed_parts.append(0)
            while len(required_parts) < 3:
                required_parts.append(0)

            return installed_parts >= required_parts
        except:
            return True  # 如果比较失败，假设满足要求

    def check_gpu_support(self) -> bool:
        """检查GPU支持"""
        log_header("GPU支持检查")

        try:
            import torch

            log_info(f"PyTorch版本: {torch.__version__}")
            log_info(f"CUDA可用: {torch.cuda.is_available()}")

            if torch.cuda.is_available():
                cuda_version = getattr(torch.version, "cuda", "Unknown")
                log_info(f"CUDA版本: {cuda_version}")
                log_info(f"GPU数量: {torch.cuda.device_count()}")
                for i in range(torch.cuda.device_count()):
                    gpu_name = torch.cuda.get_device_name(i)
                    log_info(f"GPU {i}: {gpu_name}")
                log_success("GPU支持可用")
                return True
            else:
                log_warning("CUDA不可用，将使用CPU")
                return False

        except ImportError:
            log_warning("PyTorch未安装，无法检查GPU支持")
            return False
        except Exception as e:
            log_error(f"GPU检查失败: {e}")
            return False

    def check_model_files(self) -> bool:
        """检查模型文件"""
        log_header("模型文件检查")

        model_files = [
            "models/hairnet_detection/models/hairnet_detection/hairnet_detection.pt",
            "models/hairnet_model/weights/best.pt",
            "models/hairnet_model/weights/last.pt",
            "models/yolo/yolov8n.pt",
            "models/yolo/yolov8m.pt",
        ]

        found_models = []
        missing_models = []

        for model_file in model_files:
            model_path = self.project_root / model_file
            if model_path.exists():
                size_mb = model_path.stat().st_size / (1024 * 1024)
                log_success(f"✓ {model_file} ({size_mb:.1f} MB)")
                found_models.append(model_file)
            else:
                log_warning(f"✗ {model_file} (未找到)")
                missing_models.append(model_file)

        if found_models:
            log_success(f"找到 {len(found_models)} 个模型文件")

        if missing_models:
            log_warning(f"缺少 {len(missing_models)} 个模型文件")
            log_info("可以通过训练或下载获取缺失的模型文件")

        return len(found_models) > 0

    def test_imports(self) -> bool:
        """测试关键模块导入"""
        log_header("模块导入测试")

        test_modules = [
            ("torch", "PyTorch"),
            ("torchvision", "TorchVision"),
            ("cv2", "OpenCV"),
            ("ultralytics", "Ultralytics"),
            ("fastapi", "FastAPI"),
            ("numpy", "NumPy"),
            ("PIL", "Pillow"),
            ("yaml", "PyYAML"),
            ("uvicorn", "Uvicorn"),
        ]

        success_count = 0

        for module_name, display_name in test_modules:
            try:
                __import__(module_name)
                log_success(f"✓ {display_name}")
                success_count += 1
            except ImportError as e:
                log_error(f"✗ {display_name}: {e}")
            except Exception as e:
                log_warning(f"⚠ {display_name}: {e}")

        success_rate = success_count / len(test_modules) * 100
        log_info(f"导入成功率: {success_rate:.1f}% ({success_count}/{len(test_modules)})")

        return success_rate >= 80

    def generate_report(self) -> Dict:
        """生成环境报告"""
        log_header("生成环境报告")

        python_ok = self.check_python_version()
        venv_ok = self.check_virtual_environment()
        satisfied, missing, outdated = self.check_dependencies()
        gpu_ok = self.check_gpu_support()
        models_ok = self.check_model_files()
        imports_ok = self.test_imports()

        report = {
            "timestamp": str(subprocess.check_output(["date"], text=True).strip()),
            "python_version": {
                "current": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                "required": self.required_python_version,
                "ok": python_ok,
            },
            "virtual_environment": {
                "active": hasattr(sys, "real_prefix")
                or (hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix),
                "ok": venv_ok,
            },
            "dependencies": {
                "satisfied": satisfied,
                "missing": missing,
                "outdated": outdated,
                "ok": len(missing) == 0,
            },
            "gpu_support": {"available": gpu_ok, "ok": True},  # GPU是可选的
            "model_files": {"ok": models_ok},
            "imports": {"ok": imports_ok},
            "overall_status": python_ok and len(missing) == 0 and imports_ok,
        }

        return report

    def print_summary(self, report: Dict):
        """打印总结"""
        log_header("环境检查总结")

        if report["overall_status"]:
            log_success("🎉 开发环境配置正确，可以开始开发！")
        else:
            log_warning("⚠️  开发环境存在问题，需要修复")

        print("\n详细状态:")
        print("-" * 40)

        status_items = [
            ("Python版本", report["python_version"]["ok"]),
            ("虚拟环境", report["virtual_environment"]["ok"]),
            ("依赖包", report["dependencies"]["ok"]),
            ("GPU支持", report["gpu_support"]["available"]),
            ("模型文件", report["model_files"]["ok"]),
            ("模块导入", report["imports"]["ok"]),
        ]

        for item, status in status_items:
            icon = "✓" if status else "✗"
            color = Colors.GREEN if status else Colors.RED
            print(f"{color}{icon}{Colors.NC} {item}")

        if report["dependencies"]["missing"]:
            print(f"\n{Colors.YELLOW}缺失的依赖包:{Colors.NC}")
            for pkg in report["dependencies"]["missing"]:
                print(f"  - {pkg}")
            print(f"\n安装命令: pip install {' '.join(report['dependencies']['missing'])}")

        if report["dependencies"]["outdated"]:
            print(f"\n{Colors.YELLOW}需要更新的包:{Colors.NC}")
            for pkg in report["dependencies"]["outdated"]:
                print(f"  - {pkg}")
            print(
                f"\n更新命令: pip install --upgrade {' '.join(report['dependencies']['outdated'])}"
            )

        print("\n" + "=" * 60)


def main():
    """主函数"""
    print(f"{Colors.CYAN}开发环境检测脚本{Colors.NC}")
    print(f"{Colors.CYAN}项目: 人体行为检测系统{Colors.NC}\n")

    checker = DevEnvironmentChecker()
    report = checker.generate_report()
    checker.print_summary(report)

    # 保存报告到文件
    report_file = checker.project_root / "dev_env_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    log_info(f"详细报告已保存到: {report_file}")

    return 0 if report["overall_status"] else 1


if __name__ == "__main__":
    sys.exit(main())
