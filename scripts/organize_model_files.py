#!/usr/bin/env python3
"""
模型文件整理脚本

该脚本用于整理项目中的模型文件，包括被Git忽略的文件。
解决了常规文件搜索工具无法识别Git忽略文件的问题。

功能:
1. 自动发现项目中的所有模型文件
2. 将模型文件移动到规范的目录结构中
3. 更新相关配置和文档
4. 生成整理报告
"""

import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

# 配置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ModelFileOrganizer:
    """模型文件整理器"""

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.models_dir = self.project_root / "models"

        # 模型文件扩展名
        self.model_extensions = [
            "*.pt",
            "*.pth",
            "*.onnx",
            "*.h5",
            "*.pkl",
            "*.joblib",
            "*.model",
            "*.weights",
            "*.bin",
        ]

        # 目录映射规则
        self.directory_mapping = {
            "yolov8": "models/yolo",
            "yolo": "models/yolo",
            "hairnet": "models/hairnet_detection",
            "pose": "models/pose_detection",
            "behavior": "models/behavior_recognition",
            "hand": "models/pose_detection",
            "detection": "models/hairnet_detection",
        }

        # 需要排除的目录
        self.exclude_dirs = {
            ".venv",
            "venv",
            ".git",
            "__pycache__",
            ".pytest_cache",
            "node_modules",
            ".tox",
        }

        # 受保护的目录（用户训练的模型，不应被移动）
        self.protected_dirs = {"models/hairnet_model"}  # 用户自己训练的发网检测模型

    def find_model_files(self) -> List[Tuple[Path, str]]:
        """查找所有模型文件，包括被Git忽略的文件"""
        logger.info("开始搜索模型文件...")
        model_files = []

        try:
            # 使用find命令搜索所有模型文件
            find_patterns = []
            for ext in self.model_extensions:
                find_patterns.extend(["-name", ext])
                if ext != self.model_extensions[-1]:
                    find_patterns.append("-o")

            cmd = ["find", str(self.project_root)] + find_patterns
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                for line in result.stdout.strip().split("\n"):
                    if line:
                        file_path = Path(line)
                        # 排除虚拟环境和其他不需要的目录
                        if not any(
                            exclude in file_path.parts for exclude in self.exclude_dirs
                        ):
                            # 检查是否在受保护的目录中
                            is_protected = False
                            for protected_dir in self.protected_dirs:
                                if str(file_path).startswith(
                                    str(self.project_root / protected_dir)
                                ):
                                    is_protected = True
                                    break

                            if not is_protected:
                                # 确定文件类型
                                file_type = self._determine_file_type(file_path)
                                model_files.append((file_path, file_type))
                            else:
                                logger.info(f"跳过受保护的文件: {file_path}")

            logger.info(f"找到 {len(model_files)} 个模型文件")
            return model_files

        except Exception as e:
            logger.error(f"搜索模型文件时出错: {e}")
            return []

    def _determine_file_type(self, file_path: Path) -> str:
        """根据文件名和路径确定文件类型"""
        file_name = file_path.name.lower()
        file_dir = str(file_path.parent).lower()

        # 根据文件名判断
        if "yolov8" in file_name or "yolo" in file_name:
            return "yolo"
        elif "hairnet" in file_name:
            return "hairnet_detection"
        elif "pose" in file_name or "hand" in file_name:
            return "pose_detection"
        elif "behavior" in file_name:
            return "behavior_recognition"

        # 根据目录路径判断
        if "hairnet" in file_dir:
            return "hairnet_detection"
        elif "yolo" in file_dir:
            return "yolo"
        elif "pose" in file_dir:
            return "pose_detection"
        elif "behavior" in file_dir:
            return "behavior_recognition"

        # 默认分类
        return "general"

    def create_directory_structure(self):
        """创建模型目录结构"""
        directories = [
            "models/yolo",
            "models/hairnet_detection",
            "models/hairnet_detection/weights",
            "models/pose_detection",
            "models/behavior_recognition",
        ]

        for dir_path in directories:
            full_path = self.project_root / dir_path
            full_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"创建目录: {dir_path}")

    def organize_files(
        self, model_files: List[Tuple[Path, str]]
    ) -> Dict[str, List[str]]:
        """整理模型文件"""
        organized_files = {"moved": [], "skipped": [], "errors": []}

        for file_path, file_type in model_files:
            try:
                # 确定目标目录
                if file_type == "yolo":
                    target_dir = self.models_dir / "yolo"
                elif file_type == "hairnet_detection":
                    if "weights" in str(file_path.parent).lower():
                        target_dir = self.models_dir / "hairnet_detection" / "weights"
                    else:
                        target_dir = self.models_dir / "hairnet_detection"
                elif file_type == "pose_detection":
                    target_dir = self.models_dir / "pose_detection"
                elif file_type == "behavior_recognition":
                    target_dir = self.models_dir / "behavior_recognition"
                else:
                    target_dir = self.models_dir / "general"

                target_path = target_dir / file_path.name

                # 检查是否已经在正确位置
                if file_path.parent == target_dir:
                    organized_files["skipped"].append(f"{file_path} (已在正确位置)")
                    continue

                # 检查目标文件是否已存在
                if target_path.exists():
                    logger.warning(f"目标文件已存在: {target_path}")
                    organized_files["skipped"].append(
                        f"{file_path} -> {target_path} (目标已存在)"
                    )
                    continue

                # 移动文件
                target_dir.mkdir(parents=True, exist_ok=True)
                shutil.move(str(file_path), str(target_path))
                organized_files["moved"].append(f"{file_path} -> {target_path}")
                logger.info(f"移动文件: {file_path} -> {target_path}")

            except Exception as e:
                error_msg = f"移动文件 {file_path} 时出错: {e}"
                logger.error(error_msg)
                organized_files["errors"].append(error_msg)

        return organized_files

    def generate_report(self, organized_files: Dict[str, List[str]]) -> str:
        """生成整理报告"""
        report = []
        report.append("# 模型文件整理报告\n")
        report.append(
            f"整理时间: {logging.Formatter().formatTime(logging.LogRecord('', 0, '', 0, '', (), None))}\n"
        )

        report.append("## 整理统计\n")
        report.append(f"- 成功移动: {len(organized_files['moved'])} 个文件")
        report.append(f"- 跳过处理: {len(organized_files['skipped'])} 个文件")
        report.append(f"- 处理错误: {len(organized_files['errors'])} 个文件\n")

        if organized_files["moved"]:
            report.append("## 成功移动的文件\n")
            for item in organized_files["moved"]:
                report.append(f"- {item}")
            report.append("")

        if organized_files["skipped"]:
            report.append("## 跳过的文件\n")
            for item in organized_files["skipped"]:
                report.append(f"- {item}")
            report.append("")

        if organized_files["errors"]:
            report.append("## 处理错误\n")
            for item in organized_files["errors"]:
                report.append(f"- {item}")
            report.append("")

        report.append("## 整理后的目录结构\n")
        report.append("```")
        report.append("models/")
        report.append("├── yolo/                    # YOLO系列模型")
        report.append("├── hairnet_detection/       # 发网检测模型")
        report.append("│   └── weights/            # 训练权重文件")
        report.append("├── pose_detection/          # 姿态检测模型")
        report.append("├── behavior_recognition/    # 行为识别模型")
        report.append("└── general/                # 其他模型文件")
        report.append("```\n")

        return "\n".join(report)

    def run(self):
        """执行模型文件整理"""
        logger.info("开始模型文件整理...")

        # 1. 创建目录结构
        self.create_directory_structure()

        # 2. 查找模型文件
        model_files = self.find_model_files()

        if not model_files:
            logger.info("未找到需要整理的模型文件")
            return

        # 3. 整理文件
        organized_files = self.organize_files(model_files)

        # 4. 生成报告
        report = self.generate_report(organized_files)

        # 5. 保存报告
        report_path = (
            self.project_root / "reports" / "model_files_organization_report.md"
        )
        report_path.parent.mkdir(exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)

        logger.info(f"整理完成！报告已保存到: {report_path}")
        print("\n" + "=" * 50)
        print("模型文件整理完成！")
        print(f"详细报告: {report_path}")
        print("=" * 50)


def main():
    """主函数"""
    project_root = os.getcwd()
    organizer = ModelFileOrganizer(project_root)
    organizer.run()


if __name__ == "__main__":
    main()
