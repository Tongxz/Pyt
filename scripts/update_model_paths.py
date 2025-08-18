#!/usr/bin/env python3
"""
模型路径更新脚本

该脚本用于更新项目中所有模型文件的路径引用，
将旧的路径更新为新的models目录结构。

使用方法:
    python scripts/update_model_paths.py
"""

import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

# 配置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ModelPathUpdater:
    """模型路径更新器"""

    def __init__(self, project_root: str | None = None):
        self.project_root = Path(project_root) if project_root else Path.cwd()

        # 路径映射规则
        self.path_mappings = {
            # YOLO模型路径映射
            r"yolov8n\.pt": "models/yolo/models/yolo/yolov8n.pt",
            r"yolov8s\.pt": "models/yolo/models/yolo/yolov8s.pt",
            r"yolov8m\.pt": "models/yolo/models/yolo/yolov8m.pt",
            r"yolov8l\.pt": "models/yolo/models/yolo/yolov8l.pt",
            # 发网检测模型路径映射
            r"models/hairnet_detection\.pt": "models/hairnet_detection/models/hairnet_detection/hairnet_detection.pt",
            r"hairnet_detection\.pt": "models/hairnet_detection/models/hairnet_detection/hairnet_detection.pt",
            # 用户训练模型路径（保持不变，但需要确保引用正确）
            r"models/hairnet_model/weights/best\.pt": "models/hairnet_model/weights/best.pt",
            r"models/hairnet_model/weights/last\.pt": "models/hairnet_model/weights/last.pt",
            # 其他模型路径
            r"models/yolov8n\.pt": "models/yolo/models/yolo/yolov8n.pt",
            r"models/yolov8s\.pt": "models/yolo/models/yolo/yolov8s.pt",
            r"models/yolov8m\.pt": "models/yolo/models/yolo/yolov8m.pt",
            r"models/yolov8l\.pt": "models/yolo/models/yolo/yolov8l.pt",
        }

        # 需要更新的文件类型
        self.file_extensions = {".py", ".yaml", ".yml", ".md", ".sh", ".ps1", ".bat"}

        # 需要排除的目录
        self.exclude_dirs = {
            ".git",
            ".venv",
            "venv",
            "__pycache__",
            ".pytest_cache",
            "node_modules",
            ".tox",
            "logs",
            "temp",
        }

        # 更新统计
        self.update_stats = {
            "files_processed": 0,
            "files_updated": 0,
            "total_replacements": 0,
            "replacements_by_file": {},
        }

    def should_process_file(self, file_path: Path) -> bool:
        """判断是否应该处理该文件"""
        # 检查文件扩展名
        if file_path.suffix not in self.file_extensions:
            return False

        # 检查是否在排除目录中
        for exclude_dir in self.exclude_dirs:
            if exclude_dir in file_path.parts:
                return False

        return True

    def update_file_content(self, file_path: Path) -> bool:
        """更新单个文件的内容"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            original_content = content
            replacements_made = 0

            # 应用所有路径映射
            for old_pattern, new_path in self.path_mappings.items():
                # 使用正则表达式进行替换
                matches = re.findall(old_pattern, content)
                if matches:
                    content = re.sub(old_pattern, new_path, content)
                    replacements_made += len(matches)
                    logger.info(
                        f"在 {file_path} 中替换了 {len(matches)} 个 '{old_pattern}' -> '{new_path}'"
                    )

            # 如果有更改，写回文件
            if content != original_content:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)

                self.update_stats["files_updated"] += 1
                self.update_stats["total_replacements"] += replacements_made
                self.update_stats["replacements_by_file"][
                    str(file_path)
                ] = replacements_made

                logger.info(f"✅ 更新文件: {file_path} ({replacements_made} 个替换)")
                return True

            return False

        except Exception as e:
            logger.error(f"❌ 处理文件 {file_path} 时出错: {e}")
            return False

    def find_files_to_update(self) -> List[Path]:
        """查找需要更新的文件"""
        files_to_update = []

        for root, dirs, files in os.walk(self.project_root):
            # 排除不需要的目录
            dirs[:] = [d for d in dirs if d not in self.exclude_dirs]

            for file in files:
                file_path = Path(root) / file
                if self.should_process_file(file_path):
                    files_to_update.append(file_path)

        return files_to_update

    def update_all_paths(self) -> None:
        """更新所有模型路径"""
        logger.info("开始更新模型路径...")
        logger.info(f"项目根目录: {self.project_root}")

        # 查找需要更新的文件
        files_to_update = self.find_files_to_update()
        logger.info(f"找到 {len(files_to_update)} 个文件需要检查")

        # 更新每个文件
        for file_path in files_to_update:
            self.update_stats["files_processed"] += 1
            self.update_file_content(file_path)

        # 输出统计信息
        self.print_update_summary()

    def print_update_summary(self) -> None:
        """打印更新摘要"""
        logger.info("\n" + "=" * 60)
        logger.info("模型路径更新完成")
        logger.info("=" * 60)
        logger.info(f"处理文件数: {self.update_stats['files_processed']}")
        logger.info(f"更新文件数: {self.update_stats['files_updated']}")
        logger.info(f"总替换次数: {self.update_stats['total_replacements']}")

        if self.update_stats["replacements_by_file"]:
            logger.info("\n详细更新信息:")
            for file_path, count in self.update_stats["replacements_by_file"].items():
                logger.info(f"  {file_path}: {count} 个替换")

        logger.info("\n路径映射规则:")
        for old_pattern, new_path in self.path_mappings.items():
            logger.info(f"  {old_pattern} -> {new_path}")

    def validate_model_paths(self) -> None:
        """验证更新后的模型路径是否正确"""
        logger.info("\n验证模型路径...")

        # 检查关键模型文件是否存在
        key_models = [
            "models/yolo/models/yolo/yolov8n.pt",
            "models/yolo/models/yolo/yolov8s.pt",
            "models/yolo/models/yolo/yolov8m.pt",
            "models/yolo/models/yolo/yolov8l.pt",
            "models/hairnet_detection/models/hairnet_detection/hairnet_detection.pt",
            "models/hairnet_model/weights/best.pt",
            "models/hairnet_model/weights/last.pt",
        ]

        existing_models = []
        missing_models = []

        for model_path in key_models:
            full_path = self.project_root / model_path
            if full_path.exists():
                size_mb = full_path.stat().st_size / (1024 * 1024)
                existing_models.append(f"{model_path} ({size_mb:.1f} MB)")
            else:
                missing_models.append(model_path)

        if existing_models:
            logger.info("✅ 存在的模型文件:")
            for model in existing_models:
                logger.info(f"  {model}")

        if missing_models:
            logger.warning("⚠️  缺失的模型文件:")
            for model in missing_models:
                logger.warning(f"  {model}")
            logger.warning("请确保这些模型文件已正确移动到新位置")

    def create_update_report(self) -> None:
        """创建更新报告"""
        report_path = self.project_root / "reports" / "model_path_update_report.md"
        report_path.parent.mkdir(exist_ok=True)

        report_content = f"""# 模型路径更新报告

**生成时间**: {self.get_current_time()}
**操作类型**: 模型路径更新

## 更新摘要

- **处理文件数**: {self.update_stats['files_processed']}
- **更新文件数**: {self.update_stats['files_updated']}
- **总替换次数**: {self.update_stats['total_replacements']}

## 路径映射规则

| 旧路径模式 | 新路径 |
|-----------|--------|
"""

        for old_pattern, new_path in self.path_mappings.items():
            report_content += f"| `{old_pattern}` | `{new_path}` |\n"

        if self.update_stats["replacements_by_file"]:
            report_content += "\n## 详细更新信息\n\n"
            for file_path, count in self.update_stats["replacements_by_file"].items():
                relative_path = Path(file_path).relative_to(self.project_root)
                report_content += f"- `{relative_path}`: {count} 个替换\n"

        report_content += "\n## 注意事项\n\n"
        report_content += "1. 所有YOLO模型现在位于 `models/yolo/` 目录下\n"
        report_content += "2. 发网检测模型位于 `models/hairnet_detection/` 目录下\n"
        report_content += "3. 用户训练的模型仍在 `models/hairnet_model/weights/` 目录下（受保护）\n"
        report_content += "4. 请确保所有模型文件已正确移动到新位置\n"
        report_content += "5. 如有配置文件缓存，请清除后重新加载\n"

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_content)

        logger.info(f"\n📄 更新报告已保存: {report_path}")

    def get_current_time(self) -> str:
        """获取当前时间字符串"""
        from datetime import datetime

        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def main():
    """主函数"""
    try:
        # 创建更新器实例
        updater = ModelPathUpdater()

        # 执行路径更新
        updater.update_all_paths()

        # 验证模型路径
        updater.validate_model_paths()

        # 创建更新报告
        updater.create_update_report()

        logger.info("\n🎉 模型路径更新完成！")
        logger.info("请检查更新报告以确认所有更改都正确应用。")

    except Exception as e:
        logger.error(f"❌ 更新过程中出现错误: {e}")
        raise


if __name__ == "__main__":
    main()
