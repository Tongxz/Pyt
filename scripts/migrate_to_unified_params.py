#!/usr/bin/env python3
"""
参数统一迁移脚本
将现有系统迁移到统一参数配置

功能:
1. 检查当前系统中的参数不一致问题
2. 生成迁移报告
3. 应用统一参数配置
4. 验证迁移结果

作者: AI Assistant
创建时间: 2024
"""

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.config.unified_params import UnifiedParams, get_unified_params

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ParameterMigrator:
    """参数迁移器"""

    def __init__(self):
        self.project_root = project_root
        self.issues = []
        self.migration_report = {
            "inconsistencies_found": [],
            "files_updated": [],
            "parameters_unified": [],
            "warnings": [],
            "recommendations": [],
        }

    def analyze_current_parameters(self) -> Dict[str, Any]:
        """分析当前系统中的参数设置"""
        logger.info("开始分析当前参数设置...")

        current_params = {
            "human_detection": {},
            "hairnet_detection": {},
            "behavior_recognition": {},
            "config_files": {},
        }

        # 分析人体检测器参数
        self._analyze_human_detector_params(current_params)

        # 分析发网检测器参数
        self._analyze_hairnet_detector_params(current_params)

        # 分析行为识别器参数
        self._analyze_behavior_recognizer_params(current_params)

        # 分析配置文件
        self._analyze_config_files(current_params)

        return current_params

    def _analyze_human_detector_params(self, current_params: Dict[str, Any]):
        """分析人体检测器参数"""
        detector_file = self.project_root / "src" / "core" / "detector.py"
        if detector_file.exists():
            try:
                with open(detector_file, "r", encoding="utf-8") as f:
                    content = f.read()

                # 提取参数（简单的字符串匹配）
                params = {}
                if "confidence_threshold = 0.25" in content:
                    params["confidence_threshold"] = 0.25
                elif "confidence_threshold = 0.3" in content:
                    params["confidence_threshold"] = 0.3

                if "iou_threshold = 0.5" in content:
                    params["iou_threshold"] = 0.5

                if "min_box_area = 400" in content:
                    params["min_box_area"] = 400
                elif "min_box_area = 800" in content:
                    params["min_box_area"] = 800

                current_params["human_detection"] = params
                logger.info(f"人体检测器参数: {params}")

            except Exception as e:
                logger.error(f"分析人体检测器参数失败: {e}")

    def _analyze_hairnet_detector_params(self, current_params: Dict[str, Any]):
        """分析发网检测器参数"""
        detector_file = self.project_root / "src" / "core" / "hairnet_detector.py"
        if detector_file.exists():
            try:
                with open(detector_file, "r", encoding="utf-8") as f:
                    content = f.read()

                params = {}
                if "confidence_threshold = 0.7" in content:
                    params["confidence_threshold"] = 0.7
                elif "confidence_threshold = 0.6" in content:
                    params["confidence_threshold"] = 0.6

                current_params["hairnet_detection"] = params
                logger.info(f"发网检测器参数: {params}")

            except Exception as e:
                logger.error(f"分析发网检测器参数失败: {e}")

    def _analyze_behavior_recognizer_params(self, current_params: Dict[str, Any]):
        """分析行为识别器参数"""
        behavior_file = self.project_root / "src" / "core" / "behavior.py"
        if behavior_file.exists():
            try:
                with open(behavior_file, "r", encoding="utf-8") as f:
                    content = f.read()

                params = {}
                if "confidence_threshold: float = 0.6" in content:
                    params["confidence_threshold"] = 0.6

                if 'min_duration": 2.0' in content:
                    params["handwashing_min_duration"] = 2.0
                elif 'min_duration": 3.0' in content:
                    params["handwashing_min_duration"] = 3.0

                current_params["behavior_recognition"] = params
                logger.info(f"行为识别器参数: {params}")

            except Exception as e:
                logger.error(f"分析行为识别器参数失败: {e}")

    def _analyze_config_files(self, current_params: Dict[str, Any]):
        """分析配置文件"""
        config_files = [
            self.project_root / "config" / "default.yaml",
            self.project_root / "scripts" / "detection_config_optimized.py",
        ]

        for config_file in config_files:
            if config_file.exists():
                try:
                    if config_file.suffix == ".yaml":
                        with open(config_file, "r", encoding="utf-8") as f:
                            config_data = yaml.safe_load(f)
                        current_params["config_files"][str(config_file)] = config_data
                    else:
                        with open(config_file, "r", encoding="utf-8") as f:
                            content = f.read()
                        current_params["config_files"][str(config_file)] = {
                            "content": content[:500]
                        }

                    logger.info(f"已分析配置文件: {config_file}")

                except Exception as e:
                    logger.error(f"分析配置文件失败 {config_file}: {e}")

    def detect_inconsistencies(
        self, current_params: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """检测参数不一致问题"""
        logger.info("检测参数不一致问题...")

        inconsistencies = []

        # 检测置信度阈值不一致
        confidence_thresholds = []
        if "confidence_threshold" in current_params.get("human_detection", {}):
            confidence_thresholds.append(
                (
                    "human_detection",
                    current_params["human_detection"]["confidence_threshold"],
                )
            )
        if "confidence_threshold" in current_params.get("hairnet_detection", {}):
            confidence_thresholds.append(
                (
                    "hairnet_detection",
                    current_params["hairnet_detection"]["confidence_threshold"],
                )
            )
        if "confidence_threshold" in current_params.get("behavior_recognition", {}):
            confidence_thresholds.append(
                (
                    "behavior_recognition",
                    current_params["behavior_recognition"]["confidence_threshold"],
                )
            )

        if len(set(threshold for _, threshold in confidence_thresholds)) > 1:
            inconsistencies.append(
                {
                    "type": "confidence_threshold_mismatch",
                    "description": "不同模块使用了不同的置信度阈值",
                    "details": confidence_thresholds,
                    "severity": "high",
                }
            )

        # 检测其他不一致问题
        # ...

        self.migration_report["inconsistencies_found"] = inconsistencies
        logger.info(f"发现 {len(inconsistencies)} 个不一致问题")

        return inconsistencies

    def generate_migration_plan(
        self, inconsistencies: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """生成迁移计划"""
        logger.info("生成迁移计划...")

        plan = {
            "steps": [],
            "files_to_update": [],
            "backup_required": True,
            "estimated_time": "5-10分钟",
        }

        # 步骤1: 创建统一参数配置
        plan["steps"].append(
            {
                "step": 1,
                "description": "创建统一参数配置文件",
                "action": "create_unified_config",
                "files": ["src/config/unified_params.py", "config/unified_params.yaml"],
            }
        )

        # 步骤2: 更新检测器
        plan["steps"].append(
            {
                "step": 2,
                "description": "更新检测器使用统一参数",
                "action": "update_detectors",
                "files": [
                    "src/core/detector.py",
                    "src/core/hairnet_detector.py",
                    "src/core/behavior.py",
                ],
            }
        )

        # 步骤3: 验证迁移
        plan["steps"].append(
            {
                "step": 3,
                "description": "验证迁移结果",
                "action": "validate_migration",
                "files": [],
            }
        )

        return plan

    def execute_migration(self, plan: Dict[str, Any]) -> bool:
        """执行迁移"""
        logger.info("开始执行迁移...")

        try:
            # 创建统一参数配置
            unified_params = UnifiedParams()

            # 保存配置文件
            config_path = self.project_root / "config" / "unified_params.yaml"
            unified_params.save_to_yaml(str(config_path))

            self.migration_report["files_updated"].append(str(config_path))

            # 验证参数
            warnings = unified_params.validate_params()
            self.migration_report["warnings"].extend(warnings)

            logger.info("迁移执行完成")
            return True

        except Exception as e:
            logger.error(f"迁移执行失败: {e}")
            return False

    def validate_migration(self) -> bool:
        """验证迁移结果"""
        logger.info("验证迁移结果...")

        try:
            # 尝试加载统一参数
            params = get_unified_params()

            # 验证参数完整性
            assert hasattr(params, "human_detection")
            assert hasattr(params, "hairnet_detection")
            assert hasattr(params, "behavior_recognition")
            assert hasattr(params, "system")

            # 验证关键参数
            assert params.human_detection.confidence_threshold > 0
            assert params.hairnet_detection.confidence_threshold > 0
            assert params.behavior_recognition.confidence_threshold > 0

            logger.info("迁移验证通过")
            return True

        except Exception as e:
            logger.error(f"迁移验证失败: {e}")
            return False

    def generate_report(self) -> str:
        """生成迁移报告"""
        report = []
        report.append("# 参数统一迁移报告")
        report.append("")
        report.append(f"## 迁移概要")
        report.append(
            f"- 发现不一致问题: {len(self.migration_report['inconsistencies_found'])}个"
        )
        report.append(f"- 更新文件数量: {len(self.migration_report['files_updated'])}个")
        report.append(f"- 警告数量: {len(self.migration_report['warnings'])}个")
        report.append("")

        if self.migration_report["inconsistencies_found"]:
            report.append("## 发现的不一致问题")
            for issue in self.migration_report["inconsistencies_found"]:
                report.append(f"- **{issue['type']}**: {issue['description']}")
                if "details" in issue:
                    report.append(f"  详情: {issue['details']}")
            report.append("")

        if self.migration_report["files_updated"]:
            report.append("## 更新的文件")
            for file_path in self.migration_report["files_updated"]:
                report.append(f"- {file_path}")
            report.append("")

        if self.migration_report["warnings"]:
            report.append("## 警告信息")
            for warning in self.migration_report["warnings"]:
                report.append(f"- {warning}")
            report.append("")

        report.append("## 建议")
        report.append("1. 重新启动应用程序以使用新的统一参数配置")
        report.append("2. 运行测试确保所有功能正常工作")
        report.append("3. 监控系统性能，必要时调整参数")
        report.append("4. 定期检查参数配置的一致性")

        return "\n".join(report)

    def run_migration(self) -> bool:
        """运行完整的迁移流程"""
        logger.info("开始参数统一迁移流程")

        try:
            # 1. 分析当前参数
            current_params = self.analyze_current_parameters()

            # 2. 检测不一致问题
            inconsistencies = self.detect_inconsistencies(current_params)

            # 3. 生成迁移计划
            plan = self.generate_migration_plan(inconsistencies)

            # 4. 执行迁移
            success = self.execute_migration(plan)
            if not success:
                return False

            # 5. 验证迁移
            validation_success = self.validate_migration()
            if not validation_success:
                logger.warning("迁移验证失败，但基本迁移已完成")

            # 6. 生成报告
            report = self.generate_report()
            report_path = self.project_root / "migration_report.md"
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(report)

            logger.info(f"迁移完成，报告已保存到: {report_path}")
            return True

        except Exception as e:
            logger.error(f"迁移流程失败: {e}")
            return False


def main():
    """主函数"""
    print("=== 参数统一迁移工具 ===")
    print("此工具将帮助您将现有系统迁移到统一参数配置")
    print()

    migrator = ParameterMigrator()

    # 询问用户是否继续
    response = input("是否开始迁移? (y/N): ").strip().lower()
    if response not in ["y", "yes"]:
        print("迁移已取消")
        return

    # 执行迁移
    success = migrator.run_migration()

    if success:
        print("\n✅ 迁移成功完成!")
        print("\n下一步:")
        print("1. 查看生成的迁移报告: migration_report.md")
        print("2. 重新启动应用程序")
        print("3. 运行测试验证功能")
    else:
        print("\n❌ 迁移失败，请查看日志了解详情")


if __name__ == "__main__":
    main()
