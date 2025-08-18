#!/usr/bin/env python3
"""
参数优化验证脚本
验证统一参数配置的效果和一致性

功能:
1. 验证所有模块使用统一参数配置
2. 检查参数一致性
3. 性能基准测试
4. 生成优化报告

作者: AI Assistant
创建时间: 2024
"""

import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.config.unified_params import get_unified_params
from src.core.behavior import BehaviorRecognizer
from src.core.detector import HumanDetector
from src.core.hairnet_detector import HairnetDetector

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ParameterOptimizationValidator:
    """参数优化验证器"""

    def __init__(self):
        self.unified_params = get_unified_params()
        self.validation_results = {
            "parameter_consistency": {},
            "module_initialization": {},
            "performance_metrics": {},
            "recommendations": [],
        }

    def validate_parameter_consistency(self) -> Dict[str, Any]:
        """验证参数一致性"""
        logger.info("验证参数一致性...")

        consistency_results = {
            "unified_config_loaded": False,
            "parameter_values": {},
            "consistency_issues": [],
        }

        try:
            # 验证统一配置是否正确加载
            params = self.unified_params
            consistency_results["unified_config_loaded"] = True

            # 记录关键参数值
            consistency_results["parameter_values"] = {
                "human_detection_confidence": params.human_detection.confidence_threshold,
                "human_detection_iou": params.human_detection.iou_threshold,
                "human_detection_min_area": params.human_detection.min_box_area,
                "hairnet_detection_confidence": params.hairnet_detection.confidence_threshold,
                "hairnet_edge_density_threshold": params.hairnet_detection.edge_density_threshold,
                "behavior_recognition_confidence": params.behavior_recognition.confidence_threshold,
                "handwashing_min_duration": params.behavior_recognition.handwashing_min_duration,
                "hairnet_min_duration": params.behavior_recognition.hairnet_min_duration,
            }

            # 检查参数合理性
            if (
                params.human_detection.confidence_threshold <= 0
                or params.human_detection.confidence_threshold >= 1
            ):
                consistency_results["consistency_issues"].append("人体检测置信度阈值不在合理范围内")

            if (
                params.hairnet_detection.confidence_threshold <= 0
                or params.hairnet_detection.confidence_threshold >= 1
            ):
                consistency_results["consistency_issues"].append("发网检测置信度阈值不在合理范围内")

            if (
                params.behavior_recognition.confidence_threshold <= 0
                or params.behavior_recognition.confidence_threshold >= 1
            ):
                consistency_results["consistency_issues"].append("行为识别置信度阈值不在合理范围内")

            if params.behavior_recognition.handwashing_min_duration <= 0:
                consistency_results["consistency_issues"].append("洗手最小持续时间设置不合理")

            logger.info(
                f"参数一致性验证完成，发现 {len(consistency_results['consistency_issues'])} 个问题"
            )

        except Exception as e:
            logger.error(f"参数一致性验证失败: {e}")
            consistency_results["unified_config_loaded"] = False

        self.validation_results["parameter_consistency"] = consistency_results
        return consistency_results

    def validate_module_initialization(self) -> Dict[str, Any]:
        """验证模块初始化"""
        logger.info("验证模块初始化...")

        initialization_results = {
            "human_detector": {"success": False, "error": None, "params_used": {}},
            "hairnet_detector": {"success": False, "error": None, "params_used": {}},
            "behavior_recognizer": {"success": False, "error": None, "params_used": {}},
        }

        # 测试人体检测器初始化
        try:
            detector = HumanDetector()
            initialization_results["human_detector"]["success"] = True
            initialization_results["human_detector"]["params_used"] = {
                "confidence_threshold": detector.confidence_threshold,
                "iou_threshold": detector.iou_threshold,
                "min_box_area": detector.min_box_area,
            }
            logger.info("人体检测器初始化成功")
        except Exception as e:
            initialization_results["human_detector"]["error"] = str(e)
            logger.error(f"人体检测器初始化失败: {e}")

        # 测试发网检测器初始化
        try:
            hairnet_detector = HairnetDetector()
            initialization_results["hairnet_detector"]["success"] = True
            initialization_results["hairnet_detector"]["params_used"] = {
                "confidence_threshold": hairnet_detector.confidence_threshold
            }
            logger.info("发网检测器初始化成功")
        except Exception as e:
            initialization_results["hairnet_detector"]["error"] = str(e)
            logger.error(f"发网检测器初始化失败: {e}")

        # 测试行为识别器初始化
        try:
            behavior_recognizer = BehaviorRecognizer()
            initialization_results["behavior_recognizer"]["success"] = True
            initialization_results["behavior_recognizer"]["params_used"] = {
                "confidence_threshold": behavior_recognizer.confidence_threshold
            }
            logger.info("行为识别器初始化成功")
        except Exception as e:
            initialization_results["behavior_recognizer"]["error"] = str(e)
            logger.error(f"行为识别器初始化失败: {e}")

        self.validation_results["module_initialization"] = initialization_results
        return initialization_results

    def run_performance_benchmark(self) -> Dict[str, Any]:
        """运行性能基准测试"""
        logger.info("运行性能基准测试...")

        performance_results = {
            "initialization_time": {},
            "memory_usage": {},
            "parameter_access_time": {},
        }

        # 测试初始化时间
        try:
            # 人体检测器初始化时间
            start_time = time.time()
            detector = HumanDetector()
            init_time = time.time() - start_time
            performance_results["initialization_time"]["human_detector"] = init_time

            # 发网检测器初始化时间
            start_time = time.time()
            hairnet_detector = HairnetDetector()
            init_time = time.time() - start_time
            performance_results["initialization_time"]["hairnet_detector"] = init_time

            # 行为识别器初始化时间
            start_time = time.time()
            behavior_recognizer = BehaviorRecognizer()
            init_time = time.time() - start_time
            performance_results["initialization_time"][
                "behavior_recognizer"
            ] = init_time

            logger.info("性能基准测试完成")

        except Exception as e:
            logger.error(f"性能基准测试失败: {e}")

        # 测试参数访问时间
        try:
            start_time = time.time()
            for _ in range(1000):
                params = get_unified_params()
                _ = params.human_detection.confidence_threshold
                _ = params.hairnet_detection.confidence_threshold
                _ = params.behavior_recognition.confidence_threshold
            access_time = (time.time() - start_time) / 1000
            performance_results["parameter_access_time"]["avg_per_access"] = access_time

        except Exception as e:
            logger.error(f"参数访问时间测试失败: {e}")

        self.validation_results["performance_metrics"] = performance_results
        return performance_results

    def generate_recommendations(self) -> List[str]:
        """生成优化建议"""
        recommendations = []

        # 基于验证结果生成建议
        consistency = self.validation_results.get("parameter_consistency", {})
        initialization = self.validation_results.get("module_initialization", {})
        performance = self.validation_results.get("performance_metrics", {})

        if not consistency.get("unified_config_loaded", False):
            recommendations.append("统一参数配置加载失败，需要检查配置文件")

        if consistency.get("consistency_issues"):
            recommendations.append(
                f"发现 {len(consistency['consistency_issues'])} 个参数一致性问题，需要调整参数值"
            )

        # 检查模块初始化成功率
        failed_modules = []
        for module, result in initialization.items():
            if not result.get("success", False):
                failed_modules.append(module)

        if failed_modules:
            recommendations.append(f"以下模块初始化失败: {', '.join(failed_modules)}，需要检查依赖和配置")

        # 性能建议
        init_times = performance.get("initialization_time", {})
        if init_times:
            max_init_time = max(init_times.values())
            if max_init_time > 5.0:  # 超过5秒
                recommendations.append("模块初始化时间较长，考虑优化模型加载或使用延迟加载")

        # 参数优化建议
        param_values = consistency.get("parameter_values", {})
        if param_values:
            human_conf = param_values.get("human_detection_confidence", 0)
            hairnet_conf = param_values.get("hairnet_detection_confidence", 0)

            if abs(human_conf - hairnet_conf) > 0.3:
                recommendations.append("人体检测和发网检测的置信度阈值差异较大，建议调整以保持一致性")

            if human_conf < 0.2:
                recommendations.append("人体检测置信度阈值较低，可能导致误检，建议适当提高")

            if hairnet_conf > 0.8:
                recommendations.append("发网检测置信度阈值较高，可能导致漏检，建议适当降低")

        if not recommendations:
            recommendations.append("参数优化验证通过，系统配置良好")

        self.validation_results["recommendations"] = recommendations
        return recommendations

    def generate_report(self) -> str:
        """生成验证报告"""
        report = []
        report.append("# 参数优化验证报告")
        report.append("")
        report.append(f"## 验证概要")

        # 参数一致性
        consistency = self.validation_results.get("parameter_consistency", {})
        report.append(
            f"- 统一配置加载: {'✅ 成功' if consistency.get('unified_config_loaded') else '❌ 失败'}"
        )
        report.append(f"- 一致性问题: {len(consistency.get('consistency_issues', []))}个")

        # 模块初始化
        initialization = self.validation_results.get("module_initialization", {})
        success_count = sum(
            1 for result in initialization.values() if result.get("success", False)
        )
        total_count = len(initialization)
        report.append(f"- 模块初始化成功率: {success_count}/{total_count}")

        report.append("")

        # 详细参数值
        if consistency.get("parameter_values"):
            report.append("## 当前参数配置")
            for param_name, param_value in consistency["parameter_values"].items():
                report.append(f"- {param_name}: {param_value}")
            report.append("")

        # 一致性问题
        if consistency.get("consistency_issues"):
            report.append("## 发现的问题")
            for issue in consistency["consistency_issues"]:
                report.append(f"- ❌ {issue}")
            report.append("")

        # 模块初始化详情
        report.append("## 模块初始化状态")
        for module, result in initialization.items():
            status = (
                "✅ 成功"
                if result.get("success")
                else f"❌ 失败: {result.get('error', '未知错误')}"
            )
            report.append(f"- {module}: {status}")
        report.append("")

        # 性能指标
        performance = self.validation_results.get("performance_metrics", {})
        if performance.get("initialization_time"):
            report.append("## 性能指标")
            report.append("### 初始化时间")
            for module, time_taken in performance["initialization_time"].items():
                report.append(f"- {module}: {time_taken:.3f}秒")

            if performance.get("parameter_access_time", {}).get("avg_per_access"):
                avg_access_time = performance["parameter_access_time"]["avg_per_access"]
                report.append(f"- 参数访问平均时间: {avg_access_time*1000:.3f}毫秒")
            report.append("")

        # 建议
        recommendations = self.validation_results.get("recommendations", [])
        if recommendations:
            report.append("## 优化建议")
            for i, recommendation in enumerate(recommendations, 1):
                report.append(f"{i}. {recommendation}")
            report.append("")

        report.append("## 总结")
        if consistency.get("unified_config_loaded") and success_count == total_count:
            report.append("✅ 参数统一优化成功，所有模块正常工作")
        else:
            report.append("⚠️ 参数统一优化部分成功，存在需要解决的问题")

        return "\n".join(report)

    def run_full_validation(self) -> bool:
        """运行完整验证流程"""
        logger.info("开始参数优化验证")

        try:
            # 1. 验证参数一致性
            self.validate_parameter_consistency()

            # 2. 验证模块初始化
            self.validate_module_initialization()

            # 3. 运行性能基准测试
            self.run_performance_benchmark()

            # 4. 生成建议
            self.generate_recommendations()

            # 5. 生成报告
            report = self.generate_report()
            report_path = project_root / "parameter_optimization_validation_report.md"
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(report)

            logger.info(f"验证完成，报告已保存到: {report_path}")
            return True

        except Exception as e:
            logger.error(f"验证流程失败: {e}")
            return False


def main():
    """主函数"""
    print("=== 参数优化验证工具 ===")
    print("此工具将验证统一参数配置的效果和一致性")
    print()

    validator = ParameterOptimizationValidator()

    # 运行验证
    success = validator.run_full_validation()

    if success:
        print("\n✅ 验证完成!")
        print("\n查看详细报告: parameter_optimization_validation_report.md")

        # 显示关键结果
        consistency = validator.validation_results.get("parameter_consistency", {})
        if consistency.get("unified_config_loaded"):
            print("\n📊 当前参数配置:")
            for param, value in consistency.get("parameter_values", {}).items():
                print(f"  - {param}: {value}")

        recommendations = validator.validation_results.get("recommendations", [])
        if recommendations:
            print("\n💡 主要建议:")
            for rec in recommendations[:3]:  # 显示前3个建议
                print(f"  - {rec}")
    else:
        print("\n❌ 验证失败，请查看日志了解详情")


if __name__ == "__main__":
    main()
