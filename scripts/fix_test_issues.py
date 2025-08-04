#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复测试文件中的问题

修复测试文件中的接口变化问题
"""

import os
import re
from pathlib import Path

# 获取项目根目录
ROOT_DIR = Path(__file__).parent.parent

# 测试目录
TESTS_DIR = ROOT_DIR / "tests"

# 修复HairnetDetector类的测试
def fix_hairnet_detector_tests():
    """修复HairnetDetector类的测试"""
    file_path = TESTS_DIR / "unit" / "test_hairnet_detector.py"
    
    # 读取文件内容
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # 修复方法名和属性名
    replacements = [
        ("test_extract_head_region_keypoints_bbox", "test__extract_head_roi_from_bbox_bbox"),
        ("test_extract_head_region_keypoints_keypoints", "test__optimize_head_roi_with_keypoints"),
        ("testpreprocess_image", "test__preprocess_image"),
        ("test_calculate_compliance_rate", "testcalculate_compliance_rate"),
        ("test_visualize_hairnet_detections", "test_visualize_detections"),
        ("test_get_statistics", "test_get_detection_statistics"),
        ("'has_hairnet'", "'wearing_hairnet'"),
        ("self.pipeline.detector", "self.pipeline.person_detector"),
        ("self.detector.extract_head_region_keypoints", "self.detector._extract_head_roi_from_bbox"),
        ("self.detector.preprocess_image", "self.detector._preprocess_image"),
        ("self.pipeline._calculate_compliance_rate", "self.pipeline.calculate_compliance_rate"),
        ("self.pipeline.visualize_hairnet_detections", "self.pipeline.visualize_detections"),
        ("self.pipeline.get_statistics", "self.pipeline.get_detection_statistics"),
    ]
    
    for old, new in replacements:
        content = content.replace(old, new)
    
    # 修复HairnetDetectionPipeline初始化参数
    content = re.sub(
        r"self\.pipeline = HairnetDetectionPipeline\(\)", 
        "self.pipeline = HairnetDetectionPipeline(MockPersonDetector(), MockHairnetDetector())", 
        content
    )
    
    # 修复测试方法中的问题
    # 1. 修复test__extract_head_roi_from_bbox_keypoints方法
    content = re.sub(
        r"head_region = self\.detector\._extract_head_roi_from_bbox\(self\.test_image, keypoints=keypoints\)", 
        "head_region = self.detector._optimize_head_roi_with_keypoints(self.test_image, None, keypoints)", 
        content
    )
    
    # 2. 修复confidence_threshold测试
    content = re.sub(
        r"self\.assertGreater\(self\.detector\.confidence_threshold, confidence\)", 
        "self.assertLess(confidence, self.detector.confidence_threshold)", 
        content
    )
    
    content = re.sub(
        r"self\.assertLess\(self\.detector\.confidence_threshold, confidence\)", 
        "self.assertGreater(confidence, self.detector.confidence_threshold)", 
        content
    )
    
    # 添加模拟检测器类
    mock_classes = """
# 模拟检测器类
class MockPersonDetector:
    def detect(self, frame):
        return [{'bbox': [0, 0, 100, 100], 'confidence': 0.9}]

class MockHairnetDetector:
    def _detect_hairnet_with_pytorch(self, frame):
        return {'wearing_hairnet': True, 'has_hairnet': True, 'confidence': 0.85}
"""
    
    # 替换现有的模拟类
    content = re.sub(
        r"# 模拟检测器类[\s\S]*?class MockHairnetDetector:[\s\S]*?\n\n", 
        mock_classes + "\n", 
        content
    )
    
    # 写回文件
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    
    print(f"已修复: {file_path}")

# 修复dual_channel_hairnet测试
def fix_dual_channel_hairnet_tests():
    """修复dual_channel_hairnet测试"""
    file_path = TESTS_DIR / "unit" / "test_dual_channel_hairnet.py"
    
    if not file_path.exists():
        print(f"文件不存在: {file_path}")
        return
    
    # 读取文件内容
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # 修复HairnetDetectionPipeline初始化
    content = re.sub(
        r"pipeline = HairnetDetectionPipeline\(\)", 
        "pipeline = HairnetDetectionPipeline(MockPersonDetector(), MockHairnetDetector())", 
        content
    )
    
    # 添加模拟检测器类
    if "class MockPersonDetector" not in content:
        mock_classes = """
# 模拟检测器类
class MockPersonDetector:
    def detect(self, frame):
        return [{'bbox': [0, 0, 100, 100], 'confidence': 0.9}]

class MockHairnetDetector:
    def _detect_hairnet_with_pytorch(self, frame):
        return {'wearing_hairnet': True, 'has_hairnet': True, 'confidence': 0.85}
"""
        # 在导入语句后添加模拟类
        imports_end = re.search(r"import.*?\n\n", content, re.DOTALL)
        if imports_end:
            content = content.replace(
                imports_end.group(0), 
                imports_end.group(0) + mock_classes
            )
    
    # 写回文件
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    
    print(f"已修复: {file_path}")

# 修复所有测试文件中的问题
def fix_all_test_issues():
    """修复所有测试文件中的问题"""
    # 修复HairnetDetector类的测试
    fix_hairnet_detector_tests()
    
    # 修复dual_channel_hairnet测试
    fix_dual_channel_hairnet_tests()
    
    # 可以添加更多修复函数

# 主函数
def main():
    """主函数"""
    print("=== 修复测试文件中的问题 ===")
    
    # 修复所有测试文件中的问题
    fix_all_test_issues()
    
    print("\n测试文件问题修复完成！")

if __name__ == "__main__":
    main()