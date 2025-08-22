#!/usr/bin/env python3
"""
测试用例分析脚本

分析项目中所有测试用例的分布和功能
"""

import subprocess
import re
from collections import defaultdict, Counter
from pathlib import Path

# 获取项目根目录
ROOT_DIR = Path(__file__).parent.parent

def get_all_test_cases():
    """
    获取所有测试用例
    """
    try:
        # 运行pytest --collect-only获取所有测试用例
        result = subprocess.run(
            ['pytest', '--collect-only', '-q'],
            cwd=ROOT_DIR,
            capture_output=True,
            text=True,
            encoding='utf-8'
        )
        
        if result.returncode != 0:
            print(f"Error running pytest: {result.stderr}")
            return []
        
        # 解析输出，提取测试用例
        test_cases = []
        for line in result.stdout.split('\n'):
            if '::' in line and not line.startswith('='):
                test_cases.append(line.strip())
        
        return test_cases
    
    except Exception as e:
        print(f"Error getting test cases: {e}")
        return []

def analyze_test_cases(test_cases):
    """
    分析测试用例
    """
    analysis = {
        'total_count': len(test_cases),
        'by_file': defaultdict(list),
        'by_class': defaultdict(list),
        'by_category': defaultdict(list),
        'by_functionality': defaultdict(list)
    }
    
    for test_case in test_cases:
        # 解析测试用例路径
        parts = test_case.split('::')
        if len(parts) >= 2:
            file_path = parts[0]
            test_name = parts[-1]
            
            # 按文件分类
            analysis['by_file'][file_path].append(test_case)
            
            # 按类分类（如果有的话）
            if len(parts) >= 3:
                class_name = parts[1]
                analysis['by_class'][class_name].append(test_case)
            
            # 按功能分类
            functionality = categorize_by_functionality(file_path, test_name)
            analysis['by_functionality'][functionality].append(test_case)
            
            # 按测试类型分类
            category = categorize_by_type(file_path)
            analysis['by_category'][category].append(test_case)
    
    return analysis

def categorize_by_functionality(file_path, test_name):
    """
    根据文件路径和测试名称推断功能类别
    """
    file_path_lower = file_path.lower()
    test_name_lower = test_name.lower()
    
    # API相关测试
    if 'api' in file_path_lower or 'endpoint' in test_name_lower:
        return 'API接口测试'
    
    # 检测器相关测试
    if 'detector' in file_path_lower:
        if 'hairnet' in file_path_lower:
            return '发网检测测试'
        elif 'pose' in file_path_lower:
            return '姿态检测测试'
        else:
            return '通用检测测试'
    
    # 行为识别测试
    if 'behavior' in file_path_lower or 'handwash' in test_name_lower:
        return '行为识别测试'
    
    # 运动分析测试
    if 'motion' in file_path_lower:
        return '运动分析测试'
    
    # 数学工具测试
    if 'math' in file_path_lower:
        return '数学工具测试'
    
    # 数据管理测试
    if 'data_manager' in file_path_lower:
        return '数据管理测试'
    
    # GPU配置测试
    if 'gpu' in file_path_lower or 'mediapipe' in file_path_lower:
        return 'GPU配置测试'
    
    # 阈值调整测试
    if 'threshold' in file_path_lower:
        return '阈值调整测试'
    
    return '其他测试'

def categorize_by_type(file_path):
    """
    根据文件路径推断测试类型
    """
    if 'unit' in file_path:
        return '单元测试'
    elif 'integration' in file_path:
        return '集成测试'
    elif file_path.startswith('test_'):
        return '根目录测试'
    else:
        return '其他测试'

def print_analysis(analysis):
    """
    打印分析结果
    """
    print("\n" + "="*60)
    print("🧪 测试用例分析报告")
    print("="*60)
    
    print(f"\n📊 总体统计")
    print(f"测试用例总数: {analysis['total_count']}")
    
    # 按测试类型分类
    print(f"\n📁 按测试类型分类:")
    for category, tests in sorted(analysis['by_category'].items()):
        print(f"  {category}: {len(tests)}个")
    
    # 按功能分类
    print(f"\n🔧 按功能模块分类:")
    for functionality, tests in sorted(analysis['by_functionality'].items(), key=lambda x: len(x[1]), reverse=True):
        print(f"  {functionality}: {len(tests)}个")
    
    # 按文件分类
    print(f"\n📄 按测试文件分类:")
    for file_path, tests in sorted(analysis['by_file'].items(), key=lambda x: len(x[1]), reverse=True):
        file_name = Path(file_path).name
        print(f"  {file_name}: {len(tests)}个测试")
    
    # 按测试类分类
    print(f"\n🏷️  按测试类分类:")
    class_counts = Counter()
    for class_name, tests in analysis['by_class'].items():
        class_counts[class_name] = len(tests)
    
    for class_name, count in class_counts.most_common(10):  # 显示前10个最多的类
        print(f"  {class_name}: {count}个测试")
    
    # 详细功能分析
    print(f"\n🔍 详细功能分析:")
    for functionality, tests in sorted(analysis['by_functionality'].items()):
        print(f"\n  📌 {functionality} ({len(tests)}个):")
        
        # 按文件分组显示
        file_groups = defaultdict(list)
        for test in tests:
            file_path = test.split('::')[0]
            file_name = Path(file_path).name
            file_groups[file_name].append(test)
        
        for file_name, file_tests in sorted(file_groups.items()):
            print(f"    📄 {file_name}: {len(file_tests)}个")
            
            # 显示具体的测试方法（限制显示数量）
            for test in file_tests[:3]:  # 只显示前3个
                test_method = test.split('::')[-1]
                print(f"      - {test_method}")
            
            if len(file_tests) > 3:
                print(f"      - ... 还有{len(file_tests) - 3}个测试")

def generate_test_summary():
    """
    生成测试用例总结
    """
    print("正在收集测试用例信息...")
    test_cases = get_all_test_cases()
    
    if not test_cases:
        print("未找到测试用例")
        return
    
    print(f"找到 {len(test_cases)} 个测试用例")
    
    # 分析测试用例
    analysis = analyze_test_cases(test_cases)
    
    # 打印分析结果
    print_analysis(analysis)
    
    # 生成测试覆盖建议
    print(f"\n💡 测试覆盖建议:")
    
    functionality_counts = {k: len(v) for k, v in analysis['by_functionality'].items()}
    
    # 找出测试较少的功能模块
    low_coverage = [func for func, count in functionality_counts.items() if count < 5]
    if low_coverage:
        print(f"  以下功能模块测试用例较少，建议增加测试:")
        for func in low_coverage:
            print(f"    - {func}: {functionality_counts[func]}个测试")
    
    # 测试分布建议
    unit_tests = len(analysis['by_category'].get('单元测试', []))
    integration_tests = len(analysis['by_category'].get('集成测试', []))
    
    print(f"\n  测试类型分布:")
    print(f"    - 单元测试: {unit_tests}个 ({unit_tests/analysis['total_count']*100:.1f}%)")
    print(f"    - 集成测试: {integration_tests}个 ({integration_tests/analysis['total_count']*100:.1f}%)")
    
    if unit_tests / analysis['total_count'] < 0.7:
        print(f"  ⚠️  建议增加更多单元测试，目前单元测试占比较低")
    
    if integration_tests / analysis['total_count'] < 0.2:
        print(f"  ⚠️  建议增加更多集成测试，确保模块间交互正常")

if __name__ == "__main__":
    generate_test_summary()