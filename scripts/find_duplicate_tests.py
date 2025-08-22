#!/usr/bin/env python3
"""
重复测试检测脚本

分析项目中功能重复的测试用例
"""

import subprocess
import re
from collections import defaultdict, Counter
from pathlib import Path
from typing import List, Dict, Set, Tuple
import ast
import inspect

# 获取项目根目录
ROOT_DIR = Path(__file__).parent.parent

def extract_test_functions_from_file(file_path: Path) -> List[Dict]:
    """
    从测试文件中提取测试函数信息
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 解析AST
        tree = ast.parse(content)
        
        test_functions = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name.startswith('test_'):
                # 提取函数信息
                func_info = {
                    'name': node.name,
                    'file': str(file_path),
                    'line': node.lineno,
                    'docstring': ast.get_docstring(node) or '',
                    'args': [arg.arg for arg in node.args.args],
                    'body_lines': node.end_lineno - node.lineno if hasattr(node, 'end_lineno') else 0
                }
                
                # 提取函数体中的关键词
                func_info['keywords'] = extract_keywords_from_node(node)
                
                test_functions.append(func_info)
        
        return test_functions
    
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return []

def extract_keywords_from_node(node: ast.FunctionDef) -> Set[str]:
    """
    从AST节点中提取关键词
    """
    keywords = set()
    
    for child in ast.walk(node):
        # 提取函数调用
        if isinstance(child, ast.Call):
            if isinstance(child.func, ast.Name):
                keywords.add(child.func.id)
            elif isinstance(child.func, ast.Attribute):
                keywords.add(child.func.attr)
        
        # 提取变量名
        elif isinstance(child, ast.Name):
            keywords.add(child.id)
        
        # 提取字符串常量
        elif isinstance(child, ast.Constant) and isinstance(child.value, str):
            # 只保留有意义的字符串（长度>3且不是路径）
            if len(child.value) > 3 and '/' not in child.value and '\\' not in child.value:
                keywords.add(child.value.lower())
    
    return keywords

def normalize_test_name(test_name: str) -> str:
    """
    标准化测试名称，用于比较相似性
    """
    # 移除test_前缀
    name = test_name.replace('test_', '')
    
    # 移除常见的变体词汇
    variations = {
        'with_': '',
        'without_': '',
        'when_': '',
        'should_': '',
        'can_': '',
        'will_': '',
        'invalid_': 'bad_',
        'valid_': 'good_',
        'successful_': 'success_',
        'failed_': 'failure_',
        'empty_': 'none_',
        'null_': 'none_',
    }
    
    for old, new in variations.items():
        name = name.replace(old, new)
    
    return name

def calculate_similarity(func1: Dict, func2: Dict) -> float:
    """
    计算两个测试函数的相似度
    """
    # 名称相似度
    name1 = normalize_test_name(func1['name'])
    name2 = normalize_test_name(func2['name'])
    
    name_similarity = calculate_string_similarity(name1, name2)
    
    # 关键词相似度
    keywords1 = func1['keywords']
    keywords2 = func2['keywords']
    
    if not keywords1 and not keywords2:
        keyword_similarity = 0.0
    elif not keywords1 or not keywords2:
        keyword_similarity = 0.0
    else:
        common_keywords = keywords1.intersection(keywords2)
        total_keywords = keywords1.union(keywords2)
        keyword_similarity = len(common_keywords) / len(total_keywords) if total_keywords else 0.0
    
    # 文档字符串相似度
    doc1 = func1['docstring'].lower()
    doc2 = func2['docstring'].lower()
    doc_similarity = calculate_string_similarity(doc1, doc2) if doc1 and doc2 else 0.0
    
    # 参数相似度
    args1 = set(func1['args'])
    args2 = set(func2['args'])
    
    if not args1 and not args2:
        args_similarity = 1.0
    elif not args1 or not args2:
        args_similarity = 0.0
    else:
        common_args = args1.intersection(args2)
        total_args = args1.union(args2)
        args_similarity = len(common_args) / len(total_args) if total_args else 0.0
    
    # 加权计算总相似度
    total_similarity = (
        name_similarity * 0.4 +
        keyword_similarity * 0.3 +
        doc_similarity * 0.2 +
        args_similarity * 0.1
    )
    
    return total_similarity

def calculate_string_similarity(s1: str, s2: str) -> float:
    """
    计算两个字符串的相似度（简单的Jaccard相似度）
    """
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    
    # 分词
    words1 = set(re.findall(r'\w+', s1.lower()))
    words2 = set(re.findall(r'\w+', s2.lower()))
    
    if not words1 and not words2:
        return 1.0
    if not words1 or not words2:
        return 0.0
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    return len(intersection) / len(union) if union else 0.0

def find_duplicate_tests() -> Dict:
    """
    查找重复的测试用例
    """
    print("正在扫描测试文件...")
    
    # 收集所有测试文件
    test_files = []
    
    # 扫描tests目录
    tests_dir = ROOT_DIR / 'tests'
    if tests_dir.exists():
        test_files.extend(tests_dir.rglob('test_*.py'))
    
    # 扫描根目录的测试文件
    test_files.extend(ROOT_DIR.glob('test_*.py'))
    
    print(f"找到 {len(test_files)} 个测试文件")
    
    # 提取所有测试函数
    all_test_functions = []
    
    for test_file in test_files:
        functions = extract_test_functions_from_file(test_file)
        all_test_functions.extend(functions)
    
    print(f"找到 {len(all_test_functions)} 个测试函数")
    
    # 查找重复
    duplicates = []
    similarity_threshold = 0.7  # 相似度阈值
    
    for i, func1 in enumerate(all_test_functions):
        for j, func2 in enumerate(all_test_functions[i+1:], i+1):
            # 跳过同一个文件中的函数（可能是合理的重载）
            if func1['file'] == func2['file']:
                continue
            
            similarity = calculate_similarity(func1, func2)
            
            if similarity >= similarity_threshold:
                duplicates.append({
                    'func1': func1,
                    'func2': func2,
                    'similarity': similarity
                })
    
    # 按相似度排序
    duplicates.sort(key=lambda x: x['similarity'], reverse=True)
    
    # 分析结果
    analysis = {
        'total_functions': len(all_test_functions),
        'total_files': len(test_files),
        'duplicates': duplicates,
        'duplicate_count': len(duplicates),
        'functions_by_file': defaultdict(list)
    }
    
    # 按文件分组
    for func in all_test_functions:
        file_name = Path(func['file']).name
        analysis['functions_by_file'][file_name].append(func)
    
    return analysis

def print_duplicate_analysis(analysis: Dict):
    """
    打印重复测试分析结果
    """
    print("\n" + "="*60)
    print("🔍 重复测试检测报告")
    print("="*60)
    
    print(f"\n📊 总体统计:")
    print(f"测试文件总数: {analysis['total_files']}")
    print(f"测试函数总数: {analysis['total_functions']}")
    print(f"疑似重复测试对: {analysis['duplicate_count']}")
    
    if analysis['duplicate_count'] == 0:
        print("\n✅ 未发现明显的重复测试！")
        return
    
    print(f"\n🚨 发现的重复测试 (相似度 ≥ 70%):")
    
    for i, dup in enumerate(analysis['duplicates'], 1):
        func1 = dup['func1']
        func2 = dup['func2']
        similarity = dup['similarity']
        
        print(f"\n  {i}. 相似度: {similarity:.1%}")
        print(f"     📄 {Path(func1['file']).name}:{func1['line']} - {func1['name']}")
        print(f"     📄 {Path(func2['file']).name}:{func2['line']} - {func2['name']}")
        
        # 显示文档字符串（如果有）
        if func1['docstring']:
            print(f"        描述1: {func1['docstring'][:50]}...")
        if func2['docstring']:
            print(f"        描述2: {func2['docstring'][:50]}...")
        
        # 显示共同关键词
        common_keywords = func1['keywords'].intersection(func2['keywords'])
        if common_keywords:
            keywords_str = ', '.join(sorted(list(common_keywords))[:5])
            print(f"        共同关键词: {keywords_str}")
    
    # 按文件分析重复情况
    print(f"\n📁 按文件分析重复情况:")
    
    file_duplicate_count = defaultdict(int)
    for dup in analysis['duplicates']:
        file1 = Path(dup['func1']['file']).name
        file2 = Path(dup['func2']['file']).name
        file_duplicate_count[file1] += 1
        file_duplicate_count[file2] += 1
    
    for file_name, count in sorted(file_duplicate_count.items(), key=lambda x: x[1], reverse=True):
        print(f"  {file_name}: {count} 个重复测试")
    
    # 提供建议
    print(f"\n💡 优化建议:")
    
    high_similarity = [d for d in analysis['duplicates'] if d['similarity'] > 0.9]
    if high_similarity:
        print(f"  1. 高度相似测试 ({len(high_similarity)} 对): 考虑合并或重构")
        for dup in high_similarity[:3]:  # 只显示前3个
            func1_name = Path(dup['func1']['file']).name
            func2_name = Path(dup['func2']['file']).name
            print(f"     - {func1_name}::{dup['func1']['name']} vs {func2_name}::{dup['func2']['name']}")
    
    medium_similarity = [d for d in analysis['duplicates'] if 0.7 <= d['similarity'] <= 0.9]
    if medium_similarity:
        print(f"  2. 中度相似测试 ({len(medium_similarity)} 对): 检查是否可以提取公共测试逻辑")
    
    # 检查测试命名模式
    name_patterns = defaultdict(list)
    for func in [f for file_funcs in analysis['functions_by_file'].values() for f in file_funcs]:
        # 提取命名模式
        name_parts = func['name'].replace('test_', '').split('_')
        if len(name_parts) > 1:
            pattern = '_'.join(name_parts[:-1])  # 除了最后一部分
            name_patterns[pattern].append(func)
    
    similar_patterns = {k: v for k, v in name_patterns.items() if len(v) > 2}
    if similar_patterns:
        print(f"  3. 相似命名模式: 检查以下测试是否可以参数化")
        for pattern, funcs in list(similar_patterns.items())[:3]:
            print(f"     - 模式 '{pattern}_*': {len(funcs)} 个测试")

def main():
    """
    主函数
    """
    try:
        analysis = find_duplicate_tests()
        print_duplicate_analysis(analysis)
        
        # 生成详细报告文件
        report_file = ROOT_DIR / 'reports' / 'duplicate_tests_report.md'
        report_file.parent.mkdir(exist_ok=True)
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# 重复测试检测报告\n\n")
            f.write(f"生成时间: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## 总体统计\n\n")
            f.write(f"- 测试文件总数: {analysis['total_files']}\n")
            f.write(f"- 测试函数总数: {analysis['total_functions']}\n")
            f.write(f"- 疑似重复测试对: {analysis['duplicate_count']}\n\n")
            
            if analysis['duplicates']:
                f.write("## 重复测试详情\n\n")
                for i, dup in enumerate(analysis['duplicates'], 1):
                    func1 = dup['func1']
                    func2 = dup['func2']
                    similarity = dup['similarity']
                    
                    f.write(f"### {i}. 相似度: {similarity:.1%}\n\n")
                    f.write(f"**测试1:** `{Path(func1['file']).name}:{func1['line']}` - `{func1['name']}`\n")
                    f.write(f"**测试2:** `{Path(func2['file']).name}:{func2['line']}` - `{func2['name']}`\n\n")
                    
                    if func1['docstring']:
                        f.write(f"**描述1:** {func1['docstring']}\n\n")
                    if func2['docstring']:
                        f.write(f"**描述2:** {func2['docstring']}\n\n")
                    
                    common_keywords = func1['keywords'].intersection(func2['keywords'])
                    if common_keywords:
                        f.write(f"**共同关键词:** {', '.join(sorted(common_keywords))}\n\n")
                    
                    f.write("---\n\n")
        
        print(f"\n📄 详细报告已保存到: {report_file}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()