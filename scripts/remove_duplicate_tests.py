#!/usr/bin/env python3
"""
删除重复测试脚本

根据重复测试分析报告，自动删除项目中的重复测试用例。
保留最完整和最有代表性的测试版本。

作者: Trae AI Assistant
创建时间: 2025-08-22
"""

import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))


class DuplicateTestRemover:
    """重复测试删除器"""
    
    def __init__(self, test_dir: Path):
        self.test_dir = test_dir
        self.removed_tests: List[Dict] = []
        self.backup_dir = project_root / "backup" / "tests"
        
        # 根据分析报告定义的重复测试映射
        # 格式: {"文件名": ["要删除的测试函数名", ...]}
        self.duplicate_tests_to_remove = {
            "test_hairnet_detector.py": [
                # 保留第一个test_init (line 74)，删除第二个 (line 193)
                "test_init_duplicate_193"
            ],
            "test_motion_analyzer.py": [
                # 保留更完整的test_init，删除简单版本
                "test_init_simple"
            ],
            "test_data_manager.py": [
                # 保留最完整的test_init版本
                "test_init_simple"
            ],
            "test_pose_detector.py": [
                # 删除重复的初始化测试
                "test_init_duplicate"
            ]
        }
    
    def create_backup(self) -> None:
        """创建测试文件备份"""
        print("创建测试文件备份...")
        
        if not self.backup_dir.exists():
            self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # 备份所有测试文件
        for test_file in self.test_dir.glob("**/*.py"):
            if test_file.name.startswith("test_"):
                backup_path = self.backup_dir / test_file.relative_to(self.test_dir)
                backup_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(test_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                with open(backup_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                print(f"  备份: {test_file.name} -> {backup_path}")
    
    def analyze_test_file(self, file_path: Path) -> Dict:
        """分析测试文件，找出重复的测试函数"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 查找所有测试函数
        test_functions = []
        lines = content.split('\n')
        
        for i, line in enumerate(lines, 1):
            if re.match(r'^\s*def test_', line.strip()):
                func_name = re.search(r'def (test_\w+)', line).group(1)
                test_functions.append({
                    'name': func_name,
                    'line': i,
                    'content': line.strip()
                })
        
        return {
            'file_path': file_path,
            'content': content,
            'lines': lines,
            'test_functions': test_functions
        }
    
    def identify_duplicate_init_tests(self, file_analysis: Dict) -> List[Dict]:
        """识别重复的初始化测试"""
        init_tests = []
        
        for func in file_analysis['test_functions']:
            if func['name'] == 'test_init':
                init_tests.append(func)
        
        # 如果有多个test_init，标记除第一个外的所有为重复
        duplicates = []
        if len(init_tests) > 1:
            for i in range(1, len(init_tests)):
                duplicates.append({
                    'function': init_tests[i],
                    'reason': f'重复的初始化测试 (第{i+1}个)',
                    'keep_original': init_tests[0]
                })
        
        return duplicates
    
    def remove_duplicate_function(self, file_analysis: Dict, func_to_remove: Dict) -> str:
        """从文件中删除重复的测试函数"""
        lines = file_analysis['lines']
        func_line = func_to_remove['function']['line'] - 1  # 转换为0索引
        
        # 找到函数的结束位置
        start_line = func_line
        end_line = start_line + 1
        
        # 查找函数结束位置（下一个def或class，或文件结束）
        indent_level = len(lines[start_line]) - len(lines[start_line].lstrip())
        
        for i in range(start_line + 1, len(lines)):
            line = lines[i]
            if line.strip() == '':
                continue
            
            current_indent = len(line) - len(line.lstrip())
            
            # 如果遇到同级别或更低级别的定义，说明函数结束
            if (line.strip().startswith('def ') or 
                line.strip().startswith('class ')) and current_indent <= indent_level:
                end_line = i
                break
            
            # 如果到达文件末尾
            if i == len(lines) - 1:
                end_line = len(lines)
                break
        
        # 删除函数（包括其文档字符串和实现）
        new_lines = lines[:start_line] + lines[end_line:]
        
        return '\n'.join(new_lines)
    
    def process_file(self, file_path: Path) -> bool:
        """处理单个测试文件"""
        print(f"\n处理文件: {file_path.name}")
        
        file_analysis = self.analyze_test_file(file_path)
        duplicates = self.identify_duplicate_init_tests(file_analysis)
        
        if not duplicates:
            print(f"  未发现重复测试")
            return False
        
        print(f"  发现 {len(duplicates)} 个重复测试:")
        
        modified_content = file_analysis['content']
        
        # 按行号倒序删除，避免行号偏移问题
        duplicates.sort(key=lambda x: x['function']['line'], reverse=True)
        
        for duplicate in duplicates:
            func_name = duplicate['function']['name']
            func_line = duplicate['function']['line']
            reason = duplicate['reason']
            
            print(f"    删除: {func_name} (行 {func_line}) - {reason}")
            
            # 重新分析文件（因为内容已经改变）
            temp_analysis = {
                'lines': modified_content.split('\n'),
                'content': modified_content
            }
            
            # 找到要删除的函数
            func_to_remove = {'function': duplicate['function']}
            modified_content = self.remove_duplicate_function(temp_analysis, func_to_remove)
            
            self.removed_tests.append({
                'file': file_path.name,
                'function': func_name,
                'line': func_line,
                'reason': reason
            })
        
        # 写入修改后的内容
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(modified_content)
        
        print(f"  已删除 {len(duplicates)} 个重复测试")
        return True
    
    def remove_specific_duplicates(self) -> None:
        """删除特定的重复测试（基于分析报告）"""
        print("\n=== 删除特定重复测试 ===")
        
        # 处理test_hairnet_detector.py中的重复test_init
        hairnet_file = self.test_dir / "unit" / "test_hairnet_detector.py"
        if hairnet_file.exists():
            self._remove_duplicate_init_in_hairnet_detector(hairnet_file)
    
    def _remove_duplicate_init_in_hairnet_detector(self, file_path: Path) -> None:
        """删除test_hairnet_detector.py中的重复test_init"""
        print(f"处理 {file_path.name} 中的重复 test_init...")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        lines = content.split('\n')
        
        # 查找所有test_init函数
        init_functions = []
        for i, line in enumerate(lines):
            if re.match(r'^\s*def test_init', line.strip()):
                init_functions.append(i)
        
        if len(init_functions) <= 1:
            print(f"  {file_path.name}: 未发现重复的 test_init")
            return
        
        print(f"  发现 {len(init_functions)} 个 test_init 函数")
        
        # 保留第一个，删除其他的
        for i in range(len(init_functions) - 1, 0, -1):  # 倒序删除
            start_line = init_functions[i]
            
            # 找到函数结束位置
            end_line = len(lines)
            indent_level = len(lines[start_line]) - len(lines[start_line].lstrip())
            
            for j in range(start_line + 1, len(lines)):
                line = lines[j]
                if line.strip() == '':
                    continue
                
                current_indent = len(line) - len(line.lstrip())
                if ((line.strip().startswith('def ') or line.strip().startswith('class ')) 
                    and current_indent <= indent_level):
                    end_line = j
                    break
            
            # 删除函数
            print(f"    删除第 {i+1} 个 test_init (行 {start_line + 1} - {end_line})")
            del lines[start_line:end_line]
            
            self.removed_tests.append({
                'file': file_path.name,
                'function': f'test_init_{i+1}',
                'line': start_line + 1,
                'reason': '重复的初始化测试'
            })
        
        # 写入修改后的内容
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        print(f"  已删除 {len(init_functions) - 1} 个重复的 test_init")
    
    def run(self) -> None:
        """执行重复测试删除"""
        print("=== 开始删除重复测试 ===")
        
        # 创建备份
        self.create_backup()
        
        # 处理unit测试目录
        unit_test_dir = self.test_dir / "unit"
        if not unit_test_dir.exists():
            print(f"错误: 测试目录不存在: {unit_test_dir}")
            return
        
        modified_files = 0
        
        # 处理所有测试文件
        for test_file in unit_test_dir.glob("test_*.py"):
            if self.process_file(test_file):
                modified_files += 1
        
        # 删除特定的重复测试
        self.remove_specific_duplicates()
        
        # 生成报告
        self.generate_removal_report()
        
        print(f"\n=== 删除完成 ===")
        print(f"修改的文件数: {modified_files}")
        print(f"删除的测试数: {len(self.removed_tests)}")
        print(f"备份位置: {self.backup_dir}")
    
    def generate_removal_report(self) -> None:
        """生成删除报告"""
        report_path = project_root / "reports" / "duplicate_tests_removal_report.md"
        report_path.parent.mkdir(exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 重复测试删除报告\n\n")
            f.write(f"生成时间: {os.popen('date /t').read().strip()} {os.popen('time /t').read().strip()}\n\n")
            
            f.write("## 删除统计\n\n")
            f.write(f"- 删除的测试总数: {len(self.removed_tests)}\n")
            
            # 按文件分组统计
            file_stats = {}
            for test in self.removed_tests:
                file_name = test['file']
                if file_name not in file_stats:
                    file_stats[file_name] = 0
                file_stats[file_name] += 1
            
            f.write(f"- 涉及的文件数: {len(file_stats)}\n\n")
            
            for file_name, count in file_stats.items():
                f.write(f"  - {file_name}: {count} 个测试\n")
            
            f.write("\n## 删除详情\n\n")
            
            for test in self.removed_tests:
                f.write(f"### {test['file']}\n\n")
                f.write(f"- **函数**: `{test['function']}`\n")
                f.write(f"- **行号**: {test['line']}\n")
                f.write(f"- **原因**: {test['reason']}\n\n")
            
            f.write("## 备份信息\n\n")
            f.write(f"所有原始测试文件已备份到: `{self.backup_dir}`\n\n")
            f.write("如需恢复，请从备份目录复制文件。\n")
        
        print(f"\n删除报告已生成: {report_path}")


def main():
    """主函数"""
    test_dir = project_root / "tests"
    
    if not test_dir.exists():
        print(f"错误: 测试目录不存在: {test_dir}")
        return 1
    
    remover = DuplicateTestRemover(test_dir)
    remover.run()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())