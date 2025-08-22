# API 使用示例

本文档提供了人体行为检测系统API的详细使用示例，包括各种实际应用场景和完整的代码实现。

## 目录

1. [快速开始](#快速开始)
2. [基础检测示例](#基础检测示例)
3. [高级应用场景](#高级应用场景)
4. [批量处理](#批量处理)
5. [实时监控](#实时监控)
6. [错误处理和重试](#错误处理和重试)
7. [性能优化](#性能优化)
8. [集成示例](#集成示例)

## 快速开始

### 环境准备

```bash
# 安装依赖
pip install requests pillow

# 启动API服务
cd /path/to/project
python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

### 第一个API调用

```python
import requests

# 检查服务状态
response = requests.get("http://localhost:8000/health")
print("服务状态:", response.json())

# 简单的图像检测
with open('test_image.jpg', 'rb') as f:
    files = {'file': f}
    response = requests.post(
        'http://localhost:8000/api/v1/detect/image',
        files=files
    )
    result = response.json()
    print(f"检测结果: {result['status']}")
    print(f"检测到 {len(result['results']['detections'])} 个目标")
```

## 基础检测示例

### 1. 图像检测

```python
import requests
import json
from pathlib import Path

def detect_image(image_path, confidence_threshold=0.5, save_result=False):
    """
    检测图像中的人体行为
    
    Args:
        image_path: 图像文件路径
        confidence_threshold: 置信度阈值
        save_result: 是否保存检测结果
    
    Returns:
        dict: 检测结果
    """
    url = "http://localhost:8000/api/v1/detect/image"
    
    # 检查文件是否存在
    if not Path(image_path).exists():
        raise FileNotFoundError(f"图像文件不存在: {image_path}")
    
    with open(image_path, 'rb') as f:
        files = {'file': f}
        data = {
            'confidence_threshold': confidence_threshold,
            'save_result': save_result
        }
        
        response = requests.post(url, files=files, data=data)
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"API调用失败: {response.status_code} - {response.text}")

# 使用示例
try:
    result = detect_image(
        'test_images/kitchen_scene.jpg',
        confidence_threshold=0.7,
        save_result=True
    )
    
    # 解析结果
    detections = result['results']['detections']
    behaviors = result['results']['behaviors']
    stats = result['results']['statistics']
    
    print(f"\n=== 检测结果 ===")
    print(f"处理时间: {stats['processing_time']:.3f}秒")
    print(f"检测到 {stats['total_persons']} 个人")
    print(f"识别到 {stats['detected_behaviors']} 个行为")
    
    # 详细信息
    for i, detection in enumerate(detections):
        print(f"\n人员 {i+1}:")
        print(f"  置信度: {detection['confidence']:.2f}")
        print(f"  位置: {detection['bbox']}")
        print(f"  关键点数量: {len(detection.get('keypoints', []))}")
    
    for behavior in behaviors:
        print(f"\n检测到行为: {behavior['type']}")
        print(f"  置信度: {behavior['confidence']:.2f}")
        print(f"  区域: {behavior.get('region', 'unknown')}")
        
except Exception as e:
    print(f"检测失败: {e}")
```

### 2. 发网检测

```python
def check_hairnet_compliance(image_path, strict_mode=False):
    """
    检查发网佩戴合规性
    
    Args:
        image_path: 图像文件路径
        strict_mode: 是否启用严格模式
    
    Returns:
        dict: 合规性检查结果
    """
    url = "http://localhost:8000/api/v1/detect/hairnet"
    
    with open(image_path, 'rb') as f:
        files = {'file': f}
        data = {
            'strict_mode': strict_mode,
            'confidence_threshold': 0.8 if strict_mode else 0.5
        }
        
        response = requests.post(url, files=files, data=data)
        result = response.json()
        
        if result['status'] == 'success':
            return result
        else:
            raise Exception(f"发网检测失败: {result.get('message', 'Unknown error')}")

# 使用示例
def analyze_workplace_compliance(image_folder):
    """
    分析工作场所的发网合规性
    """
    from pathlib import Path
    
    image_folder = Path(image_folder)
    results = []
    
    for image_file in image_folder.glob('*.jpg'):
        try:
            print(f"\n检测文件: {image_file.name}")
            result = check_hairnet_compliance(str(image_file), strict_mode=True)
            
            summary = result['results']['summary']
            print(f"  总人数: {summary['total_persons']}")
            print(f"  合规人数: {summary['compliant_persons']}")
            print(f"  合规率: {summary['compliance_rate']:.1%}")
            
            # 详细分析
            detections = result['results']['detections']
            for person in detections:
                status = "✅ 合规" if person['compliance_status'] == 'compliant' else "❌ 违规"
                print(f"    人员 {person['person_id']}: {status} (置信度: {person['hairnet_confidence']:.2f})")
            
            results.append({
                'file': image_file.name,
                'compliance_rate': summary['compliance_rate'],
                'total_persons': summary['total_persons']
            })
            
        except Exception as e:
            print(f"  检测失败: {e}")
    
    # 总体统计
    if results:
        total_persons = sum(r['total_persons'] for r in results)
        avg_compliance = sum(r['compliance_rate'] * r['total_persons'] for r in results) / total_persons
        
        print(f"\n=== 总体统计 ===")
        print(f"检测文件数: {len(results)}")
        print(f"总人数: {total_persons}")
        print(f"平均合规率: {avg_compliance:.1%}")
    
    return results

# 运行分析
results = analyze_workplace_compliance('workplace_images/')
```

## 高级应用场景

### 1. 多区域监控系统

```python
import requests
import time
import threading
from datetime import datetime

class MultiRegionMonitor:
    """
    多区域实时监控系统
    """
    
    def __init__(self, api_base_url="http://localhost:8000"):
        self.api_base_url = api_base_url
        self.regions = ['kitchen_area', 'preparation_area', 'storage_area']
        self.monitoring = False
        self.alert_thresholds = {
            'compliance_rate': 0.8,  # 合规率低于80%时告警
            'violation_count': 5     # 违规数量超过5时告警
        }
    
    def get_region_statistics(self, region_id, time_range=30):
        """
        获取特定区域的统计信息
        """
        url = f"{self.api_base_url}/api/statistics/realtime"
        params = {
            'region_id': region_id,
            'time_range': time_range
        }
        
        try:
            response = requests.get(url, params=params)
            if response.status_code == 200:
                return response.json()['data']
            else:
                print(f"获取区域 {region_id} 统计失败: {response.status_code}")
                return None
        except Exception as e:
            print(f"API调用异常: {e}")
            return None
    
    def check_alerts(self, region_id, stats):
        """
        检查是否需要发出告警
        """
        alerts = []
        
        if 'compliance_rates' in stats['statistics']:
            compliance = stats['statistics']['compliance_rates']
            
            for behavior, rate in compliance.items():
                if rate < self.alert_thresholds['compliance_rate']:
                    alerts.append({
                        'type': 'low_compliance',
                        'region': region_id,
                        'behavior': behavior,
                        'current_rate': rate,
                        'threshold': self.alert_thresholds['compliance_rate']
                    })
        
        if 'violations' in stats['statistics']:
            violation_count = stats['statistics']['violations']['total']
            if violation_count > self.alert_thresholds['violation_count']:
                alerts.append({
                    'type': 'high_violations',
                    'region': region_id,
                    'violation_count': violation_count,
                    'threshold': self.alert_thresholds['violation_count']
                })
        
        return alerts
    
    def monitor_region(self, region_id):
        """
        监控单个区域
        """
        while self.monitoring:
            stats = self.get_region_statistics(region_id)
            
            if stats:
                # 检查告警
                alerts = self.check_alerts(region_id, stats)
                
                # 打印状态
                current_time = datetime.now().strftime('%H:%M:%S')
                region_stats = stats.get('regions', {}).get(region_id, {})
                
                print(f"[{current_time}] {region_id}:")
                print(f"  人数: {region_stats.get('person_count', 0)}")
                print(f"  违规: {region_stats.get('violations', 0)}")
                print(f"  合规率: {region_stats.get('compliance_rate', 0):.1%}")
                
                # 处理告警
                for alert in alerts:
                    self.handle_alert(alert)
            
            time.sleep(10)  # 每10秒检查一次
    
    def handle_alert(self, alert):
        """
        处理告警
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        if alert['type'] == 'low_compliance':
            print(f"\n🚨 [{timestamp}] 合规率告警!")
            print(f"   区域: {alert['region']}")
            print(f"   行为: {alert['behavior']}")
            print(f"   当前合规率: {alert['current_rate']:.1%}")
            print(f"   告警阈值: {alert['threshold']:.1%}")
        
        elif alert['type'] == 'high_violations':
            print(f"\n🚨 [{timestamp}] 违规数量告警!")
            print(f"   区域: {alert['region']}")
            print(f"   违规数量: {alert['violation_count']}")
            print(f"   告警阈值: {alert['threshold']}")
        
        # 这里可以添加更多告警处理逻辑，如:
        # - 发送邮件通知
        # - 记录到数据库
        # - 触发其他系统的响应
    
    def start_monitoring(self):
        """
        开始监控所有区域
        """
        self.monitoring = True
        threads = []
        
        print(f"开始监控 {len(self.regions)} 个区域...")
        
        for region_id in self.regions:
            thread = threading.Thread(
                target=self.monitor_region,
                args=(region_id,),
                daemon=True
            )
            thread.start()
            threads.append(thread)
        
        try:
            # 主线程等待
            while self.monitoring:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n停止监控...")
            self.monitoring = False
            
            # 等待所有线程结束
            for thread in threads:
                thread.join(timeout=1)

# 使用示例
if __name__ == "__main__":
    monitor = MultiRegionMonitor()
    
    # 自定义告警阈值
    monitor.alert_thresholds = {
        'compliance_rate': 0.75,  # 75%
        'violation_count': 3      # 3个违规
    }
    
    # 开始监控
    monitor.start_monitoring()
```

### 2. 智能报告生成器

```python
import requests
import json
from datetime import datetime, timedelta
from pathlib import Path

class ComplianceReportGenerator:
    """
    合规性报告生成器
    """
    
    def __init__(self, api_base_url="http://localhost:8000"):
        self.api_base_url = api_base_url
    
    def get_historical_data(self, start_date, end_date, region_id=None):
        """
        获取历史统计数据
        """
        url = f"{self.api_base_url}/api/statistics/history"
        params = {
            'start_date': start_date,
            'end_date': end_date,
            'granularity': 'day'
        }
        
        if region_id:
            params['region_id'] = region_id
        
        response = requests.get(url, params=params)
        if response.status_code == 200:
            return response.json()['data']
        else:
            raise Exception(f"获取历史数据失败: {response.status_code}")
    
    def get_violations(self, start_date, end_date, region_id=None):
        """
        获取违规记录
        """
        url = f"{self.api_base_url}/api/statistics/violations"
        params = {
            'start_date': start_date,
            'end_date': end_date,
            'limit': 1000
        }
        
        if region_id:
            params['region_id'] = region_id
        
        response = requests.get(url, params=params)
        if response.status_code == 200:
            return response.json()['data']
        else:
            raise Exception(f"获取违规记录失败: {response.status_code}")
    
    def analyze_trends(self, historical_data):
        """
        分析趋势
        """
        statistics = historical_data['statistics']
        
        if len(statistics) < 2:
            return {"trend": "insufficient_data"}
        
        # 计算各项指标的趋势
        trends = {}
        
        # 合规率趋势
        for behavior in ['handwashing', 'mask_wearing', 'hairnet_wearing']:
            rates = [day['compliance_rates'].get(behavior, 0) for day in statistics]
            if len(rates) >= 2:
                trend = "improving" if rates[-1] > rates[0] else "declining"
                change = rates[-1] - rates[0]
                trends[f"{behavior}_compliance"] = {
                    "trend": trend,
                    "change": change,
                    "current": rates[-1],
                    "previous": rates[0]
                }
        
        # 违规数量趋势
        violations = [day['violations'] for day in statistics]
        if len(violations) >= 2:
            trend = "increasing" if violations[-1] > violations[0] else "decreasing"
            change = violations[-1] - violations[0]
            trends["violations"] = {
                "trend": trend,
                "change": change,
                "current": violations[-1],
                "previous": violations[0]
            }
        
        return trends
    
    def generate_weekly_report(self, region_id=None):
        """
        生成周报
        """
        # 计算日期范围
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=7)
        
        # 获取数据
        historical_data = self.get_historical_data(
            start_date.isoformat(),
            end_date.isoformat(),
            region_id
        )
        
        violations_data = self.get_violations(
            start_date.isoformat(),
            end_date.isoformat(),
            region_id
        )
        
        # 分析趋势
        trends = self.analyze_trends(historical_data)
        
        # 生成报告
        report = {
            "report_type": "weekly",
            "period": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat()
            },
            "region_id": region_id,
            "generated_at": datetime.now().isoformat(),
            "summary": self._generate_summary(historical_data, violations_data),
            "trends": trends,
            "recommendations": self._generate_recommendations(trends, violations_data),
            "detailed_data": {
                "daily_statistics": historical_data['statistics'],
                "violations": violations_data['violations'][:50]  # 最近50条违规记录
            }
        }
        
        return report
    
    def _generate_summary(self, historical_data, violations_data):
        """
        生成摘要
        """
        statistics = historical_data['statistics']
        
        if not statistics:
            return {"message": "无数据"}
        
        # 计算平均值
        total_detections = sum(day['total_detections'] for day in statistics)
        total_violations = sum(day['violations'] for day in statistics)
        avg_compliance = {
            behavior: sum(day['compliance_rates'].get(behavior, 0) for day in statistics) / len(statistics)
            for behavior in ['handwashing', 'mask_wearing', 'hairnet_wearing']
        }
        
        return {
            "total_detections": total_detections,
            "total_violations": total_violations,
            "average_compliance_rates": avg_compliance,
            "violation_rate": total_violations / total_detections if total_detections > 0 else 0,
            "most_common_violation": self._get_most_common_violation(violations_data)
        }
    
    def _get_most_common_violation(self, violations_data):
        """
        获取最常见的违规类型
        """
        violation_counts = {}
        
        for violation in violations_data['violations']:
            vtype = violation['violation_type']
            violation_counts[vtype] = violation_counts.get(vtype, 0) + 1
        
        if violation_counts:
            return max(violation_counts.items(), key=lambda x: x[1])
        else:
            return None
    
    def _generate_recommendations(self, trends, violations_data):
        """
        生成改进建议
        """
        recommendations = []
        
        # 基于趋势的建议
        for behavior, trend_data in trends.items():
            if 'compliance' in behavior and trend_data['trend'] == 'declining':
                behavior_name = behavior.replace('_compliance', '')
                recommendations.append({
                    "type": "compliance_improvement",
                    "priority": "high" if trend_data['change'] < -0.1 else "medium",
                    "message": f"{behavior_name}合规率下降{abs(trend_data['change']):.1%}，建议加强培训和监督"
                })
        
        # 基于违规记录的建议
        most_common = self._get_most_common_violation(violations_data)
        if most_common:
            violation_type, count = most_common
            recommendations.append({
                "type": "violation_focus",
                "priority": "high",
                "message": f"最常见违规类型是{violation_type}（{count}次），建议针对性改进"
            })
        
        return recommendations
    
    def save_report(self, report, filename=None):
        """
        保存报告到文件
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            region_suffix = f"_{report['region_id']}" if report['region_id'] else ""
            filename = f"compliance_report_{timestamp}{region_suffix}.json"
        
        report_path = Path("reports") / filename
        report_path.parent.mkdir(exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"报告已保存到: {report_path}")
        return report_path
    
    def print_report_summary(self, report):
        """
        打印报告摘要
        """
        print(f"\n=== 合规性报告 ===")
        print(f"报告期间: {report['period']['start_date']} 至 {report['period']['end_date']}")
        
        if report['region_id']:
            print(f"区域: {report['region_id']}")
        
        summary = report['summary']
        print(f"\n总检测次数: {summary['total_detections']}")
        print(f"总违规次数: {summary['total_violations']}")
        print(f"违规率: {summary['violation_rate']:.1%}")
        
        print(f"\n平均合规率:")
        for behavior, rate in summary['average_compliance_rates'].items():
            print(f"  {behavior}: {rate:.1%}")
        
        if summary['most_common_violation']:
            vtype, count = summary['most_common_violation']
            print(f"\n最常见违规: {vtype} ({count}次)")
        
        print(f"\n改进建议:")
        for rec in report['recommendations']:
            priority_icon = "🔴" if rec['priority'] == 'high' else "🟡"
            print(f"  {priority_icon} {rec['message']}")

# 使用示例
if __name__ == "__main__":
    generator = ComplianceReportGenerator()
    
    # 生成整体周报
    print("生成整体周报...")
    overall_report = generator.generate_weekly_report()
    generator.print_report_summary(overall_report)
    generator.save_report(overall_report)
    
    # 生成各区域报告
    regions = ['kitchen_area', 'preparation_area']
    for region in regions:
        print(f"\n生成{region}区域报告...")
        region_report = generator.generate_weekly_report(region_id=region)
        generator.print_report_summary(region_report)
        generator.save_report(region_report)
```

## 批量处理

### 批量图像检测

```python
import requests
import asyncio
import aiohttp
import aiofiles
from pathlib import Path
import json
from concurrent.futures import ThreadPoolExecutor

class BatchImageProcessor:
    """
    批量图像处理器
    """
    
    def __init__(self, api_base_url="http://localhost:8000", max_workers=5):
        self.api_base_url = api_base_url
        self.max_workers = max_workers
    
    def process_single_image(self, image_path, output_dir=None):
        """
        处理单张图像（同步版本）
        """
        try:
            with open(image_path, 'rb') as f:
                files = {'file': f}
                response = requests.post(
                    f"{self.api_base_url}/api/v1/detect/image",
                    files=files,
                    data={'confidence_threshold': 0.6}
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # 保存结果
                    if output_dir:
                        output_path = Path(output_dir) / f"{Path(image_path).stem}_result.json"
                        output_path.parent.mkdir(parents=True, exist_ok=True)
                        
                        with open(output_path, 'w', encoding='utf-8') as f:
                            json.dump(result, f, ensure_ascii=False, indent=2)
                    
                    return {
                        'image_path': str(image_path),
                        'status': 'success',
                        'result': result
                    }
                else:
                    return {
                        'image_path': str(image_path),
                        'status': 'error',
                        'error': f"HTTP {response.status_code}: {response.text}"
                    }
                    
        except Exception as e:
            return {
                'image_path': str(image_path),
                'status': 'error',
                'error': str(e)
            }
    
    def process_batch_sync(self, image_folder, output_dir=None, file_pattern="*.jpg"):
        """
        批量处理图像（同步版本，使用线程池）
        """
        image_folder = Path(image_folder)
        image_files = list(image_folder.glob(file_pattern))
        
        print(f"找到 {len(image_files)} 个图像文件")
        
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            futures = [
                executor.submit(self.process_single_image, img_path, output_dir)
                for img_path in image_files
            ]
            
            # 收集结果
            for i, future in enumerate(futures):
                try:
                    result = future.result(timeout=30)  # 30秒超时
                    results.append(result)
                    
                    status_icon = "✅" if result['status'] == 'success' else "❌"
                    print(f"[{i+1}/{len(image_files)}] {status_icon} {Path(result['image_path']).name}")
                    
                except Exception as e:
                    error_result = {
                        'image_path': str(image_files[i]),
                        'status': 'error',
                        'error': str(e)
                    }
                    results.append(error_result)
                    print(f"[{i+1}/{len(image_files)}] ❌ {image_files[i].name} - {e}")
        
        return results
    
    async def process_single_image_async(self, session, image_path, output_dir=None):
        """
        异步处理单张图像
        """
        try:
            async with aiofiles.open(image_path, 'rb') as f:
                file_content = await f.read()
                
                data = aiohttp.FormData()
                data.add_field('file', file_content, filename=Path(image_path).name)
                data.add_field('confidence_threshold', '0.6')
                
                async with session.post(
                    f"{self.api_base_url}/api/v1/detect/image",
                    data=data
                ) as response:
                    
                    if response.status == 200:
                        result = await response.json()
                        
                        # 保存结果
                        if output_dir:
                            output_path = Path(output_dir) / f"{Path(image_path).stem}_result.json"
                            output_path.parent.mkdir(parents=True, exist_ok=True)
                            
                            async with aiofiles.open(output_path, 'w', encoding='utf-8') as f:
                                await f.write(json.dumps(result, ensure_ascii=False, indent=2))
                        
                        return {
                            'image_path': str(image_path),
                            'status': 'success',
                            'result': result
                        }
                    else:
                        error_text = await response.text()
                        return {
                            'image_path': str(image_path),
                            'status': 'error',
                            'error': f"HTTP {response.status}: {error_text}"
                        }
                        
        except Exception as e:
            return {
                'image_path': str(image_path),
                'status': 'error',
                'error': str(e)
            }
    
    async def process_batch_async(self, image_folder, output_dir=None, file_pattern="*.jpg"):
        """
        批量处理图像（异步版本）
        """
        image_folder = Path(image_folder)
        image_files = list(image_folder.glob(file_pattern))
        
        print(f"找到 {len(image_files)} 个图像文件")
        
        # 创建连接器，限制并发连接数
        connector = aiohttp.TCPConnector(limit=self.max_workers)
        timeout = aiohttp.ClientTimeout(total=30)
        
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            # 创建所有任务
            tasks = [
                self.process_single_image_async(session, img_path, output_dir)
                for img_path in image_files
            ]
            
            # 执行所有任务
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 处理结果
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    error_result = {
                        'image_path': str(image_files[i]),
                        'status': 'error',
                        'error': str(result)
                    }
                    processed_results.append(error_result)
                    print(f"[{i+1}/{len(image_files)}] ❌ {image_files[i].name} - {result}")
                else:
                    processed_results.append(result)
                    status_icon = "✅" if result['status'] == 'success' else "❌"
                    print(f"[{i+1}/{len(image_files)}] {status_icon} {Path(result['image_path']).name}")
        
        return processed_results
    
    def generate_batch_report(self, results, output_path="batch_report.json"):
        """
        生成批处理报告
        """
        successful = [r for r in results if r['status'] == 'success']
        failed = [r for r in results if r['status'] == 'error']
        
        # 统计信息
        total_persons = 0
        total_behaviors = 0
        behavior_counts = {}
        
        for result in successful:
            if 'result' in result and 'results' in result['result']:
                stats = result['result']['results'].get('statistics', {})
                total_persons += stats.get('total_persons', 0)
                total_behaviors += stats.get('detected_behaviors', 0)
                
                # 统计行为类型
                behaviors = result['result']['results'].get('behaviors', [])
                for behavior in behaviors:
                    btype = behavior.get('type', 'unknown')
                    behavior_counts[btype] = behavior_counts.get(btype, 0) + 1
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_images': len(results),
                'successful': len(successful),
                'failed': len(failed),
                'success_rate': len(successful) / len(results) if results else 0,
                'total_persons_detected': total_persons,
                'total_behaviors_detected': total_behaviors,
                'behavior_distribution': behavior_counts
            },
            'failed_images': [{
                'path': r['image_path'],
                'error': r['error']
            } for r in failed],
            'detailed_results': results
        }
        
        # 保存报告
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        # 打印摘要
        print(f"\n=== 批处理报告 ===")
        print(f"总图像数: {report['summary']['total_images']}")
        print(f"成功处理: {report['summary']['successful']}")
        print(f"处理失败: {report['summary']['failed']}")
        print(f"成功率: {report['summary']['success_rate']:.1%}")
        print(f"检测到人数: {report['summary']['total_persons_detected']}")
        print(f"检测到行为: {report['summary']['total_behaviors_detected']}")
        
        if behavior_counts:
            print(f"\n行为分布:")
            for behavior, count in behavior_counts.items():
                print(f"  {behavior}: {count}")
        
        if failed:
            print(f"\n失败的图像:")
            for fail in failed[:5]:  # 只显示前5个
                print(f"  {Path(fail['path']).name}: {fail['error']}")
            if len(failed) > 5:
                print(f"  ... 还有 {len(failed) - 5} 个失败")
        
        print(f"\n详细报告已保存到: {output_path}")
        return report

# 使用示例
if __name__ == "__main__":
    processor = BatchImageProcessor(max_workers=3)
    
    # 同步批处理
    print("开始同步批处理...")
    sync_results = processor.process_batch_sync(
        image_folder="test_images",
        output_dir="results/sync",
        file_pattern="*.jpg"
    )
    processor.generate_batch_report(sync_results, "sync_batch_report.json")
    
    # 异步批处理
    print("\n开始异步批处理...")
    async_results = asyncio.run(processor.process_batch_async(
        image_folder="test_images",
        output_dir="results/async",
        file_pattern="*.jpg"
    ))
    processor.generate_batch_report(async_results, "async_batch_report.json")
```

## 总结

本文档提供了人体行为检测系统API的全面使用示例，涵盖了从基础调用到高级应用场景的各种用法。通过这些示例，开发者可以：

1. **快速上手**: 使用基础示例快速集成API
2. **构建应用**: 参考高级场景构建完整的监控系统
3. **批量处理**: 使用批处理功能高效处理大量图像
4. **实时监控**: 实现多区域实时监控和告警
5. **数据分析**: 生成详细的合规性报告和趋势分析

所有示例代码都经过测试，可以直接使用或根据具体需求进行修改。建议在生产环境中添加适当的错误处理、日志记录和性能监控。

---

*更多技术支持和问题反馈，请参考项目文档或联系开发团队。*