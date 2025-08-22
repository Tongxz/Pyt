# API 接口文档

## 概述

人体行为检测系统提供了一套完整的 RESTful API 接口，支持图像检测、发网检测、实时统计等功能。本文档详细介绍了所有可用的 API 端点、请求参数、响应格式和使用示例。

## 基础信息

- **基础URL**: `http://localhost:8000`
- **API版本**: v1
- **内容类型**: `application/json` 或 `multipart/form-data`
- **字符编码**: UTF-8

## 认证

当前版本的API不需要认证，但建议在生产环境中添加适当的认证机制。

## 通用响应格式

所有API响应都遵循以下格式：

```json
{
  "status": "success|error",
  "message": "响应消息",
  "data": {},
  "timestamp": "2024-01-01T12:00:00Z"
}
```

## API 端点

### 1. 健康检查

检查API服务的运行状态。

**端点**: `GET /health`

**请求参数**: 无

**响应示例**:
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T12:00:00.000Z",
  "version": "1.0.0",
  "services": {
    "detection_service": "running",
    "region_service": "running"
  }
}
```

**使用示例**:
```bash
curl -X GET "http://localhost:8000/health"
```

```python
import requests

response = requests.get("http://localhost:8000/health")
print(response.json())
```

### 2. 图像检测

对上传的图像进行人体行为检测。

**端点**: `POST /api/v1/detect/image`

**请求参数**:
- `file` (文件): 要检测的图像文件 (支持 JPG, PNG, BMP 格式)
- `confidence_threshold` (可选, float): 置信度阈值，默认 0.5
- `save_result` (可选, bool): 是否保存检测结果，默认 false

**响应示例**:
```json
{
  "filename": "test_image.jpg",
  "detection_type": "image",
  "status": "success",
  "results": {
    "detections": [
      {
        "class": "person",
        "confidence": 0.95,
        "bbox": [100, 150, 300, 450],
        "keypoints": [
          {"x": 200, "y": 180, "confidence": 0.9},
          {"x": 220, "y": 200, "confidence": 0.85}
        ]
      }
    ],
    "behaviors": [
      {
        "type": "handwashing",
        "confidence": 0.78,
        "person_id": 0,
        "region": "sink_area"
      }
    ],
    "statistics": {
      "total_persons": 1,
      "detected_behaviors": 1,
      "processing_time": 0.234
    }
  },
  "timestamp": "2024-01-01T12:00:00.000Z"
}
```

**使用示例**:
```bash
# 基础检测
curl -X POST "http://localhost:8000/api/v1/detect/image" \
  -F "file=@test_image.jpg"

# 带参数检测
curl -X POST "http://localhost:8000/api/v1/detect/image" \
  -F "file=@test_image.jpg" \
  -F "confidence_threshold=0.7" \
  -F "save_result=true"
```

```python
import requests

# 基础检测
with open('test_image.jpg', 'rb') as f:
    files = {'file': f}
    response = requests.post(
        'http://localhost:8000/api/v1/detect/image',
        files=files
    )
    result = response.json()
    print(f"检测到 {result['results']['statistics']['total_persons']} 个人")

# 带参数检测
with open('test_image.jpg', 'rb') as f:
    files = {'file': f}
    data = {
        'confidence_threshold': 0.7,
        'save_result': True
    }
    response = requests.post(
        'http://localhost:8000/api/v1/detect/image',
        files=files,
        data=data
    )
    result = response.json()
```

### 3. 发网检测

专门用于检测人员是否佩戴发网。

**端点**: `POST /api/v1/detect/hairnet`

**请求参数**:
- `file` (文件): 要检测的图像文件
- `confidence_threshold` (可选, float): 置信度阈值，默认 0.5
- `strict_mode` (可选, bool): 严格模式，默认 false

**响应示例**:
```json
{
  "filename": "worker_image.jpg",
  "detection_type": "hairnet",
  "status": "success",
  "results": {
    "detections": [
      {
        "person_id": 0,
        "bbox": [120, 80, 280, 400],
        "head_bbox": [150, 80, 250, 180],
        "hairnet_detected": true,
        "hairnet_confidence": 0.87,
        "compliance_status": "compliant"
      },
      {
        "person_id": 1,
        "bbox": [300, 100, 450, 420],
        "head_bbox": [320, 100, 430, 200],
        "hairnet_detected": false,
        "hairnet_confidence": 0.12,
        "compliance_status": "non_compliant"
      }
    ],
    "summary": {
      "total_persons": 2,
      "compliant_persons": 1,
      "non_compliant_persons": 1,
      "compliance_rate": 0.5
    },
    "processing_time": 0.156
  },
  "timestamp": "2024-01-01T12:00:00.000Z"
}
```

**使用示例**:
```bash
# 基础发网检测
curl -X POST "http://localhost:8000/api/v1/detect/hairnet" \
  -F "file=@worker_image.jpg"

# 严格模式检测
curl -X POST "http://localhost:8000/api/v1/detect/hairnet" \
  -F "file=@worker_image.jpg" \
  -F "strict_mode=true" \
  -F "confidence_threshold=0.8"
```

```python
import requests

def check_hairnet_compliance(image_path, strict_mode=False):
    """检查发网合规性"""
    with open(image_path, 'rb') as f:
        files = {'file': f}
        data = {'strict_mode': strict_mode}
        
        response = requests.post(
            'http://localhost:8000/api/v1/detect/hairnet',
            files=files,
            data=data
        )
        
        result = response.json()
        summary = result['results']['summary']
        
        print(f"检测结果:")
        print(f"  总人数: {summary['total_persons']}")
        print(f"  合规人数: {summary['compliant_persons']}")
        print(f"  合规率: {summary['compliance_rate']:.1%}")
        
        return result

# 使用示例
result = check_hairnet_compliance('worker_image.jpg', strict_mode=True)
```

### 4. 实时统计

获取系统的实时检测统计信息。

**端点**: `GET /api/statistics/realtime`

**请求参数**:
- `region_id` (可选, string): 特定区域ID
- `time_range` (可选, int): 时间范围（分钟），默认 60

**响应示例**:
```json
{
  "status": "success",
  "data": {
    "current_time": "2024-01-01T12:00:00.000Z",
    "time_range_minutes": 60,
    "statistics": {
      "total_detections": 156,
      "person_count": 23,
      "behavior_detections": {
        "handwashing": 45,
        "mask_wearing": 78,
        "hairnet_wearing": 67
      },
      "compliance_rates": {
        "handwashing_compliance": 0.85,
        "mask_compliance": 0.92,
        "hairnet_compliance": 0.78
      },
      "violations": {
        "total": 12,
        "by_type": {
          "no_handwashing": 5,
          "no_mask": 3,
          "no_hairnet": 4
        }
      }
    },
    "regions": {
      "kitchen_area": {
        "person_count": 8,
        "violations": 3,
        "compliance_rate": 0.82
      },
      "preparation_area": {
        "person_count": 15,
        "violations": 9,
        "compliance_rate": 0.75
      }
    }
  },
  "timestamp": "2024-01-01T12:00:00.000Z"
}
```

**使用示例**:
```bash
# 获取全部统计
curl -X GET "http://localhost:8000/api/statistics/realtime"

# 获取特定区域统计
curl -X GET "http://localhost:8000/api/statistics/realtime?region_id=kitchen_area&time_range=30"
```

```python
import requests
import time

def monitor_realtime_statistics(region_id=None, interval=30):
    """实时监控统计信息"""
    url = "http://localhost:8000/api/statistics/realtime"
    params = {}
    
    if region_id:
        params['region_id'] = region_id
    
    while True:
        try:
            response = requests.get(url, params=params)
            data = response.json()['data']
            
            stats = data['statistics']
            print(f"\n=== 实时统计 ({data['current_time']}) ===")
            print(f"总检测次数: {stats['total_detections']}")
            print(f"当前人数: {stats['person_count']}")
            print(f"违规总数: {stats['violations']['total']}")
            
            compliance = stats['compliance_rates']
            print(f"合规率:")
            print(f"  洗手: {compliance['handwashing_compliance']:.1%}")
            print(f"  口罩: {compliance['mask_compliance']:.1%}")
            print(f"  发网: {compliance['hairnet_compliance']:.1%}")
            
            time.sleep(interval)
            
        except KeyboardInterrupt:
            print("\n监控已停止")
            break
        except Exception as e:
            print(f"获取统计信息失败: {e}")
            time.sleep(5)

# 使用示例
monitor_realtime_statistics(region_id="kitchen_area", interval=10)
```

### 5. 历史统计

获取历史统计数据。

**端点**: `GET /api/statistics/history`

**请求参数**:
- `start_date` (string): 开始日期 (YYYY-MM-DD)
- `end_date` (string): 结束日期 (YYYY-MM-DD)
- `region_id` (可选, string): 区域ID
- `granularity` (可选, string): 数据粒度 (hour/day/week)，默认 day

**响应示例**:
```json
{
  "status": "success",
  "data": {
    "date_range": {
      "start": "2024-01-01",
      "end": "2024-01-07"
    },
    "granularity": "day",
    "statistics": [
      {
        "date": "2024-01-01",
        "total_detections": 1250,
        "unique_persons": 89,
        "violations": 45,
        "compliance_rates": {
          "handwashing": 0.87,
          "mask_wearing": 0.94,
          "hairnet_wearing": 0.82
        }
      },
      {
        "date": "2024-01-02",
        "total_detections": 1180,
        "unique_persons": 76,
        "violations": 38,
        "compliance_rates": {
          "handwashing": 0.89,
          "mask_wearing": 0.96,
          "hairnet_wearing": 0.85
        }
      }
    ]
  },
  "timestamp": "2024-01-01T12:00:00.000Z"
}
```

### 6. 违规记录

获取违规记录列表。

**端点**: `GET /api/statistics/violations`

**请求参数**:
- `limit` (可选, int): 返回记录数量，默认 50
- `offset` (可选, int): 偏移量，默认 0
- `region_id` (可选, string): 区域ID
- `violation_type` (可选, string): 违规类型
- `start_date` (可选, string): 开始日期
- `end_date` (可选, string): 结束日期

**响应示例**:
```json
{
  "status": "success",
  "data": {
    "total_count": 156,
    "violations": [
      {
        "id": "viol_001",
        "timestamp": "2024-01-01T10:30:15.000Z",
        "region_id": "kitchen_area",
        "violation_type": "no_handwashing",
        "person_id": "person_123",
        "confidence": 0.92,
        "image_path": "/violations/2024/01/01/viol_001.jpg",
        "resolved": false
      },
      {
        "id": "viol_002",
        "timestamp": "2024-01-01T11:15:42.000Z",
        "region_id": "preparation_area",
        "violation_type": "no_hairnet",
        "person_id": "person_456",
        "confidence": 0.88,
        "image_path": "/violations/2024/01/01/viol_002.jpg",
        "resolved": true
      }
    ],
    "pagination": {
      "limit": 50,
      "offset": 0,
      "has_more": true
    }
  },
  "timestamp": "2024-01-01T12:00:00.000Z"
}
```

## 错误处理

### 错误响应格式

```json
{
  "status": "error",
  "error": {
    "code": "ERROR_CODE",
    "message": "错误描述",
    "details": "详细错误信息"
  },
  "timestamp": "2024-01-01T12:00:00.000Z"
}
```

### 常见错误码

| 错误码 | HTTP状态码 | 描述 |
|--------|------------|------|
| `INVALID_FILE_FORMAT` | 400 | 不支持的文件格式 |
| `FILE_TOO_LARGE` | 400 | 文件大小超过限制 |
| `MISSING_PARAMETER` | 400 | 缺少必需参数 |
| `INVALID_PARAMETER` | 400 | 参数值无效 |
| `SERVICE_UNAVAILABLE` | 503 | 检测服务不可用 |
| `PROCESSING_ERROR` | 500 | 处理过程中发生错误 |
| `REGION_NOT_FOUND` | 404 | 指定区域不存在 |

### 错误处理示例

```python
import requests

def safe_api_call(url, **kwargs):
    """安全的API调用，包含错误处理"""
    try:
        response = requests.post(url, **kwargs)
        
        if response.status_code == 200:
            return response.json()
        else:
            error_data = response.json()
            print(f"API错误: {error_data['error']['message']}")
            return None
            
    except requests.exceptions.ConnectionError:
        print("连接错误: 无法连接到API服务")
        return None
    except requests.exceptions.Timeout:
        print("超时错误: API响应超时")
        return None
    except Exception as e:
        print(f"未知错误: {e}")
        return None

# 使用示例
with open('test_image.jpg', 'rb') as f:
    result = safe_api_call(
        'http://localhost:8000/api/v1/detect/image',
        files={'file': f}
    )
    
    if result:
        print("检测成功!")
    else:
        print("检测失败!")
```

## 性能优化建议

### 1. 文件上传优化

- 推荐图像分辨率: 640x640 到 1280x1280
- 支持的格式: JPG (推荐), PNG, BMP
- 文件大小限制: 10MB
- 批量处理时建议使用异步请求

### 2. 请求频率限制

- 单个客户端: 每秒最多 10 个请求
- 实时统计接口: 建议间隔 ≥ 5 秒
- 历史数据接口: 建议间隔 ≥ 30 秒

### 3. 缓存策略

- 统计数据会缓存 30 秒
- 相同图像的检测结果会缓存 5 分钟
- 建议客户端实现本地缓存

## SDK 和工具

### Python SDK 示例

```python
class HumanDetectionAPI:
    """人体行为检测API客户端"""
    
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def detect_image(self, image_path, **kwargs):
        """图像检测"""
        with open(image_path, 'rb') as f:
            files = {'file': f}
            response = self.session.post(
                f"{self.base_url}/api/v1/detect/image",
                files=files,
                data=kwargs
            )
            return response.json()
    
    def detect_hairnet(self, image_path, **kwargs):
        """发网检测"""
        with open(image_path, 'rb') as f:
            files = {'file': f}
            response = self.session.post(
                f"{self.base_url}/api/v1/detect/hairnet",
                files=files,
                data=kwargs
            )
            return response.json()
    
    def get_realtime_stats(self, **params):
        """获取实时统计"""
        response = self.session.get(
            f"{self.base_url}/api/statistics/realtime",
            params=params
        )
        return response.json()

# 使用示例
api = HumanDetectionAPI()

# 图像检测
result = api.detect_image('test.jpg', confidence_threshold=0.7)
print(f"检测到 {len(result['results']['detections'])} 个目标")

# 发网检测
hairnet_result = api.detect_hairnet('worker.jpg', strict_mode=True)
print(f"合规率: {hairnet_result['results']['summary']['compliance_rate']:.1%}")

# 实时统计
stats = api.get_realtime_stats(region_id='kitchen_area')
print(f"当前人数: {stats['data']['statistics']['person_count']}")
```

## 更新日志

### v1.0.0 (2024-01-01)
- 初始版本发布
- 支持图像检测、发网检测、实时统计功能
- 提供完整的 RESTful API

### 后续版本计划
- v1.1.0: 添加视频流检测支持
- v1.2.0: 增加用户认证和权限管理
- v1.3.0: 支持自定义检测模型

## 技术支持

如有问题或建议，请联系开发团队或查看项目文档。

---

*本文档最后更新: 2024-01-01*