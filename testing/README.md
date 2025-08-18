# 测试脚本目录

本目录包含模型测试和API测试相关的脚本文件。

## 脚本说明

### start_testing.sh
- **功能**: 发网检测模型测试启动脚本
- **主要功能**:
  - 自动检查ultralytics依赖
  - 加载训练好的模型进行推理测试
  - 支持多种输入源（图片、视频、摄像头）
  - 可配置检测参数
  - 支持结果可视化和保存
- **使用方法**:
  ```bash
  ./start_testing.sh [选项]
  ```
- **支持参数**:
  - `--weights FILE`: 模型权重文件（默认models/hairnet_detection/models/hairnet_detection/hairnet_detection.pt）
  - `--source PATH`: 输入源（图片/视频路径或摄像头ID）
  - `--conf-thres N`: 置信度阈值（默认0.25）
  - `--iou-thres N`: IoU阈值（默认0.45）
  - `--device STR`: 推理设备（如cuda:0或cpu）
  - `--view-img`: 显示检测结果
  - `--save-txt`: 保存检测结果为文本
  - `--nosave`: 不保存检测结果图片

### test_api_curl.sh
- **功能**: API接口测试脚本
- **主要功能**:
  - 测试API服务健康状态
  - 测试发网检测API接口
  - 使用测试图片验证检测功能
  - 查询检测历史记录
- **使用方法**:
  ```bash
  ./test_api_curl.sh
  ```
- **测试内容**:
  - 健康检查: `GET /health`
  - API信息: `GET /api/v1/info`
  - 发网检测: `POST /api/v1/detect/hairnet`
  - 历史查询: `GET /api/statistics/history`

## 使用前准备

### 1. 模型文件
确保有可用的训练模型：
- `models/hairnet_detection/models/hairnet_detection/hairnet_detection.pt` - 主模型文件
- `models/hairnet_model/weights/best.pt` - 训练生成的最佳模型

### 2. 测试数据
确保测试图片存在：
- `tests/fixtures/images/person/test_person.png`
- `tests/fixtures/images/hairnet/7月23日.png`

### 3. API服务
对于API测试，确保服务正在运行：
```bash
python src/api/app.py
```

## 测试示例

### 模型推理测试
```bash
# 基础测试
./testing/start_testing.sh --source tests/fixtures/images/hairnet/

# 实时摄像头测试
./testing/start_testing.sh --source 0 --view-img

# 高精度测试
./testing/start_testing.sh --source test.jpg --conf-thres 0.5 --view-img
```

### API接口测试
```bash
# 运行完整API测试
./testing/test_api_curl.sh

# 单独测试健康检查
curl http://localhost:8000/health
```

## 测试结果

### 模型测试输出
- 检测结果图片保存在 `runs/detect/` 目录
- 文本结果保存为 `.txt` 文件（如启用）
- 控制台显示检测统计信息

### API测试输出
- JSON格式的检测结果
- 置信度和边界框信息
- 检测历史统计

## 故障排除

### 常见问题
1. **模型文件不存在**
   - 确保已完成模型训练或下载预训练模型
   - 检查模型文件路径是否正确

2. **API连接失败**
   - 确认API服务正在运行
   - 检查端口8000是否被占用
   - 验证防火墙设置

3. **测试图片不存在**
   - 检查测试数据是否完整
   - 使用自己的图片进行测试

## 相关文档

- [模型测试文档](../docs/README_WEB_TESTING.md)
- [API集成文档](../docs/README_YOLO_INTEGRATION.md)
- [项目主文档](../README.md)
