# 模型路径更新报告

**生成时间**: 2025-08-18 16:55:44
**操作类型**: 模型路径更新

## 更新摘要

- **处理文件数**: 148
- **更新文件数**: 33
- **总替换次数**: 207

## 路径映射规则

| 旧路径模式 | 新路径 |
|-----------|--------|
| `yolov8n\.pt` | `models/yolo/yolov8n.pt` |
| `yolov8s\.pt` | `models/yolo/yolov8s.pt` |
| `yolov8m\.pt` | `models/yolo/yolov8m.pt` |
| `yolov8l\.pt` | `models/yolo/yolov8l.pt` |
| `models/hairnet_detection\.pt` | `models/hairnet_detection/hairnet_detection.pt` |
| `hairnet_detection\.pt` | `models/hairnet_detection/hairnet_detection.pt` |
| `models/hairnet_model/weights/best\.pt` | `models/hairnet_model/weights/best.pt` |
| `models/hairnet_model/weights/last\.pt` | `models/hairnet_model/weights/last.pt` |
| `models/yolov8n\.pt` | `models/yolo/yolov8n.pt` |
| `models/yolov8s\.pt` | `models/yolo/yolov8s.pt` |
| `models/yolov8m\.pt` | `models/yolo/yolov8m.pt` |
| `models/yolov8l\.pt` | `models/yolo/yolov8l.pt` |

## 详细更新信息

- `CONTRIBUTING.md`: 1 个替换
- `config/unified_params.yaml`: 1 个替换
- `config/default.yaml`: 3 个替换
- `development/start_dev.sh`: 2 个替换
- `development/setup_dev_env.sh`: 4 个替换
- `training/start_training.sh`: 6 个替换
- `training/start_training.ps1`: 6 个替换
- `training/README.md`: 3 个替换
- `models/README.md`: 23 个替换
- `models/hairnet_model/args.yaml`: 2 个替换
- `docs/README_YOLO_INTEGRATION.md`: 20 个替换
- `docs/README_TRAINING.md`: 14 个替换
- `docs/敏捷迭代执行方案.md`: 1 个替换
- `docs/README_DEV_SCRIPTS.md`: 3 个替换
- `docs/README_HAIRNET_DETECTION.md`: 5 个替换
- `testing/README.md`: 5 个替换
- `testing/start_testing.sh`: 4 个替换
- `deployment/deploy_win.bat`: 4 个替换
- `examples/example_usage.py`: 6 个替换
- `examples/integrate_yolo_detector.py`: 4 个替换
- `examples/use_yolo_hairnet_detector.py`: 6 个替换
- `scripts/optimize_human_detector.py`: 9 个替换
- `scripts/update_model_paths.py`: 19 个替换
- `scripts/setup_dev.sh`: 5 个替换
- `scripts/check_dev_env.py`: 6 个替换
- `scripts/train_hairnet_model.py`: 3 个替换
- `scripts/compare_yolo_models.py`: 9 个替换
- `reports/model_protection_update.md`: 7 个替换
- `reports/model_files_organization_report.md`: 17 个替换
- `src/core/yolo_hairnet_detector.py`: 4 个替换
- `src/core/hairnet_detection_factory.py`: 2 个替换
- `src/config/model_config.py`: 2 个替换
- `src/config/unified_params.py`: 1 个替换

## 注意事项

1. 所有YOLO模型现在位于 `models/yolo/` 目录下
2. 发网检测模型位于 `models/hairnet_detection/` 目录下
3. 用户训练的模型仍在 `models/hairnet_model/weights/` 目录下（受保护）
4. 请确保所有模型文件已正确移动到新位置
5. 如有配置文件缓存，请清除后重新加载
