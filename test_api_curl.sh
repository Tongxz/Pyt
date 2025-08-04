#!/bin/bash

# 测试发网检测API的简单脚本

echo "=== 测试健康检查 ==="
curl -s http://localhost:8000/health

echo "\n=== 测试API信息 ==="
curl -s http://localhost:8000/api/v1/info

echo "\n=== 测试发网检测API (使用测试图片) ==="
curl -s -X POST -F "file=@tests/fixtures/images/person/test_person.png" http://localhost:8000/api/v1/detect/hairnet

echo "\n=== 测试发网检测API (使用发网图片) ==="
curl -s -X POST -F "file=@tests/fixtures/images/hairnet/7月23日.png" http://localhost:8000/api/v1/detect/hairnet

echo "\n=== 测试检测历史查询 ==="
curl -s http://localhost:8000/api/statistics/history

echo "\n测试完成！"