#!/usr/bin/env python
"""
使用测试图片目录中的图片测试发网检测模型

用法:
    python test_with_fixtures.py
"""

import os
import sys
from pathlib import Path
import glob


def find_test_images():
    """查找测试图片"""
    test_dirs = [
        "tests/fixtures/images/hairnet",
        "tests/fixtures/images/person", 
        "datasets/hairnet/valid/images",
        "datasets/hairnet/train/images",
        "datasets/hairnet/test/images"
    ]
    
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.webp']
    found_images = []
    
    for test_dir in test_dirs:
        if os.path.exists(test_dir):
            for ext in image_extensions:
                pattern = os.path.join(test_dir, ext)
                found_images.extend(glob.glob(pattern))
                # 也搜索大写扩展名
                pattern_upper = os.path.join(test_dir, ext.upper())
                found_images.extend(glob.glob(pattern_upper))
    
    return found_images


def test_model_with_images(model_path, images):
    """使用找到的图片测试模型"""
    if not images:
        print("未找到测试图片！")
        print("\n请将测试图片放入以下任一目录：")
        print("- tests/fixtures/images/hairnet/")
        print("- tests/fixtures/images/person/")
        print("- datasets/hairnet/valid/images/")
        print("\n支持的图片格式：jpg, jpeg, png, bmp, tiff, webp")
        return False
    
    print(f"找到 {len(images)} 张测试图片：")
    for i, img in enumerate(images[:5]):  # 只显示前5张
        print(f"  {i+1}. {img}")
    if len(images) > 5:
        print(f"  ... 还有 {len(images)-5} 张图片")
    
    print(f"\n使用模型 {model_path} 进行测试...")
    
    # 导入并运行测试
    try:
        from ultralytics import YOLO
        
        # 加载模型
        model = YOLO(model_path)
        
        # 对每张图片进行推理
        for img_path in images:
            print(f"\n正在处理: {os.path.basename(img_path)}")
            results = model.predict(
                source=img_path,
                conf=0.25,
                save=True,
                project="runs/detect",
                name="test_fixtures",
                exist_ok=True,
                show=False  # 设为True可显示结果窗口
            )
            
            # 显示检测结果
            if results and len(results) > 0:
                result = results[0]
                if result.boxes is not None and len(result.boxes) > 0:
                    print(f"  检测到 {len(result.boxes)} 个目标：")
                    for box in result.boxes:
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        class_name = model.names[class_id]
                        print(f"    - {class_name}: {confidence:.2f}")
                else:
                    print("  未检测到目标")
        
        print(f"\n测试完成！结果保存在: runs/detect/test_fixtures/")
        return True
        
    except ImportError:
        print("错误: 未安装ultralytics库")
        print("请运行: pip install ultralytics")
        return False
    except Exception as e:
        print(f"测试过程中出错: {e}")
        return False


def main():
    # 检查模型文件
    model_path = "models/hairnet_detection.pt"
    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在: {model_path}")
        print("请先运行训练脚本生成模型文件")
        return 1
    
    print("=== 发网检测模型测试 ===")
    print(f"模型路径: {model_path}")
    
    # 查找测试图片
    print("\n正在搜索测试图片...")
    test_images = find_test_images()
    
    # 测试模型
    success = test_model_with_images(model_path, test_images)
    
    if success:
        print("\n测试成功完成！")
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())