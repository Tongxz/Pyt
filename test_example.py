#!/usr/bin/env python
"""
发网检测模型测试示例

这个脚本展示如何使用训练好的模型测试图片
"""

import os
import sys

def test_single_image(image_path):
    """测试单张图片"""
    model_path = "models/hairnet_detection.pt"
    
    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在: {model_path}")
        return False
    
    if not os.path.exists(image_path):
        print(f"错误: 图片文件不存在: {image_path}")
        return False
    
    try:
        from ultralytics import YOLO
        
        print(f"加载模型: {model_path}")
        model = YOLO(model_path)
        
        print(f"正在检测图片: {image_path}")
        results = model.predict(
            source=image_path,
            conf=0.25,  # 置信度阈值
            save=True,  # 保存结果图片
            show=False, # 设为True可显示结果窗口
            project="runs/detect",
            name="single_test",
            exist_ok=True
        )
        
        # 显示检测结果
        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None and len(result.boxes) > 0:
                print(f"\n检测结果:")
                print(f"检测到 {len(result.boxes)} 个目标：")
                
                # 按类别统计
                class_counts = {}
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    class_name = model.names[class_id]
                    
                    if class_name not in class_counts:
                        class_counts[class_name] = []
                    class_counts[class_name].append(confidence)
                    
                    print(f"  - {class_name}: {confidence:.2f}")
                
                print(f"\n统计结果:")
                for class_name, confidences in class_counts.items():
                    count = len(confidences)
                    avg_conf = sum(confidences) / count
                    print(f"  {class_name}: {count} 个 (平均置信度: {avg_conf:.2f})")
                    
            else:
                print("未检测到任何目标")
        
        print(f"\n结果已保存到: runs/detect/single_test/")
        return True
        
    except ImportError:
        print("错误: 未安装ultralytics库")
        return False
    except Exception as e:
        print(f"检测过程中出错: {e}")
        return False

def main():
    print("=== 发网检测模型测试示例 ===")
    
    # 查找一张测试图片
    test_dirs = [
        "tests/fixtures/images/hairnet",
        "tests/fixtures/images/person",
        "datasets/hairnet/valid/images",
        "datasets/hairnet/train/images"
    ]
    
    test_image = None
    for test_dir in test_dirs:
        if os.path.exists(test_dir):
            for file in os.listdir(test_dir):
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    test_image = os.path.join(test_dir, file)
                    break
        if test_image:
            break
    
    if test_image:
        print(f"找到测试图片: {test_image}")
        success = test_single_image(test_image)
        if success:
            print("\n测试完成！")
            print("\n使用方法:")
            print("1. 直接运行此脚本测试找到的图片")
            print("2. 或使用以下命令测试指定图片:")
            print("   python test_hairnet_model.py --weights models/hairnet_detection.pt --source your_image.jpg --view-img")
            return 0
        else:
            return 1
    else:
        print("未找到测试图片")
        print("\n请将测试图片放入以下目录之一:")
        for test_dir in test_dirs:
            print(f"  - {test_dir}/")
        print("\n然后使用以下命令测试:")
        print("python test_hairnet_model.py --weights models/hairnet_detection.pt --source your_image.jpg --view-img")
        return 1

if __name__ == "__main__":
    sys.exit(main())