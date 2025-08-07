#!/usr/bin/env python3
"""
使用真实人体图像测试检测功能
"""

import requests
import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image, ImageDraw

def create_realistic_person_image():
    """创建一个更真实的人体图像"""
    # 创建一个更大的图像
    width, height = 800, 600
    img = Image.new('RGB', (width, height), color='lightblue')
    draw = ImageDraw.Draw(img)
    
    # 绘制一个更真实的人形轮廓
    # 人物位置
    center_x = width // 2
    
    # 头部 (椭圆形)
    head_width, head_height = 80, 100
    head_top = 50
    draw.ellipse([
        center_x - head_width//2, head_top,
        center_x + head_width//2, head_top + head_height
    ], fill='peachpuff', outline='black', width=2)
    
    # 颈部
    neck_width = 20
    neck_height = 30
    neck_top = head_top + head_height
    draw.rectangle([
        center_x - neck_width//2, neck_top,
        center_x + neck_width//2, neck_top + neck_height
    ], fill='peachpuff', outline='black', width=2)
    
    # 躯干 (矩形)
    torso_width, torso_height = 120, 200
    torso_top = neck_top + neck_height
    draw.rectangle([
        center_x - torso_width//2, torso_top,
        center_x + torso_width//2, torso_top + torso_height
    ], fill='blue', outline='black', width=2)
    
    # 手臂
    arm_width, arm_length = 25, 150
    arm_top = torso_top + 20
    
    # 左臂
    draw.rectangle([
        center_x - torso_width//2 - arm_width, arm_top,
        center_x - torso_width//2, arm_top + arm_length
    ], fill='peachpuff', outline='black', width=2)
    
    # 右臂
    draw.rectangle([
        center_x + torso_width//2, arm_top,
        center_x + torso_width//2 + arm_width, arm_top + arm_length
    ], fill='peachpuff', outline='black', width=2)
    
    # 腿部
    leg_width, leg_length = 30, 180
    leg_top = torso_top + torso_height
    
    # 左腿
    draw.rectangle([
        center_x - leg_width - 10, leg_top,
        center_x - 10, leg_top + leg_length
    ], fill='darkblue', outline='black', width=2)
    
    # 右腿
    draw.rectangle([
        center_x + 10, leg_top,
        center_x + leg_width + 10, leg_top + leg_length
    ], fill='darkblue', outline='black', width=2)
    
    # 添加一些细节
    # 眼睛
    draw.ellipse([center_x - 25, head_top + 30, center_x - 15, head_top + 40], fill='black')
    draw.ellipse([center_x + 15, head_top + 30, center_x + 25, head_top + 40], fill='black')
    
    # 嘴巴
    draw.arc([center_x - 15, head_top + 60, center_x + 15, head_top + 80], 0, 180, fill='black', width=2)
    
    # 转换为OpenCV格式
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    return img_cv

def download_sample_image():
    """下载一个包含人体的示例图像"""
    try:
        # 使用一个公开的测试图像URL
        url = "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=800&h=600&fit=crop"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            img_array = np.frombuffer(response.content, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            return img
    except Exception as e:
        print(f"下载图像失败: {e}")
    return None

def test_with_image(image, image_name):
    """测试指定图像的检测功能"""
    if image is None:
        print(f"图像 {image_name} 无效")
        return
    
    print(f"\n测试图像: {image_name}")
    print(f"图像尺寸: {image.shape}")
    
    # 保存测试图像
    cv2.imwrite(f'{image_name}.jpg', image)
    print(f"已保存测试图像: {image_name}.jpg")
    
    # 将图像编码为JPEG格式
    _, buffer = cv2.imencode('.jpg', image)
    
    # 准备文件数据
    files = {
        'file': (f'{image_name}.jpg', BytesIO(buffer.tobytes()), 'image/jpeg')
    }
    
    try:
        # 发送请求到综合检测API
        response = requests.post(
            'http://localhost:8000/api/v1/detect/comprehensive',
            files=files,
            timeout=30
        )
        
        print(f"响应状态码: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("检测成功!")
            print(f"检测到的人数: {result.get('person_count', 0)}")
            print(f"发网统计: {result.get('hairnet_stats', {})}")
            print(f"洗手统计: {result.get('handwash_stats', {})}")
            print(f"消毒统计: {result.get('sanitize_stats', {})}")
            print(f"处理时间: {result.get('processing_times', {})}")
            print(f"优化统计: {result.get('optimization_stats', {})}")
            
            # 打印完整响应以便调试
            print(f"\n完整响应: {result}")
            
            # 如果有检测结果图像，保存它
            if 'annotated_image' in result:
                img_data = base64.b64decode(result['annotated_image'])
                with open(f'{image_name}_result.jpg', 'wb') as f:
                    f.write(img_data)
                print(f"检测结果图像已保存为: {image_name}_result.jpg")
        else:
            print(f"检测失败: {response.text}")
            
    except Exception as e:
        print(f"请求失败: {e}")

def main():
    print("=== 真实图像检测测试 ===")
    
    # 测试指定的真实图片
    test_image_path = "tests/fixtures/images/hairnet/7月23日.png"
    try:
        real_img = cv2.imread(test_image_path)
        if real_img is not None:
            test_with_image(real_img, "hairnet_test")
        else:
            print(f"无法加载图像: {test_image_path}")
    except Exception as e:
        print(f"加载图像失败: {e}")
    
    # 测试1: 创建的真实人体图像
    realistic_img = create_realistic_person_image()
    test_with_image(realistic_img, "realistic_person")
    
    # 测试2: 尝试下载真实图像
    print("\n尝试下载真实人体图像...")
    downloaded_img = download_sample_image()
    if downloaded_img is not None:
        test_with_image(downloaded_img, "downloaded_person")
    else:
        print("无法下载真实图像，跳过此测试")
    
    # 测试3: 创建多人图像
    print("\n创建多人图像测试...")
    multi_person_img = create_multi_person_image()
    test_with_image(multi_person_img, "multi_person")

def create_multi_person_image():
    """创建包含多个人的图像"""
    width, height = 1200, 800
    img = Image.new('RGB', (width, height), color='lightgray')
    draw = ImageDraw.Draw(img)
    
    # 绘制三个人
    positions = [(200, 100), (600, 150), (1000, 120)]
    colors = [('blue', 'peachpuff'), ('red', 'peachpuff'), ('green', 'peachpuff')]
    
    for i, (pos, (shirt_color, skin_color)) in enumerate(zip(positions, colors)):
        center_x, top_y = pos
        
        # 头部
        head_size = 60
        draw.ellipse([
            center_x - head_size//2, top_y,
            center_x + head_size//2, top_y + head_size
        ], fill=skin_color, outline='black', width=2)
        
        # 躯干
        torso_width, torso_height = 80, 150
        torso_top = top_y + head_size + 10
        draw.rectangle([
            center_x - torso_width//2, torso_top,
            center_x + torso_width//2, torso_top + torso_height
        ], fill=shirt_color, outline='black', width=2)
        
        # 手臂
        arm_width, arm_length = 20, 100
        arm_top = torso_top + 15
        
        # 左臂
        draw.rectangle([
            center_x - torso_width//2 - arm_width, arm_top,
            center_x - torso_width//2, arm_top + arm_length
        ], fill=skin_color, outline='black', width=2)
        
        # 右臂
        draw.rectangle([
            center_x + torso_width//2, arm_top,
            center_x + torso_width//2 + arm_width, arm_top + arm_length
        ], fill=skin_color, outline='black', width=2)
        
        # 腿部
        leg_width, leg_length = 25, 120
        leg_top = torso_top + torso_height
        
        # 左腿
        draw.rectangle([
            center_x - leg_width - 5, leg_top,
            center_x - 5, leg_top + leg_length
        ], fill='darkblue', outline='black', width=2)
        
        # 右腿
        draw.rectangle([
            center_x + 5, leg_top,
            center_x + leg_width + 5, leg_top + leg_length
        ], fill='darkblue', outline='black', width=2)
    
    # 转换为OpenCV格式
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    return img_cv

if __name__ == "__main__":
    main()