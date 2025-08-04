#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
准备Roboflow数据集

处理从Roboflow下载的数据集，确保其格式正确并准备用于YOLOv8训练
"""

import os
import sys
import argparse
import logging
import shutil
import yaml
from pathlib import Path
import zipfile

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('dataset_preparation')

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='准备Roboflow数据集')
    parser.add_argument('--input', type=str, required=True, 
                        help='Roboflow数据集路径（ZIP文件或解压后的目录）')
    parser.add_argument('--output', type=str, default='./datasets/hairnet',
                        help='输出数据集目录')
    parser.add_argument('--format', type=str, default='yolov8',
                        choices=['yolov8', 'yolov5', 'coco'],
                        help='数据集格式')
    parser.add_argument('--extract-only', action='store_true',
                        help='仅解压数据集，不进行处理')
    
    return parser.parse_args()

def extract_dataset(input_path, output_path):
    """解压数据集"""
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    # 如果输入是目录，则假设已经解压
    if input_path.is_dir():
        logger.info(f'输入是目录，假设已经解压: {input_path}')
        # 如果输入和输出不同，则复制目录
        if str(input_path) != str(output_path):
            if output_path.exists():
                logger.warning(f'输出目录已存在，将被覆盖: {output_path}')
                shutil.rmtree(output_path)
            shutil.copytree(input_path, output_path)
            logger.info(f'已复制数据集到: {output_path}')
        return output_path
    
    # 如果输入是ZIP文件，则解压
    if input_path.suffix.lower() == '.zip':
        logger.info(f'解压数据集: {input_path}')
        output_path.mkdir(parents=True, exist_ok=True)
        
        with zipfile.ZipFile(input_path, 'r') as zip_ref:
            zip_ref.extractall(output_path)
        
        logger.info(f'数据集已解压到: {output_path}')
        return output_path
    
    raise ValueError(f'不支持的输入格式: {input_path}')

def validate_dataset_structure(dataset_path, format='yolov8'):
    """验证数据集结构"""
    dataset_path = Path(dataset_path)
    
    # 检查基本目录结构
    train_dir = dataset_path / 'train'
    valid_dir = dataset_path / 'valid'
    test_dir = dataset_path / 'test'
    
    # 检查train目录
    if not train_dir.exists():
        logger.warning(f'训练集目录不存在: {train_dir}')
        # 尝试查找其他可能的目录名
        alt_train_dirs = [dataset_path / 'training', dataset_path / 'train_data']
        for alt_dir in alt_train_dirs:
            if alt_dir.exists():
                logger.info(f'找到替代训练集目录: {alt_dir}')
                train_dir = alt_dir
                break
    
    # 检查valid目录
    if not valid_dir.exists():
        logger.warning(f'验证集目录不存在: {valid_dir}')
        # 尝试查找其他可能的目录名
        alt_valid_dirs = [dataset_path / 'validation', dataset_path / 'val']
        for alt_dir in alt_valid_dirs:
            if alt_dir.exists():
                logger.info(f'找到替代验证集目录: {alt_dir}')
                valid_dir = alt_dir
                break
    
    # 检查test目录（可选）
    if not test_dir.exists():
        logger.info(f'测试集目录不存在: {test_dir} (可选)')
    
    # 根据格式检查子目录结构
    if format in ['yolov8', 'yolov5']:
        # 检查images和labels目录
        for data_dir in [d for d in [train_dir, valid_dir, test_dir] if d.exists()]:
            images_dir = data_dir / 'images'
            labels_dir = data_dir / 'labels'
            
            if not images_dir.exists():
                logger.warning(f'图像目录不存在: {images_dir}')
                # 尝试查找图像文件
                image_files = list(data_dir.glob('*.jpg')) + list(data_dir.glob('*.png'))
                if image_files:
                    logger.info(f'在 {data_dir} 中找到 {len(image_files)} 个图像文件')
                    # 创建images目录并移动文件
                    images_dir.mkdir(parents=True, exist_ok=True)
                    for img_file in image_files:
                        shutil.move(img_file, images_dir / img_file.name)
                    logger.info(f'已移动图像文件到: {images_dir}')
            
            if not labels_dir.exists():
                logger.warning(f'标签目录不存在: {labels_dir}')
                # 尝试查找标签文件
                label_files = list(data_dir.glob('*.txt'))
                if label_files:
                    logger.info(f'在 {data_dir} 中找到 {len(label_files)} 个标签文件')
                    # 创建labels目录并移动文件
                    labels_dir.mkdir(parents=True, exist_ok=True)
                    for lbl_file in label_files:
                        # 跳过data.yaml等配置文件
                        if lbl_file.name in ['data.yaml', 'classes.txt']:
                            continue
                        shutil.move(lbl_file, labels_dir / lbl_file.name)
                    logger.info(f'已移动标签文件到: {labels_dir}')
    
    # 检查数据集文件数量
    dataset_stats = {}
    for split, split_dir in [('train', train_dir), ('valid', valid_dir), ('test', test_dir)]:
        if not split_dir.exists():
            continue
        
        images_dir = split_dir / 'images'
        labels_dir = split_dir / 'labels'
        
        if images_dir.exists():
            image_count = len(list(images_dir.glob('*.jpg'))) + len(list(images_dir.glob('*.png')))
        else:
            image_count = 0
        
        if labels_dir.exists():
            label_count = len(list(labels_dir.glob('*.txt')))
        else:
            label_count = 0
        
        dataset_stats[split] = {'images': image_count, 'labels': label_count}
    
    logger.info('数据集统计:')
    for split, stats in dataset_stats.items():
        logger.info(f'  {split}: {stats["images"]} 图像, {stats["labels"]} 标签')
    
    return dataset_stats

def create_data_yaml(dataset_path, format='yolov8'):
    """创建数据集YAML配置文件"""
    dataset_path = Path(dataset_path)
    yaml_path = dataset_path / 'data.yaml'
    
    # 检查是否已存在data.yaml文件
    if yaml_path.exists():
        logger.info(f'使用现有的数据集YAML配置: {yaml_path}')
        with open(yaml_path, 'r') as f:
            data_yaml = yaml.safe_load(f)
        return data_yaml
    
    # 查找类别信息
    classes = []
    classes_file = dataset_path / 'classes.txt'
    
    if classes_file.exists():
        with open(classes_file, 'r') as f:
            classes = [line.strip() for line in f if line.strip()]
    
    # 如果没有找到类别文件，尝试从标签文件推断
    if not classes:
        train_dir = dataset_path / 'train'
        if train_dir.exists():
            labels_dir = train_dir / 'labels'
            if labels_dir.exists():
                label_files = list(labels_dir.glob('*.txt'))
                if label_files:
                    # 读取第一个标签文件，获取类别ID
                    with open(label_files[0], 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if parts:
                                class_id = int(parts[0])
                                # 确保类别列表足够长
                                while len(classes) <= class_id:
                                    classes.append(f'class_{len(classes)}')
    
    # 如果仍然没有类别，假设只有一个类别：hairnet
    if not classes:
        classes = ['hairnet']
    
    # 查找数据集路径
    train_images = dataset_path / 'train' / 'images'
    valid_images = dataset_path / 'valid' / 'images'
    test_images = dataset_path / 'test' / 'images'
    
    # 创建YAML配置
    data_yaml = {
        'path': str(dataset_path),
        'train': str(train_images) if train_images.exists() else '',
        'val': str(valid_images) if valid_images.exists() else '',
        'test': str(test_images) if test_images.exists() else '',
        'nc': len(classes),
        'names': classes
    }
    
    # 写入YAML文件
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, sort_keys=False)
    
    logger.info(f'创建数据集YAML配置: {yaml_path}')
    logger.info(f'类别: {classes}')
    
    return data_yaml

def main():
    """主函数"""
    args = parse_args()
    
    logger.info('='*50)
    logger.info('Roboflow数据集准备开始')
    logger.info(f'输入路径: {args.input}')
    logger.info(f'输出路径: {args.output}')
    logger.info(f'数据集格式: {args.format}')
    logger.info('='*50)
    
    try:
        # 解压数据集
        dataset_path = extract_dataset(args.input, args.output)
        
        # 如果只需解压，则结束
        if args.extract_only:
            logger.info('仅解压模式，处理完成')
            return
        
        # 验证数据集结构
        validate_dataset_structure(dataset_path, args.format)
        
        # 创建数据集YAML配置
        data_yaml = create_data_yaml(dataset_path, args.format)
        
        logger.info('数据集准备完成!')
        logger.info(f'数据集路径: {dataset_path}')
        logger.info(f'配置文件: {dataset_path}/data.yaml')
        logger.info(f'类别数量: {data_yaml["nc"]}')
        logger.info(f'类别名称: {data_yaml["names"]}')
        
        # 打印训练命令示例
        logger.info('\n训练命令示例:')
        logger.info(f'python scripts/train_hairnet_model.py --data {dataset_path}/data.yaml --epochs 100 --batch-size 16 --img-size 640 --model yolov8n.pt --name hairnet_model')
        
    except Exception as e:
        logger.error(f'数据集准备过程中发生错误: {e}', exc_info=True)
        sys.exit(1)

if __name__ == '__main__':
    main()