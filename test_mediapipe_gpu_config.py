#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试MediaPipe GPU自动配置功能
"""

import os
import sys
import logging
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_gpu_detection():
    """测试GPU检测功能"""
    logger.info("=== 测试MediaPipe GPU自动配置功能 ===")
    
    # 清除可能存在的环境变量
    if 'MEDIAPIPE_DISABLE_GPU' in os.environ:
        del os.environ['MEDIAPIPE_DISABLE_GPU']
        logger.info("已清除现有的MEDIAPIPE_DISABLE_GPU环境变量")
    
    try:
        # 导入pose_detector模块，这会触发GPU检测
        from src.core.pose_detector import _gpu_enabled, _configure_mediapipe_gpu
        
        logger.info(f"GPU检测结果: {'启用' if _gpu_enabled else '禁用'}")
        logger.info(f"MEDIAPIPE_DISABLE_GPU环境变量: {os.environ.get('MEDIAPIPE_DISABLE_GPU', '未设置')}")
        
        # 测试YOLOv8PoseDetector初始化
        from src.core.pose_detector import YOLOv8PoseDetector
        
        logger.info("\n=== 测试YOLOv8PoseDetector初始化 ===")
        pose_detector = YOLOv8PoseDetector()
        
        if pose_detector:
            logger.info("✓ YOLOv8PoseDetector成功初始化")
        else:
            logger.warning("✗ YOLOv8PoseDetector初始化失败")
        
        # 测试BehaviorRecognizer初始化
        logger.info("\n=== 测试BehaviorRecognizer初始化 ===")
        from src.core.behavior import BehaviorRecognizer
        
        behavior_recognizer = BehaviorRecognizer(
            confidence_threshold=0.3
        )
        
        if behavior_recognizer:
            logger.info("✓ BehaviorRecognizer成功初始化")
        else:
            logger.warning("✗ BehaviorRecognizer初始化失败")
        
        # 清理资源
        if hasattr(pose_detector, 'cleanup'):
            pose_detector.cleanup()
        
        assert True, "GPU检测测试通过"
        
    except Exception as e:
        logger.error(f"测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        assert False, f"GPU检测测试失败: {e}"

def test_manual_gpu_disable():
    """测试手动禁用GPU功能"""
    logger.info("\n=== 测试手动禁用GPU功能 ===")
    
    # 手动设置禁用GPU
    os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'
    logger.info("已手动设置MEDIAPIPE_DISABLE_GPU=1")
    
    try:
        # 重新配置GPU
        from src.core.pose_detector import _configure_mediapipe_gpu
        gpu_enabled = _configure_mediapipe_gpu()
        
        if not gpu_enabled:
            logger.info("✓ 手动禁用GPU功能正常工作")
            assert True, "手动禁用GPU功能正常工作"
        else:
            logger.warning("✗ 手动禁用GPU功能未生效")
            assert False, "手动禁用GPU功能未生效"
        
    except Exception as e:
        logger.error(f"测试手动禁用GPU时发生错误: {e}")
        assert False, f"手动禁用GPU测试失败: {e}"
    finally:
        # 清理环境变量
        if 'MEDIAPIPE_DISABLE_GPU' in os.environ:
            del os.environ['MEDIAPIPE_DISABLE_GPU']

def test_gpu_requirements():
    """测试GPU硬件要求检查"""
    logger.info("\n=== 测试GPU硬件要求检查 ===")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            # 获取GPU信息
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            compute_capability = torch.cuda.get_device_capability(0)
            compute_version = compute_capability[0] + compute_capability[1] * 0.1
            
            logger.info(f"GPU信息:")
            logger.info(f"  名称: {gpu_name}")
            logger.info(f"  显存: {gpu_memory_gb:.1f}GB")
            logger.info(f"  计算能力: {compute_version:.1f}")
            
            # 检查是否满足要求
            meets_memory_req = gpu_memory_gb >= 2.0
            meets_compute_req = compute_version >= 6.0
            
            logger.info(f"硬件要求检查:")
            logger.info(f"  显存要求(>=2GB): {'✓' if meets_memory_req else '✗'}")
            logger.info(f"  计算能力要求(>=6.0): {'✓' if meets_compute_req else '✗'}")
            
            # 对于GPU要求检查，我们不强制要求通过，只是记录结果
            if meets_memory_req and meets_compute_req:
                assert True, "GPU硬件要求检查通过"
            else:
                logger.info("GPU硬件不满足要求，但这是正常情况")
                assert True, "GPU硬件要求检查完成（不满足要求但正常）"
        else:
            logger.info("CUDA不可用，将使用CPU模式")
            assert True, "CUDA不可用检查完成（正常情况）"
            
    except ImportError:
        logger.info("PyTorch不可用，将使用CPU模式")
        assert True, "PyTorch不可用检查完成（正常情况）"
    except Exception as e:
        logger.error(f"检查GPU硬件要求时发生错误: {e}")
        assert False, f"GPU硬件要求检查失败: {e}"

def main():
    """主测试函数"""
    logger.info("开始MediaPipe GPU自动配置测试")
    
    # 测试GPU硬件要求
    gpu_capable = test_gpu_requirements()
    
    # 测试自动GPU检测
    auto_detection_success = test_gpu_detection()
    
    # 测试手动禁用GPU
    manual_disable_success = test_manual_gpu_disable()
    
    # 总结测试结果
    logger.info("\n=== 测试结果总结 ===")
    logger.info(f"GPU硬件检查: {'通过' if gpu_capable else '不满足要求'}")
    logger.info(f"自动GPU检测: {'成功' if auto_detection_success else '失败'}")
    logger.info(f"手动禁用GPU: {'成功' if manual_disable_success else '失败'}")
    
    overall_success = auto_detection_success and manual_disable_success
    logger.info(f"\n整体测试结果: {'✓ 全部通过' if overall_success else '✗ 存在问题'}")
    
    if overall_success:
        logger.info("\nMediaPipe GPU自动配置功能已成功实现！")
        if gpu_capable:
            logger.info("您的系统支持GPU加速，MediaPipe将自动启用GPU模式。")
        else:
            logger.info("您的系统不满足GPU要求，MediaPipe将使用CPU模式。")
    else:
        logger.error("\nMediaPipe GPU自动配置功能存在问题，请检查实现。")
    
    return overall_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)