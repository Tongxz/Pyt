#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
人体行为检测系统主入口文件
Human Behavior Detection System Main Entry Point

作者: AI Assistant
版本: 1.0.0
创建时间: 2024
"""

import sys
import argparse
from pathlib import Path

# 添加src目录到Python路径
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from utils.logger import setup_project_logger
from config import Settings


def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(
        description="人体行为检测系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python main.py --mode detection --source 0                    # 使用摄像头进行检测
  python main.py --mode detection --source video.mp4           # 使用视频文件进行检测
  python main.py --mode api --port 5000                        # 启动API服务
  python main.py --mode training --config config/train.yaml    # 训练模型
        """
    )
    
    parser.add_argument(
        '--mode', 
        choices=['detection', 'api', 'training', 'demo'],
        default='detection',
        help='运行模式 (默认: detection)'
    )
    
    parser.add_argument(
        '--source',
        type=str,
        default='0',
        help='输入源: 摄像头索引(0,1...) 或 视频文件路径 (默认: 0)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config/default.yaml',
        help='配置文件路径 (默认: config/default.yaml)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='输出目录路径'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=5000,
        help='API服务端口 (默认: 5000)'
    )
    
    parser.add_argument(
        '--host',
        type=str,
        default='0.0.0.0',
        help='API服务主机 (默认: 0.0.0.0)'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='启用调试模式'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='日志级别 (默认: INFO)'
    )
    
    args = parser.parse_args()
    
    # 设置日志
    logger = setup_project_logger()
    if args.debug:
        logger.setLevel('DEBUG')
    else:
        logger.setLevel(args.log_level)
    
    logger.info("="*50)
    logger.info("人体行为检测系统启动")
    logger.info(f"运行模式: {args.mode}")
    logger.info("="*50)
    
    try:
        if args.mode == 'detection':
            run_detection(args, logger)
        elif args.mode == 'api':
            run_api_server(args, logger)
        elif args.mode == 'training':
            run_training(args, logger)
        elif args.mode == 'demo':
            run_demo(args, logger)
        else:
            logger.error(f"未知的运行模式: {args.mode}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("用户中断程序")
    except Exception as e:
        logger.error(f"程序运行出错: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)
    finally:
        logger.info("程序结束")


def run_detection(args, logger):
    """
    运行检测模式
    """
    logger.info(f"开始检测，输入源: {args.source}")
    
    # TODO: 实现检测逻辑
    # from core.detector import HumanDetector
    # from core.tracker import MultiObjectTracker
    # from core.behavior import BehaviorRecognizer
    
    logger.info("检测模式暂未实现，请等待后续版本")


def run_api_server(args, logger):
    """
    运行API服务器
    """
    logger.info(f"启动API服务器: {args.host}:{args.port}")
    
    try:
        from api.app import create_app
        app = create_app()
        app.run(
            host=args.host,
            port=args.port,
            debug=args.debug
        )
    except ImportError as e:
        logger.error(f"无法导入API模块: {e}")
        logger.info("请确保已安装Flask相关依赖: pip install Flask Flask-CORS")


def run_training(args, logger):
    """
    运行训练模式
    """
    logger.info(f"开始训练，配置文件: {args.config}")
    
    # TODO: 实现训练逻辑
    logger.info("训练模式暂未实现，请等待后续版本")


def run_demo(args, logger):
    """
    运行演示模式
    """
    logger.info("启动演示模式")
    
    # TODO: 实现演示逻辑
    logger.info("演示模式暂未实现，请等待后续版本")


if __name__ == '__main__':
    main()