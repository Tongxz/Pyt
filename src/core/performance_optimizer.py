import logging
import time
import threading
from typing import Dict, List, Optional, Any, Callable, Tuple
from concurrent.futures import ThreadPoolExecutor, Future
from queue import Queue, Empty
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import cv2
torch_available = True
try:
    import torch
    import torch.nn as nn
    from torch.quantization import quantize_dynamic
except ImportError:
    torch_available = False
    logger = logging.getLogger(__name__)
    logger.warning("PyTorch not available, some optimization features will be disabled")

logger = logging.getLogger(__name__)


@dataclass
class ProcessingTask:
    """处理任务数据类"""
    task_id: str
    task_type: str  # 'detection', 'analysis', 'recognition'
    data: Any
    callback: Optional[Callable] = None
    priority: int = 0  # 优先级，数字越大优先级越高
    created_at: float = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()


class FrameBuffer:
    """帧缓冲区
    
    用于高效管理视频帧的缓存和处理
    """
    
    def __init__(self, max_size: int = 30, auto_resize: bool = True):
        """
        初始化帧缓冲区
        
        Args:
            max_size: 最大缓存帧数
            auto_resize: 是否自动调整帧大小
        """
        self.max_size = max_size
        self.auto_resize = auto_resize
        self.buffer = Queue(maxsize=max_size)
        self.lock = threading.Lock()
        
        # 统计信息
        self.total_frames = 0
        self.dropped_frames = 0
        self.processing_times = []
        
        # 自动调整参数
        self.target_size = (640, 480)  # 目标分辨率
        self.quality_factor = 0.8  # JPEG压缩质量
    
    def add_frame(self, frame: np.ndarray, timestamp: float = None) -> bool:
        """添加帧到缓冲区
        
        Args:
            frame: 视频帧
            timestamp: 时间戳
            
        Returns:
            是否成功添加
        """
        if timestamp is None:
            timestamp = time.time()
        
        # 预处理帧
        processed_frame = self._preprocess_frame(frame)
        
        frame_data = {
            'frame': processed_frame,
            'timestamp': timestamp,
            'original_shape': frame.shape
        }
        
        try:
            with self.lock:
                if self.buffer.full():
                    # 移除最旧的帧
                    try:
                        self.buffer.get_nowait()
                        self.dropped_frames += 1
                    except Empty:
                        pass
                
                self.buffer.put_nowait(frame_data)
                self.total_frames += 1
                return True
        except Exception as e:
            logger.error(f"Failed to add frame to buffer: {e}")
            return False
    
    def get_frame(self, timeout: float = 0.1) -> Optional[Dict[str, Any]]:
        """从缓冲区获取帧
        
        Args:
            timeout: 超时时间
            
        Returns:
            帧数据或None
        """
        try:
            return self.buffer.get(timeout=timeout)
        except Empty:
            return None
    
    def get_latest_frame(self) -> Optional[Dict[str, Any]]:
        """获取最新的帧"""
        latest_frame = None
        
        # 清空缓冲区，只保留最新帧
        while True:
            try:
                frame_data = self.buffer.get_nowait()
                if latest_frame is not None:
                    self.dropped_frames += 1
                latest_frame = frame_data
            except Empty:
                break
        
        return latest_frame
    
    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """预处理帧"""
        if not self.auto_resize:
            return frame
        
        # 调整大小
        if frame.shape[:2] != self.target_size[::-1]:
            frame = cv2.resize(frame, self.target_size)
        
        return frame
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓冲区统计信息"""
        with self.lock:
            return {
                'buffer_size': self.buffer.qsize(),
                'max_size': self.max_size,
                'total_frames': self.total_frames,
                'dropped_frames': self.dropped_frames,
                'drop_rate': self.dropped_frames / max(1, self.total_frames),
                'avg_processing_time': np.mean(self.processing_times) if self.processing_times else 0.0
            }
    
    def clear(self):
        """清空缓冲区"""
        with self.lock:
            while not self.buffer.empty():
                try:
                    self.buffer.get_nowait()
                except Empty:
                    break


class ModelQuantizer:
    """模型量化器
    
    用于优化深度学习模型的推理性能
    """
    
    def __init__(self):
        self.quantized_models = {}
        self.quantization_configs = {
            'dynamic': {
                'qconfig_spec': {
                    nn.Linear: torch.quantization.default_dynamic_qconfig,
                    nn.LSTM: torch.quantization.default_dynamic_qconfig,
                    nn.GRU: torch.quantization.default_dynamic_qconfig,
                }
            },
            'static': {
                'backend': 'fbgemm',  # 'fbgemm' for x86, 'qnnpack' for ARM
            }
        }
    
    def quantize_model_dynamic(self, model: 'nn.Module', model_name: str) -> 'nn.Module':
        """动态量化模型
        
        Args:
            model: PyTorch模型
            model_name: 模型名称
            
        Returns:
            量化后的模型
        """
        if not torch_available:
            logger.warning("PyTorch not available, returning original model")
            return model
        
        try:
            # 设置为评估模式
            model.eval()
            
            # 动态量化
            quantized_model = quantize_dynamic(
                model,
                qconfig_spec=self.quantization_configs['dynamic']['qconfig_spec'],
                dtype=torch.qint8
            )
            
            self.quantized_models[model_name] = quantized_model
            
            # 计算模型大小减少
            original_size = self._get_model_size(model)
            quantized_size = self._get_model_size(quantized_model)
            compression_ratio = original_size / quantized_size if quantized_size > 0 else 1.0
            
            logger.info(f"Model {model_name} quantized: {original_size:.2f}MB -> {quantized_size:.2f}MB "
                       f"(compression ratio: {compression_ratio:.2f}x)")
            
            return quantized_model
            
        except Exception as e:
            logger.error(f"Failed to quantize model {model_name}: {e}")
            return model
    
    def quantize_model_static(self, 
                            model: 'nn.Module', 
                            model_name: str,
                            calibration_data: List[torch.Tensor]) -> 'nn.Module':
        """静态量化模型
        
        Args:
            model: PyTorch模型
            model_name: 模型名称
            calibration_data: 校准数据
            
        Returns:
            量化后的模型
        """
        if not torch_available:
            logger.warning("PyTorch not available, returning original model")
            return model
        
        try:
            # 设置量化配置
            model.qconfig = torch.quantization.get_default_qconfig(
                self.quantization_configs['static']['backend']
            )
            
            # 准备量化
            model_prepared = torch.quantization.prepare(model)
            
            # 校准
            model_prepared.eval()
            with torch.no_grad():
                for data in calibration_data[:100]:  # 使用前100个样本校准
                    model_prepared(data)
            
            # 转换为量化模型
            quantized_model = torch.quantization.convert(model_prepared)
            
            self.quantized_models[model_name] = quantized_model
            
            logger.info(f"Model {model_name} statically quantized")
            
            return quantized_model
            
        except Exception as e:
            logger.error(f"Failed to statically quantize model {model_name}: {e}")
            return model
    
    def _get_model_size(self, model: 'nn.Module') -> float:
        """获取模型大小（MB）"""
        if not torch_available:
            return 0.0
        
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_mb = (param_size + buffer_size) / 1024 / 1024
        return size_mb
    
    def get_quantized_model(self, model_name: str) -> Optional['nn.Module']:
        """获取量化后的模型"""
        return self.quantized_models.get(model_name)
    
    def benchmark_model(self, 
                       original_model: 'nn.Module', 
                       quantized_model: 'nn.Module',
                       test_data: torch.Tensor,
                       num_runs: int = 100) -> Dict[str, float]:
        """基准测试模型性能"""
        if not torch_available:
            return {'speedup': 1.0, 'accuracy_loss': 0.0}
        
        # 测试原始模型
        original_times = []
        original_model.eval()
        with torch.no_grad():
            for _ in range(num_runs):
                start_time = time.time()
                _ = original_model(test_data)
                original_times.append(time.time() - start_time)
        
        # 测试量化模型
        quantized_times = []
        quantized_model.eval()
        with torch.no_grad():
            for _ in range(num_runs):
                start_time = time.time()
                _ = quantized_model(test_data)
                quantized_times.append(time.time() - start_time)
        
        avg_original_time = np.mean(original_times)
        avg_quantized_time = np.mean(quantized_times)
        speedup = avg_original_time / avg_quantized_time if avg_quantized_time > 0 else 1.0
        
        return {
            'original_time': avg_original_time,
            'quantized_time': avg_quantized_time,
            'speedup': speedup,
            'original_std': np.std(original_times),
            'quantized_std': np.std(quantized_times)
        }


class ThreadPoolManager:
    """线程池管理器
    
    管理多个专用线程池，优化不同类型任务的处理
    """
    
    def __init__(self, 
                 detection_workers: int = 2,
                 analysis_workers: int = 2,
                 recognition_workers: int = 1):
        """
        初始化线程池管理器
        
        Args:
            detection_workers: 检测任务工作线程数
            analysis_workers: 分析任务工作线程数
            recognition_workers: 识别任务工作线程数
        """
        self.pools = {
            'detection': ThreadPoolExecutor(max_workers=detection_workers, thread_name_prefix='detection'),
            'analysis': ThreadPoolExecutor(max_workers=analysis_workers, thread_name_prefix='analysis'),
            'recognition': ThreadPoolExecutor(max_workers=recognition_workers, thread_name_prefix='recognition'),
            'general': ThreadPoolExecutor(max_workers=2, thread_name_prefix='general')
        }
        
        # 任务队列
        self.task_queues = {
            'detection': Queue(),
            'analysis': Queue(),
            'recognition': Queue(),
            'general': Queue()
        }
        
        # 统计信息
        self.task_stats = {
            'submitted': 0,
            'completed': 0,
            'failed': 0,
            'avg_processing_time': 0.0
        }
        
        # 性能监控
        self.processing_times = []
        self.lock = threading.Lock()
        
        logger.info(f"ThreadPoolManager initialized with {sum(pool._max_workers for pool in self.pools.values())} total workers")
    
    def submit_task(self, task: ProcessingTask) -> Future:
        """提交任务
        
        Args:
            task: 处理任务
            
        Returns:
            Future对象
        """
        pool_name = task.task_type if task.task_type in self.pools else 'general'
        pool = self.pools[pool_name]
        
        # 包装任务执行函数
        def execute_task():
            start_time = time.time()
            try:
                result = self._execute_task(task)
                
                # 执行回调
                if task.callback:
                    task.callback(result)
                
                # 更新统计
                processing_time = time.time() - start_time
                with self.lock:
                    self.task_stats['completed'] += 1
                    self.processing_times.append(processing_time)
                    
                    # 保持最近1000次的处理时间
                    if len(self.processing_times) > 1000:
                        self.processing_times = self.processing_times[-1000:]
                    
                    self.task_stats['avg_processing_time'] = np.mean(self.processing_times)
                
                return result
                
            except Exception as e:
                logger.error(f"Task {task.task_id} failed: {e}")
                with self.lock:
                    self.task_stats['failed'] += 1
                raise
        
        # 提交任务
        future = pool.submit(execute_task)
        
        with self.lock:
            self.task_stats['submitted'] += 1
        
        return future
    
    def _execute_task(self, task: ProcessingTask) -> Any:
        """执行具体任务"""
        # 这里可以根据任务类型执行不同的处理逻辑
        # 实际实现中，这个方法会被具体的处理器重写
        return task.data
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """获取线程池统计信息"""
        stats = {}
        
        for name, pool in self.pools.items():
            stats[name] = {
                'max_workers': pool._max_workers,
                'active_threads': len([t for t in pool._threads if t.is_alive()]),
                'queue_size': self.task_queues[name].qsize() if name in self.task_queues else 0
            }
        
        stats['overall'] = self.task_stats.copy()
        
        return stats
    
    def shutdown(self, wait: bool = True):
        """关闭所有线程池"""
        for name, pool in self.pools.items():
            logger.info(f"Shutting down {name} thread pool")
            pool.shutdown(wait=wait)


class PerformanceOptimizer:
    """性能优化器
    
    集成多种性能优化技术
    """
    
    def __init__(self, 
                 enable_multithreading: bool = True,
                 enable_quantization: bool = True,
                 enable_frame_buffering: bool = True,
                 max_buffer_size: int = 30):
        """
        初始化性能优化器
        
        Args:
            enable_multithreading: 启用多线程处理
            enable_quantization: 启用模型量化
            enable_frame_buffering: 启用帧缓冲
            max_buffer_size: 最大缓冲区大小
        """
        self.enable_multithreading = enable_multithreading
        self.enable_quantization = enable_quantization
        self.enable_frame_buffering = enable_frame_buffering
        
        # 初始化组件
        if enable_multithreading:
            self.thread_manager = ThreadPoolManager()
        else:
            self.thread_manager = None
        
        if enable_quantization and torch_available:
            self.quantizer = ModelQuantizer()
        else:
            self.quantizer = None
        
        if enable_frame_buffering:
            self.frame_buffer = FrameBuffer(max_size=max_buffer_size)
        else:
            self.frame_buffer = None
        
        # 性能监控
        self.performance_metrics = {
            'frame_processing_times': [],
            'detection_times': [],
            'analysis_times': [],
            'total_frames_processed': 0,
            'optimization_enabled': {
                'multithreading': enable_multithreading,
                'quantization': enable_quantization and torch_available,
                'frame_buffering': enable_frame_buffering
            }
        }
        
        logger.info(f"PerformanceOptimizer initialized with optimizations: "
                   f"multithreading={enable_multithreading}, "
                   f"quantization={enable_quantization and torch_available}, "
                   f"frame_buffering={enable_frame_buffering}")
    
    def optimize_model(self, model: 'nn.Module', model_name: str, 
                      calibration_data: Optional[List] = None) -> 'nn.Module':
        """优化模型
        
        Args:
            model: 原始模型
            model_name: 模型名称
            calibration_data: 校准数据（用于静态量化）
            
        Returns:
            优化后的模型
        """
        if not self.enable_quantization or not self.quantizer:
            return model
        
        if calibration_data:
            return self.quantizer.quantize_model_static(model, model_name, calibration_data)
        else:
            return self.quantizer.quantize_model_dynamic(model, model_name)
    
    def process_frame_async(self, 
                          frame: np.ndarray, 
                          processor_func: Callable,
                          callback: Optional[Callable] = None,
                          task_type: str = 'detection') -> Optional[Future]:
        """异步处理帧
        
        Args:
            frame: 视频帧
            processor_func: 处理函数
            callback: 回调函数
            task_type: 任务类型
            
        Returns:
            Future对象或None
        """
        if not self.enable_multithreading or not self.thread_manager:
            # 同步处理
            start_time = time.time()
            result = processor_func(frame)
            processing_time = time.time() - start_time
            
            self.performance_metrics['frame_processing_times'].append(processing_time)
            self.performance_metrics['total_frames_processed'] += 1
            
            if callback:
                callback(result)
            
            return None
        
        # 异步处理
        task = ProcessingTask(
            task_id=f"{task_type}_{int(time.time() * 1000)}",
            task_type=task_type,
            data={'frame': frame, 'processor': processor_func},
            callback=callback
        )
        
        return self.thread_manager.submit_task(task)
    
    def add_frame_to_buffer(self, frame: np.ndarray, timestamp: float = None) -> bool:
        """添加帧到缓冲区"""
        if not self.enable_frame_buffering or not self.frame_buffer:
            return False
        
        return self.frame_buffer.add_frame(frame, timestamp)
    
    def get_buffered_frame(self, latest_only: bool = False) -> Optional[Dict[str, Any]]:
        """从缓冲区获取帧"""
        if not self.enable_frame_buffering or not self.frame_buffer:
            return None
        
        if latest_only:
            return self.frame_buffer.get_latest_frame()
        else:
            return self.frame_buffer.get_frame()
    
    def benchmark_optimization(self, 
                             test_frames: List[np.ndarray],
                             processor_func: Callable,
                             num_runs: int = 10) -> Dict[str, Any]:
        """基准测试优化效果
        
        Args:
            test_frames: 测试帧列表
            processor_func: 处理函数
            num_runs: 运行次数
            
        Returns:
            基准测试结果
        """
        results = {
            'sync_times': [],
            'async_times': [],
            'buffer_stats': None,
            'thread_stats': None
        }
        
        # 同步处理基准
        for _ in range(num_runs):
            start_time = time.time()
            for frame in test_frames:
                processor_func(frame)
            results['sync_times'].append(time.time() - start_time)
        
        # 异步处理基准（如果启用）
        if self.enable_multithreading and self.thread_manager:
            for _ in range(num_runs):
                start_time = time.time()
                futures = []
                
                for frame in test_frames:
                    future = self.process_frame_async(frame, processor_func)
                    if future:
                        futures.append(future)
                
                # 等待所有任务完成
                for future in futures:
                    future.result()
                
                results['async_times'].append(time.time() - start_time)
        
        # 收集统计信息
        if self.frame_buffer:
            results['buffer_stats'] = self.frame_buffer.get_stats()
        
        if self.thread_manager:
            results['thread_stats'] = self.thread_manager.get_pool_stats()
        
        # 计算性能提升
        avg_sync_time = np.mean(results['sync_times'])
        if results['async_times']:
            avg_async_time = np.mean(results['async_times'])
            speedup = avg_sync_time / avg_async_time if avg_async_time > 0 else 1.0
        else:
            avg_async_time = avg_sync_time
            speedup = 1.0
        
        results['summary'] = {
            'avg_sync_time': avg_sync_time,
            'avg_async_time': avg_async_time,
            'speedup': speedup,
            'frames_per_second_sync': len(test_frames) / avg_sync_time,
            'frames_per_second_async': len(test_frames) / avg_async_time if avg_async_time > 0 else 0
        }
        
        return results
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        metrics = self.performance_metrics.copy()
        
        # 计算统计信息
        if metrics['frame_processing_times']:
            metrics['avg_frame_processing_time'] = np.mean(metrics['frame_processing_times'])
            metrics['fps_estimate'] = 1.0 / metrics['avg_frame_processing_time']
        else:
            metrics['avg_frame_processing_time'] = 0.0
            metrics['fps_estimate'] = 0.0
        
        # 添加组件统计
        if self.frame_buffer:
            metrics['buffer_stats'] = self.frame_buffer.get_stats()
        
        if self.thread_manager:
            metrics['thread_stats'] = self.thread_manager.get_pool_stats()
        
        if self.quantizer:
            metrics['quantized_models'] = list(self.quantizer.quantized_models.keys())
        
        return metrics
    
    def cleanup(self):
        """清理资源"""
        if self.thread_manager:
            self.thread_manager.shutdown()
        
        if self.frame_buffer:
            self.frame_buffer.clear()
        
        logger.info("PerformanceOptimizer cleaned up")
    
    def auto_tune_parameters(self, 
                           target_fps: float = 30.0,
                           sample_frames: List[np.ndarray] = None) -> Dict[str, Any]:
        """自动调优参数
        
        Args:
            target_fps: 目标帧率
            sample_frames: 样本帧用于测试
            
        Returns:
            调优结果
        """
        tuning_results = {
            'original_config': {
                'buffer_size': self.frame_buffer.max_size if self.frame_buffer else 0,
                'target_size': self.frame_buffer.target_size if self.frame_buffer else None,
            },
            'optimized_config': {},
            'performance_improvement': 0.0
        }
        
        if not sample_frames:
            logger.warning("No sample frames provided for auto-tuning")
            return tuning_results
        
        # 测试不同的缓冲区大小
        if self.frame_buffer:
            best_fps = 0.0
            best_buffer_size = self.frame_buffer.max_size
            
            for buffer_size in [10, 20, 30, 50]:
                self.frame_buffer.max_size = buffer_size
                
                # 简单性能测试
                start_time = time.time()
                for frame in sample_frames[:10]:  # 使用前10帧测试
                    self.frame_buffer.add_frame(frame)
                    self.frame_buffer.get_frame()
                
                elapsed_time = time.time() - start_time
                fps = 10 / elapsed_time if elapsed_time > 0 else 0
                
                if fps > best_fps and fps <= target_fps * 1.1:  # 允许10%的超调
                    best_fps = fps
                    best_buffer_size = buffer_size
            
            self.frame_buffer.max_size = best_buffer_size
            tuning_results['optimized_config']['buffer_size'] = best_buffer_size
            tuning_results['performance_improvement'] = (
                best_fps / (10 / len(sample_frames)) - 1.0 if sample_frames else 0.0
            )
        
        logger.info(f"Auto-tuning completed: {tuning_results}")
        return tuning_results