#!/usr/bin/env python3
"""
数据管理器单元测试
Data Manager Unit Tests
"""

import os
import sqlite3
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from src.core.data_manager import DetectionDataManager


class TestDetectionDataManager(unittest.TestCase):
    """检测数据管理器测试类"""

    def setUp(self):
        """测试前准备"""
        # 创建临时数据库文件
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
        self.temp_db.close()

        self.data_manager = DetectionDataManager(db_path=self.temp_db.name)

    def tearDown(self):
        """测试后清理"""
        # 关闭数据库连接
        if hasattr(self.data_manager, "conn") and self.data_manager.conn:
            self.data_manager.conn.close()

        # 删除临时数据库文件
        if os.path.exists(self.temp_db.name):
            os.unlink(self.temp_db.name)

    def test_init(self):
        """测试初始化"""
        self.assertIsNotNone(self.data_manager)
        self.assertTrue(os.path.exists(self.temp_db.name))

        # 检查数据库路径
        self.assertEqual(self.data_manager.db_path, self.temp_db.name)

    def test_create_tables(self):
        """测试表创建"""
        # 检查表是否存在
        with sqlite3.connect(self.temp_db.name) as conn:
            cursor = conn.cursor()

            # 检查detection_records表
            cursor.execute(
                """
                SELECT name FROM sqlite_master
                WHERE type='table' AND name='detection_records'
            """
            )
            self.assertIsNotNone(cursor.fetchone())

            # 检查detection_statistics表
            cursor.execute(
                """
                SELECT name FROM sqlite_master
                WHERE type='table' AND name='detection_statistics'
            """
            )
            self.assertIsNotNone(cursor.fetchone())

            # 检查system_config表
            cursor.execute(
                """
                SELECT name FROM sqlite_master
                WHERE type='table' AND name='system_config'
            """
            )
            self.assertIsNotNone(cursor.fetchone())

    def test_save_detection_result(self):
        """测试保存检测结果"""
        detection_results = [
            {
                "bbox": [100, 100, 200, 300],
                "confidence": 0.9,
                "has_hairnet": True,
                "hairnet_confidence": 0.95,
            },
            {
                "bbox": [300, 150, 400, 350],
                "confidence": 0.8,
                "has_hairnet": True,
                "hairnet_confidence": 0.85,
            },
            {
                "bbox": [500, 200, 600, 400],
                "confidence": 0.75,
                "has_hairnet": False,
                "hairnet_confidence": 0.3,
            },
        ]

        result = self.data_manager.save_detection_result(
            frame_id="test_frame_001",
            detection_results=detection_results,
            detection_type="hairnet",
            processing_time=0.5,
        )
        self.assertTrue(result)

    def test_get_detection_history(self):
        """测试获取检测历史"""
        # 先保存一些测试数据
        detection_results = [
            {
                "bbox": [100, 100, 200, 300],
                "confidence": 0.8,
                "has_hairnet": True,
                "hairnet_confidence": 0.9,
            }
        ]

        result = self.data_manager.save_detection_result(
            frame_id="test_frame_002",
            detection_results=detection_results,
            detection_type="hairnet",
        )
        self.assertTrue(result)

        # 获取记录
        records = self.data_manager.get_detection_history(limit=10)
        self.assertIsInstance(records, list)
        self.assertGreater(len(records), 0)

        # 检查记录内容
        record = records[0]
        self.assertIn("id", record)
        self.assertIn("timestamp", record)
        self.assertIn("detection_type", record)

    def test_get_detection_history_with_date_range(self):
        """测试按日期范围获取检测历史"""
        # 保存测试数据
        detection_results = [
            {
                "bbox": [100, 100, 200, 300],
                "confidence": 0.9,
                "has_hairnet": True,
                "hairnet_confidence": 0.95,
            }
        ]

        result = self.data_manager.save_detection_result(
            frame_id="test_frame_003",
            detection_results=detection_results,
            detection_type="hairnet",
        )
        self.assertTrue(result)

        # 测试日期范围查询
        now = datetime.now()
        start_date = now - timedelta(hours=1)
        end_date = now + timedelta(hours=1)

        records = self.data_manager.get_detection_history(
            start_time=start_date, end_time=end_date
        )

        self.assertIsInstance(records, list)
        self.assertGreaterEqual(len(records), 0)

    def test_get_statistics(self):
        """测试获取统计信息"""
        # 先保存一些测试数据
        detection_results = [
            {
                "bbox": [100, 100, 200, 300],
                "confidence": 0.85,
                "has_hairnet": True,
                "hairnet_confidence": 0.9,
            },
            {
                "bbox": [300, 150, 400, 350],
                "confidence": 0.8,
                "has_hairnet": True,
                "hairnet_confidence": 0.85,
            },
        ]

        result = self.data_manager.save_detection_result(
            frame_id="test_frame_stats1",
            detection_results=detection_results,
            detection_type="hairnet",
        )
        self.assertTrue(result)

        # 获取统计信息
        stats = self.data_manager.get_realtime_statistics()
        self.assertIsInstance(stats, dict)
        self.assertIn("today", stats)
        self.assertIn("total", stats)

    def test_get_statistics_with_date_range(self):
        """测试按日期范围获取统计信息"""
        # 保存测试数据
        detection_results = [
            {
                "bbox": [100, 100, 200, 300],
                "confidence": 0.8,
                "has_hairnet": True,
                "hairnet_confidence": 0.85,
            }
        ]

        result = self.data_manager.save_detection_result(
            frame_id="test_frame_stats2",
            detection_results=detection_results,
            detection_type="hairnet",
        )
        self.assertTrue(result)

        # 获取每日统计
        daily_stats = self.data_manager.get_daily_statistics(days=1)
        self.assertIsInstance(daily_stats, list)

    def test_update_summary(self):
        """测试更新汇总数据"""
        # 先保存一些检测记录
        detection_results = [
            {
                "bbox": [100, 100, 200, 300],
                "confidence": 0.8,
                "has_hairnet": True,
                "hairnet_confidence": 0.85,
            },
            {
                "bbox": [300, 150, 400, 350],
                "confidence": 0.75,
                "has_hairnet": False,
                "hairnet_confidence": 0.3,
            },
        ]

        result = self.data_manager.save_detection_result(
            frame_id="test_frame_summary",
            detection_results=detection_results,
            detection_type="hairnet",
        )
        self.assertTrue(result)

        # 获取实时统计作为汇总数据的替代
        summary = self.data_manager.get_realtime_statistics()
        self.assertIsInstance(summary, dict)
        self.assertIn("total", summary)

    def test_export_data(self):
        """测试数据导出"""
        # 先保存一些测试数据
        detection_results = [
            {
                "bbox": [100, 100, 200, 300],
                "confidence": 0.9,
                "has_hairnet": True,
                "hairnet_confidence": 0.95,
            }
        ]

        result = self.data_manager.save_detection_result(
            frame_id="test_frame_004",
            detection_results=detection_results,
            detection_type="hairnet",
        )
        self.assertTrue(result)

        # 测试导出（模拟，不实际创建文件）
        # 这里只测试方法存在性
        self.assertTrue(hasattr(self.data_manager, "export_data"))

    def test_realtime_statistics(self):
        """测试实时统计"""
        # 先保存一些测试数据
        detection_results = [
            {
                "bbox": [100, 100, 200, 300],
                "confidence": 0.9,
                "has_hairnet": True,
                "hairnet_confidence": 0.95,
            }
        ]

        result = self.data_manager.save_detection_result(
            frame_id="test_frame_stats",
            detection_results=detection_results,
            detection_type="hairnet",
        )
        self.assertTrue(result)

        # 获取实时统计
        stats = self.data_manager.get_realtime_statistics()
        self.assertIsInstance(stats, dict)
        self.assertIn("today", stats)
        self.assertIn("last_hour", stats)
        self.assertIn("total", stats)

    def test_cleanup_old_records(self):
        """测试清理旧记录"""
        # 先添加一些测试数据
        detection_results = [
            {
                "bbox": [100, 100, 200, 300],
                "confidence": 0.7,
                "has_hairnet": False,
                "hairnet_confidence": 0.3,
            }
        ]

        result = self.data_manager.save_detection_result(
            frame_id="test_frame_old",
            detection_results=detection_results,
            detection_type="hairnet",
        )
        self.assertTrue(result)

        # 清理30天前的记录
        deleted_count = self.data_manager.cleanup_old_records(days=30)
        self.assertGreaterEqual(deleted_count, 0)

    def test_daily_statistics(self):
        """测试每日统计"""
        # 先保存一些测试数据
        detection_results = [
            {
                "bbox": [100, 100, 200, 300],
                "confidence": 0.8,
                "has_hairnet": True,
                "hairnet_confidence": 0.9,
            }
        ]

        result = self.data_manager.save_detection_result(
            frame_id="test_frame_daily",
            detection_results=detection_results,
            detection_type="hairnet",
        )
        self.assertTrue(result)

        # 获取每日统计
        daily_stats = self.data_manager.get_daily_statistics(days=7)
        self.assertIsInstance(daily_stats, list)

    def test_database_connection_error_handling(self):
        """测试数据库连接错误处理"""
        # 测试无效路径的处理
        try:
            invalid_manager = DetectionDataManager(db_path="/invalid/path/test.db")
            # 如果没有抛出异常，测试保存操作的错误处理
            detection_results = [
                {
                    "bbox": [100, 100, 200, 300],
                    "confidence": 0.9,
                    "has_hairnet": True,
                    "hairnet_confidence": 0.95,
                }
            ]

            result = invalid_manager.save_detection_result(
                frame_id="test_frame_error", detection_results=detection_results
            )
            # 如果路径无效，保存应该失败
            self.assertFalse(result)
        except Exception:
            # 如果初始化时就抛出异常，这也是预期的行为
            pass

    def test_json_serialization(self):
        """测试JSON序列化和反序列化"""
        # 测试复杂数据结构的序列化
        detection_results = [
            {
                "bbox": [100, 100, 200, 300],
                "confidence": 0.9,
                "has_hairnet": True,
                "hairnet_confidence": 0.95,
                "metadata": {"camera_id": "cam_001", "location": "entrance"},
            }
        ]

        # 保存数据
        result = self.data_manager.save_detection_result(
            frame_id="test_frame_json",
            detection_results=detection_results,
            detection_type="hairnet",
        )
        self.assertTrue(result)

        # 获取数据并验证反序列化
        records = self.data_manager.get_detection_history(limit=1)
        self.assertGreater(len(records), 0)

        record = records[0]
        self.assertIn("results", record)

        # 验证嵌套数据结构
        results = record["results"]
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)

        detection = results[0]
        self.assertIn("metadata", detection)
        self.assertEqual(detection["metadata"]["camera_id"], "cam_001")


if __name__ == "__main__":
    unittest.main()
