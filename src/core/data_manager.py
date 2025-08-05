import json
import logging
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class DetectionDataManager:
    """检测数据管理器

    负责存储和查询人体检测和发网检测结果
    """

    def __init__(self, db_path: Optional[str] = None):
        """
        初始化数据管理器

        Args:
            db_path: 数据库文件路径，如果为None则使用默认路径
        """
        if db_path is None:
            # 使用项目根目录下的data目录
            project_root = Path(__file__).parent.parent.parent
            data_dir = project_root / "data"
            # 确保data目录存在
            if not data_dir.exists():
                data_dir.mkdir(exist_ok=True)
            self.db_path = str(data_dir / "detection_results.db")
        else:
            self.db_path = db_path

        self._init_database()

        logger.info(f"DetectionDataManager initialized with database: {self.db_path}")

    def _init_database(self):
        """初始化数据库表结构"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # 创建检测记录表
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS detection_records (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        frame_id TEXT,
                        detection_type TEXT,  -- 'person' or 'hairnet'
                        results TEXT,  -- JSON格式的检测结果
                        total_persons INTEGER,
                        persons_with_hairnet INTEGER,
                        accuracy_score REAL,
                        processing_time REAL
                    )
                """
                )

                # 创建统计汇总表
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS detection_statistics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        date DATE,
                        total_detections INTEGER,
                        total_persons INTEGER,
                        total_with_hairnet INTEGER,
                        hairnet_compliance_rate REAL,
                        avg_confidence REAL,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """
                )

                # 创建配置表
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS system_config (
                        key TEXT PRIMARY KEY,
                        value TEXT,
                        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """
                )

                conn.commit()
                logger.info("数据库表结构初始化完成")

        except Exception as e:
            logger.error(f"数据库初始化失败: {e}")
            raise

    def save_detection_result(
        self,
        frame_id: str,
        detection_results: List[Dict],
        detection_type: str = "hairnet",
        processing_time: float = 0.0,
    ) -> bool:
        """
        保存检测结果

        Args:
            frame_id: 帧ID
            detection_results: 检测结果列表
            detection_type: 检测类型
            processing_time: 处理时间（秒）

        Returns:
            保存是否成功
        """
        try:
            # 计算统计信息
            total_persons = len(detection_results)
            persons_with_hairnet = sum(
                1 for r in detection_results if r.get("has_hairnet", False)
            )

            # 计算平均置信度
            if detection_results:
                avg_confidence = (
                    sum(r.get("hairnet_confidence", 0) for r in detection_results)
                    / total_persons
                )
            else:
                avg_confidence = 0.0

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """
                    INSERT INTO detection_records
                    (frame_id, detection_type, results, total_persons,
                     persons_with_hairnet, accuracy_score, processing_time)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        frame_id,
                        detection_type,
                        json.dumps(detection_results, ensure_ascii=False),
                        total_persons,
                        persons_with_hairnet,
                        avg_confidence,
                        processing_time,
                    ),
                )

                conn.commit()
                logger.debug(
                    f"保存检测结果: frame_id={frame_id}, persons={total_persons}, with_hairnet={persons_with_hairnet}"
                )
                return True

        except Exception as e:
            logger.error(f"保存检测结果失败: {e}")
            return False

    def get_detection_history(
        self,
        limit: int = 100,
        detection_type: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[Dict]:
        """
        获取检测历史记录

        Args:
            limit: 返回记录数量限制
            detection_type: 检测类型过滤
            start_time: 开始时间
            end_time: 结束时间

        Returns:
            历史记录列表
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                query = "SELECT * FROM detection_records WHERE 1=1"
                params = []

                if detection_type:
                    query += " AND detection_type = ?"
                    params.append(detection_type)

                if start_time:
                    query += " AND timestamp >= ?"
                    params.append(start_time.isoformat())

                if end_time:
                    query += " AND timestamp <= ?"
                    params.append(end_time.isoformat())

                query += " ORDER BY timestamp DESC LIMIT ?"
                params.append(limit)

                cursor.execute(query, params)
                rows = cursor.fetchall()

                results = []
                for row in rows:
                    result = dict(row)
                    # 解析JSON结果
                    try:
                        result["results"] = json.loads(result["results"])
                    except:
                        result["results"] = []
                    results.append(result)

                return results

        except Exception as e:
            logger.error(f"获取检测历史失败: {e}")
            return []

    def get_daily_statistics(self, days: int = 7) -> List[Dict]:
        """
        获取每日统计数据

        Args:
            days: 统计天数

        Returns:
            每日统计列表
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                cursor.execute(
                    """
                    SELECT
                        DATE(timestamp) as date,
                        COUNT(*) as total_detections,
                        SUM(total_persons) as total_persons,
                        SUM(persons_with_hairnet) as total_with_hairnet,
                        ROUND(AVG(CAST(persons_with_hairnet AS FLOAT) / NULLIF(total_persons, 0) * 100), 2) as compliance_rate,
                        ROUND(AVG(accuracy_score), 3) as avg_confidence
                    FROM detection_records
                    WHERE timestamp >= date('now', '-{} days')
                    GROUP BY DATE(timestamp)
                    ORDER BY date DESC
                """.format(
                        days
                    )
                )

                rows = cursor.fetchall()
                return [dict(row) for row in rows]

        except Exception as e:
            logger.error(f"获取每日统计失败: {e}")
            return []

    def get_realtime_statistics(self) -> Dict:
        """
        获取实时统计数据

        Returns:
            实时统计字典
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                # 今日统计
                cursor.execute(
                    """
                    SELECT
                        COUNT(*) as today_detections,
                        SUM(total_persons) as today_persons,
                        SUM(persons_with_hairnet) as today_with_hairnet,
                        ROUND(AVG(CAST(persons_with_hairnet AS FLOAT) / NULLIF(total_persons, 0) * 100), 2) as today_compliance_rate
                    FROM detection_records
                    WHERE DATE(timestamp) = DATE('now')
                """
                )

                today_stats = dict(cursor.fetchone() or {})

                # 最近1小时统计
                cursor.execute(
                    """
                    SELECT
                        COUNT(*) as hour_detections,
                        SUM(total_persons) as hour_persons,
                        SUM(persons_with_hairnet) as hour_with_hairnet
                    FROM detection_records
                    WHERE timestamp >= datetime('now', '-1 hour')
                """
                )

                hour_stats = dict(cursor.fetchone() or {})

                # 总体统计
                cursor.execute(
                    """
                    SELECT
                        COUNT(*) as total_detections,
                        SUM(total_persons) as total_persons,
                        SUM(persons_with_hairnet) as total_with_hairnet
                    FROM detection_records
                """
                )

                total_stats = dict(cursor.fetchone() or {})

                return {
                    "today": today_stats,
                    "last_hour": hour_stats,
                    "total": total_stats,
                    "updated_at": datetime.now().isoformat(),
                }

        except Exception as e:
            logger.error(f"获取实时统计失败: {e}")
            return {}

    def cleanup_old_records(self, days: int = 30) -> int:
        """
        清理旧记录

        Args:
            days: 保留天数

        Returns:
            删除的记录数
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """
                    DELETE FROM detection_records
                    WHERE timestamp < date('now', '-{} days')
                """.format(
                        days
                    )
                )

                deleted_count = cursor.rowcount
                conn.commit()

                logger.info(f"清理了 {deleted_count} 条旧记录")
                return deleted_count

        except Exception as e:
            logger.error(f"清理旧记录失败: {e}")
            return 0

    def export_data(
        self,
        output_path: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> bool:
        """
        导出数据到CSV文件

        Args:
            output_path: 输出文件路径
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)

        Returns:
            导出是否成功
        """
        try:
            import csv

            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                query = "SELECT * FROM detection_records WHERE 1=1"
                params = []

                if start_date:
                    query += " AND DATE(timestamp) >= ?"
                    params.append(start_date)

                if end_date:
                    query += " AND DATE(timestamp) <= ?"
                    params.append(end_date)

                query += " ORDER BY timestamp"

                cursor.execute(query, params)
                rows = cursor.fetchall()

                with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
                    if rows:
                        fieldnames = rows[0].keys()
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                        writer.writeheader()

                        for row in rows:
                            writer.writerow(dict(row))

                logger.info(f"数据导出成功: {output_path}, 共 {len(rows)} 条记录")
                return True

        except Exception as e:
            logger.error(f"数据导出失败: {e}")
            return False
