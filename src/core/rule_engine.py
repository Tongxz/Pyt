import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class RuleType(Enum):
    """规则类型枚举"""

    BEHAVIOR_REQUIRED = "behavior_required"  # 必需行为
    BEHAVIOR_FORBIDDEN = "behavior_forbidden"  # 禁止行为
    OCCUPANCY_LIMIT = "occupancy_limit"  # 人数限制
    DURATION_LIMIT = "duration_limit"  # 时长限制
    TIME_RESTRICTION = "time_restriction"  # 时间限制
    CUSTOM = "custom"  # 自定义规则


class RulePriority(Enum):
    """规则优先级枚举"""

    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class ViolationSeverity(Enum):
    """违规严重程度枚举"""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class RuleCondition:
    """规则条件"""

    field: str  # 检查的字段
    operator: str  # 操作符 (eq, ne, gt, lt, gte, lte, in, not_in)
    value: Any  # 比较值
    description: str = ""  # 条件描述


@dataclass
class Rule:
    """规则定义"""

    rule_id: str
    name: str
    rule_type: RuleType
    priority: RulePriority
    conditions: List[RuleCondition] = field(default_factory=list)
    actions: List[str] = field(default_factory=list)  # 触发的动作
    is_active: bool = True
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


@dataclass
class Violation:
    """违规记录"""

    violation_id: str
    rule_id: str
    rule_name: str
    region_id: Optional[str]
    track_id: Optional[int]
    severity: ViolationSeverity
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    acknowledged: bool = False
    resolved: bool = False


class RuleEngine:
    """规则引擎

    负责管理和执行检测规则，处理违规检测和告警
    """

    def __init__(self):
        """初始化规则引擎"""
        self.rules: Dict[str, Rule] = {}
        self.violations: Dict[str, Violation] = {}
        self.rule_handlers: Dict[RuleType, Callable] = {
            RuleType.BEHAVIOR_REQUIRED: self._check_behavior_required,
            RuleType.BEHAVIOR_FORBIDDEN: self._check_behavior_forbidden,
            RuleType.OCCUPANCY_LIMIT: self._check_occupancy_limit,
            RuleType.DURATION_LIMIT: self._check_duration_limit,
            RuleType.TIME_RESTRICTION: self._check_time_restriction,
            RuleType.CUSTOM: self._check_custom_rule,
        }

        # 统计信息
        self.stats = {
            "total_rules": 0,
            "active_rules": 0,
            "total_violations": 0,
            "resolved_violations": 0,
            "last_check_time": None,
        }

        logger.info("RuleEngine initialized")

    def add_rule(self, rule: Rule) -> bool:
        """添加规则

        Args:
            rule: 规则对象

        Returns:
            True if successfully added
        """
        if rule.rule_id in self.rules:
            logger.warning(f"Rule {rule.rule_id} already exists")
            return False

        self.rules[rule.rule_id] = rule
        self._update_stats()

        logger.info(f"Rule '{rule.name}' added successfully")
        return True

    def remove_rule(self, rule_id: str) -> bool:
        """移除规则

        Args:
            rule_id: 规则ID

        Returns:
            True if successfully removed
        """
        if rule_id not in self.rules:
            logger.warning(f"Rule {rule_id} not found")
            return False

        del self.rules[rule_id]
        self._update_stats()

        logger.info(f"Rule {rule_id} removed successfully")
        return True

    def update_rule(self, rule_id: str, **kwargs) -> bool:
        """更新规则

        Args:
            rule_id: 规则ID
            **kwargs: 更新的字段

        Returns:
            True if successfully updated
        """
        if rule_id not in self.rules:
            logger.warning(f"Rule {rule_id} not found")
            return False

        rule = self.rules[rule_id]
        for key, value in kwargs.items():
            if hasattr(rule, key):
                setattr(rule, key, value)

        rule.updated_at = time.time()
        self._update_stats()

        logger.info(f"Rule {rule_id} updated successfully")
        return True

    def evaluate_rules(self, context: Dict[str, Any]) -> List[Violation]:
        """评估所有规则

        Args:
            context: 评估上下文，包含当前状态信息

        Returns:
            违规列表
        """
        violations = []

        # 按优先级排序规则
        sorted_rules = sorted(
            [rule for rule in self.rules.values() if rule.is_active],
            key=lambda r: r.priority.value,
            reverse=True,
        )

        for rule in sorted_rules:
            try:
                violation = self._evaluate_single_rule(rule, context)
                if violation:
                    violations.append(violation)
                    self.violations[violation.violation_id] = violation
            except Exception as e:
                logger.error(f"Error evaluating rule {rule.rule_id}: {e}")

        self.stats["last_check_time"] = time.time()
        self.stats["total_violations"] = len(self.violations)

        return violations

    def _evaluate_single_rule(
        self, rule: Rule, context: Dict[str, Any]
    ) -> Optional[Violation]:
        """评估单个规则"""
        handler = self.rule_handlers.get(rule.rule_type)
        if not handler:
            logger.warning(f"No handler for rule type {rule.rule_type}")
            return None

        return handler(rule, context)

    def _check_behavior_required(
        self, rule: Rule, context: Dict[str, Any]
    ) -> Optional[Violation]:
        """检查必需行为规则"""
        required_behaviors = rule.metadata.get("required_behaviors", [])
        current_behaviors = context.get("behaviors", {})

        for behavior in required_behaviors:
            if behavior not in current_behaviors or not current_behaviors[behavior].get(
                "is_active", False
            ):
                return Violation(
                    violation_id=f"{rule.rule_id}_{int(time.time() * 1000)}",
                    rule_id=rule.rule_id,
                    rule_name=rule.name,
                    region_id=context.get("region_id"),
                    track_id=context.get("track_id"),
                    severity=ViolationSeverity.WARNING,
                    message=f"Required behavior '{behavior}' not detected",
                    details={"missing_behavior": behavior, "context": context},
                )

        return None

    def _check_behavior_forbidden(
        self, rule: Rule, context: Dict[str, Any]
    ) -> Optional[Violation]:
        """检查禁止行为规则"""
        forbidden_behaviors = rule.metadata.get("forbidden_behaviors", [])
        current_behaviors = context.get("behaviors", {})

        for behavior in forbidden_behaviors:
            if behavior in current_behaviors and current_behaviors[behavior].get(
                "is_active", False
            ):
                return Violation(
                    violation_id=f"{rule.rule_id}_{int(time.time() * 1000)}",
                    rule_id=rule.rule_id,
                    rule_name=rule.name,
                    region_id=context.get("region_id"),
                    track_id=context.get("track_id"),
                    severity=ViolationSeverity.ERROR,
                    message=f"Forbidden behavior '{behavior}' detected",
                    details={"forbidden_behavior": behavior, "context": context},
                )

        return None

    def _check_occupancy_limit(
        self, rule: Rule, context: Dict[str, Any]
    ) -> Optional[Violation]:
        """检查人数限制规则"""
        max_occupancy = rule.metadata.get("max_occupancy", -1)
        current_occupancy = context.get("current_occupancy", 0)

        if max_occupancy > 0 and current_occupancy > max_occupancy:
            return Violation(
                violation_id=f"{rule.rule_id}_{int(time.time() * 1000)}",
                rule_id=rule.rule_id,
                rule_name=rule.name,
                region_id=context.get("region_id"),
                track_id=None,
                severity=ViolationSeverity.WARNING,
                message=f"Occupancy limit exceeded: {current_occupancy}/{max_occupancy}",
                details={
                    "current_occupancy": current_occupancy,
                    "max_occupancy": max_occupancy,
                },
            )

        return None

    def _check_duration_limit(
        self, rule: Rule, context: Dict[str, Any]
    ) -> Optional[Violation]:
        """检查时长限制规则"""
        max_duration = rule.metadata.get("max_duration", -1)
        current_duration = context.get("duration", 0)

        if max_duration > 0 and current_duration > max_duration:
            return Violation(
                violation_id=f"{rule.rule_id}_{int(time.time() * 1000)}",
                rule_id=rule.rule_id,
                rule_name=rule.name,
                region_id=context.get("region_id"),
                track_id=context.get("track_id"),
                severity=ViolationSeverity.WARNING,
                message=f"Duration limit exceeded: {current_duration:.1f}s > {max_duration}s",
                details={
                    "current_duration": current_duration,
                    "max_duration": max_duration,
                },
            )

        return None

    def _check_time_restriction(
        self, rule: Rule, context: Dict[str, Any]
    ) -> Optional[Violation]:
        """检查时间限制规则"""
        allowed_hours = rule.metadata.get("allowed_hours", [])
        current_hour = context.get("current_hour", time.localtime().tm_hour)

        if allowed_hours and current_hour not in allowed_hours:
            return Violation(
                violation_id=f"{rule.rule_id}_{int(time.time() * 1000)}",
                rule_id=rule.rule_id,
                rule_name=rule.name,
                region_id=context.get("region_id"),
                track_id=context.get("track_id"),
                severity=ViolationSeverity.INFO,
                message=f"Access outside allowed hours: {current_hour}:00",
                details={"current_hour": current_hour, "allowed_hours": allowed_hours},
            )

        return None

    def _check_custom_rule(
        self, rule: Rule, context: Dict[str, Any]
    ) -> Optional[Violation]:
        """检查自定义规则"""
        # 评估所有条件
        for condition in rule.conditions:
            if not self._evaluate_condition(condition, context):
                return Violation(
                    violation_id=f"{rule.rule_id}_{int(time.time() * 1000)}",
                    rule_id=rule.rule_id,
                    rule_name=rule.name,
                    region_id=context.get("region_id"),
                    track_id=context.get("track_id"),
                    severity=ViolationSeverity.WARNING,
                    message=f"Custom rule violation: {condition.description}",
                    details={"condition": condition.__dict__, "context": context},
                )

        return None

    def _evaluate_condition(
        self, condition: RuleCondition, context: Dict[str, Any]
    ) -> bool:
        """评估条件"""
        field_value = context.get(condition.field)

        if condition.operator == "eq":
            return field_value == condition.value
        elif condition.operator == "ne":
            return field_value != condition.value
        elif condition.operator == "gt":
            return field_value > condition.value
        elif condition.operator == "lt":
            return field_value < condition.value
        elif condition.operator == "gte":
            return field_value >= condition.value
        elif condition.operator == "lte":
            return field_value <= condition.value
        elif condition.operator == "in":
            return field_value in condition.value
        elif condition.operator == "not_in":
            return field_value not in condition.value
        else:
            logger.warning(f"Unknown operator: {condition.operator}")
            return True

    def acknowledge_violation(self, violation_id: str) -> bool:
        """确认违规"""
        if violation_id not in self.violations:
            return False

        self.violations[violation_id].acknowledged = True
        logger.info(f"Violation {violation_id} acknowledged")
        return True

    def resolve_violation(self, violation_id: str) -> bool:
        """解决违规"""
        if violation_id not in self.violations:
            return False

        self.violations[violation_id].resolved = True
        self.stats["resolved_violations"] = sum(
            1 for v in self.violations.values() if v.resolved
        )
        logger.info(f"Violation {violation_id} resolved")
        return True

    def get_active_violations(self) -> List[Violation]:
        """获取活跃违规"""
        return [v for v in self.violations.values() if not v.resolved]

    def get_violations_by_severity(
        self, severity: ViolationSeverity
    ) -> List[Violation]:
        """按严重程度获取违规"""
        return [v for v in self.violations.values() if v.severity == severity]

    def _update_stats(self):
        """更新统计信息"""
        self.stats["total_rules"] = len(self.rules)
        self.stats["active_rules"] = sum(
            1 for rule in self.rules.values() if rule.is_active
        )

    def load_rules_config(self, config_path: str) -> bool:
        """从配置文件加载规则

        Args:
            config_path: 配置文件路径

        Returns:
            True if successfully loaded
        """
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)

            # 清空现有规则
            self.rules.clear()

            # 加载规则
            for rule_data in config.get("rules", []):
                rule = Rule(
                    rule_id=rule_data["rule_id"],
                    name=rule_data["name"],
                    rule_type=RuleType(rule_data["rule_type"]),
                    priority=RulePriority(rule_data["priority"]),
                    conditions=[
                        RuleCondition(**cond)
                        for cond in rule_data.get("conditions", [])
                    ],
                    actions=rule_data.get("actions", []),
                    is_active=rule_data.get("is_active", True),
                    description=rule_data.get("description", ""),
                    metadata=rule_data.get("metadata", {}),
                    created_at=rule_data.get("created_at", time.time()),
                    updated_at=rule_data.get("updated_at", time.time()),
                )
                self.add_rule(rule)

            logger.info(f"Rules config loaded from {config_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to load rules config: {e}")
            return False

    def save_rules_config(self, config_path: str) -> bool:
        """保存规则配置到文件

        Args:
            config_path: 配置文件路径

        Returns:
            True if successfully saved
        """
        try:
            config = {
                "rules": [
                    {
                        "rule_id": rule.rule_id,
                        "name": rule.name,
                        "rule_type": rule.rule_type.value,
                        "priority": rule.priority.value,
                        "conditions": [cond.__dict__ for cond in rule.conditions],
                        "actions": rule.actions,
                        "is_active": rule.is_active,
                        "description": rule.description,
                        "metadata": rule.metadata,
                        "created_at": rule.created_at,
                        "updated_at": rule.updated_at,
                    }
                    for rule in self.rules.values()
                ]
            }

            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2, ensure_ascii=False)

            logger.info(f"Rules config saved to {config_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save rules config: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return self.stats.copy()

    def reset(self):
        """重置规则引擎"""
        self.rules.clear()
        self.violations.clear()
        self.stats = {
            "total_rules": 0,
            "active_rules": 0,
            "total_violations": 0,
            "resolved_violations": 0,
            "last_check_time": None,
        }
        logger.info("RuleEngine reset")
