"""
智慧课堂考勤系统 - 核心模块
包含：
- config_manager: 配置管理
- database: 数据库连接
- models: 数据模型
- data_access: 数据访问层
- security: 安全工具
- crypto_manager: 加密管理
- audit: 审计日志
- runtime_settings: 运行时设置
- model_factory: 模型工厂
- utils: 通用工具
- csrf: CSRF 保护
"""

from core.config_manager import Config, global_config, get_global_config
from core.database import Base, engine, get_db, SessionLocal
from core.models import (
    Student,
    Attendance,
    AdminUser,
    Course,
    UserAccount,
    AuditLog,
    CourseStudent,
)
from core.data_access import (
    resolve_course_id,
    get_teacher_course_ids,
    build_attendance_query,
    query_attendance_joined,
)
from core.security import verify_password, get_password_hash
from core.crypto_manager import encrypt_bytes, decrypt_bytes, encrypt_to_b64, decrypt_from_b64
from core.audit import write_audit_log
from core.runtime_settings import get_runtime_settings, load_raw_config, save_raw_config
from core.model_factory import ModelFactory
from core.utils import (
    parse_date,
    parse_datetime,
    get_date_range,
    format_datetime,
    format_date,
    is_today,
    days_ago,
)

__all__ = [
    # 配置
    "Config",
    "global_config",
    "get_global_config",
    # 数据库
    "Base",
    "engine",
    "get_db",
    "SessionLocal",
    # 模型
    "Student",
    "Attendance",
    "AdminUser",
    "Course",
    "UserAccount",
    "AuditLog",
    "CourseStudent",
    # 数据访问
    "resolve_course_id",
    "get_teacher_course_ids",
    "build_attendance_query",
    "query_attendance_joined",
    # 安全
    "verify_password",
    "get_password_hash",
    # 加密
    "encrypt_bytes",
    "decrypt_bytes",
    "encrypt_to_b64",
    "decrypt_from_b64",
    # 审计
    "write_audit_log",
    # 运行时设置
    "get_runtime_settings",
    "load_raw_config",
    "save_raw_config",
    # 模型工厂
    "ModelFactory",
    # 工具
    "parse_date",
    "parse_datetime",
    "get_date_range",
    "format_datetime",
    "format_date",
    "is_today",
    "days_ago",
]
