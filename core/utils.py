from datetime import datetime, date, time, timedelta
from typing import Optional, Tuple


def parse_date(date_str: Optional[str]) -> Optional[date]:
    """解析日期字符串为 date 对象"""
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def parse_datetime(datetime_str: Optional[str]) -> Optional[datetime]:
    """解析日期时间字符串为 datetime 对象"""
    if not datetime_str:
        return None
    try:
        return datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError):
        try:
            return datetime.strptime(datetime_str, "%Y-%m-%d")
        except (ValueError, TypeError):
            return None


def get_date_range(start_date: Optional[date], end_date: Optional[date]) -> Tuple[datetime, datetime]:
    """获取日期范围的开始和结束时间"""
    today = date.today()
    
    if not start_date:
        start_date = today - timedelta(days=30)
    if not end_date:
        end_date = today
    
    start_datetime = datetime.combine(start_date, time.min)
    end_datetime = datetime.combine(end_date, time.max)
    
    return start_datetime, end_datetime


def format_datetime(dt: Optional[datetime], fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    """格式化 datetime 为字符串"""
    if not dt:
        return ""
    return dt.strftime(fmt)


def format_date(d: Optional[date], fmt: str = "%Y-%m-%d") -> str:
    """格式化 date 为字符串"""
    if not d:
        return ""
    return d.strftime(fmt)


def is_today(dt: datetime) -> bool:
    """判断是否为今天"""
    return dt.date() == date.today()


def days_ago(dt: datetime) -> int:
    """计算距离今天的天数"""
    return (date.today() - dt.date()).days
