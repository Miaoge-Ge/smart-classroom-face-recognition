from __future__ import annotations

from datetime import datetime

from sqlalchemy import or_
from sqlalchemy.orm import Session

from core.models import Attendance, Course, Student


def resolve_course_id(db: Session, course_name: str | None) -> int | None:
    if not course_name:
        return None
    row = db.query(Course.course_id).filter(Course.course_name == course_name).first()
    return row[0] if row else None


def get_teacher_course_ids(db: Session, teacher_key: str | None) -> list[int]:
    if not teacher_key:
        return []
    rows = db.query(Course.course_id).filter(Course.teacher == teacher_key).all()
    return [r[0] for r in rows]


def build_attendance_query(
    db: Session,
    *,
    task_id: int | None = None,
    course_id: int | None = None,
    teacher_course_ids: list[int] | None = None,
    college: str | None = None,
    search_query: str | None = None,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
):
    q = (
        db.query(Attendance, Student, Course)
        .join(Student, Attendance.student_id == Student.student_id)
        .outerjoin(Course, Attendance.course_id == Course.course_id)
        .order_by(Attendance.check_time.desc())
    )
    if teacher_course_ids is not None:
        q = q.filter(Attendance.course_id.in_(teacher_course_ids)) if teacher_course_ids else q.filter(False)
    if task_id:
        q = q.filter(Attendance.task_id == task_id)
    if course_id:
        q = q.filter(Attendance.course_id == course_id)
    if college:
        q = q.filter(Student.college == college)
    if start_time:
        q = q.filter(Attendance.check_time >= start_time)
    if end_time:
        q = q.filter(Attendance.check_time <= end_time)
    if search_query:
        q = q.filter(or_(Student.name.like(f"%{search_query}%"), Student.student_no.like(f"%{search_query}%")))
    return q


def query_attendance_joined(
    db: Session,
    *,
    task_id: int | None = None,
    course_id: int | None = None,
    teacher_course_ids: list[int] | None = None,
    college: str | None = None,
    search_query: str | None = None,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
    limit: int = 500,
    offset: int = 0,
):
    q = build_attendance_query(
        db,
        task_id=task_id,
        course_id=course_id,
        teacher_course_ids=teacher_course_ids,
        college=college,
        search_query=search_query,
        start_time=start_time,
        end_time=end_time,
    )
    if offset:
        q = q.offset(offset)
    return q.limit(limit).all()
