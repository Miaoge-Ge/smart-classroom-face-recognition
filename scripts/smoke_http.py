import time
from datetime import datetime, timedelta
from pathlib import Path

import sys

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.database import SessionLocal
from core.models import Course, Student, UserAccount


BASE = "http://127.0.0.1:8000"


def must(condition: bool, msg: str):
    if not condition:
        raise AssertionError(msg)


def login(session: requests.Session, username: str, password: str) -> requests.Response:
    r = session.post(
        f"{BASE}/login",
        data={"username": username, "password": password},
        allow_redirects=False,
        timeout=15,
    )
    must(r.status_code in (302, 303), f"login status={r.status_code}")
    must("access_token" in session.cookies, "missing access_token cookie")
    return r


def get_ok(session: requests.Session, path: str) -> str:
    r = session.get(f"{BASE}{path}", timeout=20, allow_redirects=False)
    must(r.status_code == 200, f"GET {path} status={r.status_code}")
    ct = (r.headers.get("content-type") or "").lower()
    must("text/html" in ct, f"GET {path} content-type={ct}")
    return r.text


def post_redirect(session: requests.Session, path: str, data) -> requests.Response:
    r = session.post(f"{BASE}{path}", data=data, timeout=30, allow_redirects=False)
    must(r.status_code in (302, 303), f"POST {path} status={r.status_code}, body={r.text[:200]}")
    return r


def ensure_smoke_student(class_name: str) -> tuple[Student, bool]:
    db = SessionLocal()
    try:
        s = db.query(Student).filter(Student.student_no.like("SMOKE-%")).first()
        if s:
            if not s.class_name:
                s.class_name = class_name
                db.commit()
            return s, False
        s = Student(
            student_no=f"SMOKE-{int(time.time())}",
            name="SMOKE_STUDENT",
            class_name=class_name,
            college="SMOKE_COLLEGE",
            gender="UNKNOWN",
            created_at=datetime.now(),
        )
        db.add(s)
        db.commit()
        db.refresh(s)
        return s, True
    finally:
        db.close()


def create_smoke_course(class_names: list[str]) -> Course:
    db = SessionLocal()
    try:
        code = f"SMOKE-C-{int(time.time())}"
        start = datetime.now().replace(second=0, microsecond=0) + timedelta(minutes=10)
        end = start + timedelta(minutes=45)
        c = Course(
            course_no=code,
            course_name="SMOKE_COURSE",
            teacher="SMOKE_TEACHER",
            schedule=f"{start.strftime('%Y-%m-%d %H:%M')} - {end.strftime('%Y-%m-%d %H:%M')}",
            start_time=start,
            end_time=end,
            location="SMOKE_ROOM",
            class_names=",".join(class_names),
            created_at=datetime.now(),
        )
        db.add(c)
        db.commit()
        db.refresh(c)
        return c
    finally:
        db.close()


def delete_course(course_id: int):
    db = SessionLocal()
    try:
        db.query(Course).filter(Course.course_id == course_id).delete(synchronize_session=False)
        db.commit()
    finally:
        db.close()


def delete_user_by_username(username: str):
    db = SessionLocal()
    try:
        db.query(UserAccount).filter(UserAccount.username == username).delete(synchronize_session=False)
        db.commit()
    finally:
        db.close()

def delete_student_by_no(student_no: str):
    db = SessionLocal()
    try:
        db.query(Student).filter(Student.student_no == student_no).delete(synchronize_session=False)
        db.commit()
    finally:
        db.close()


def main():
    s = requests.Session()

    login(s, "admin", "admin123")

    get_ok(s, "/")
    get_ok(s, "/history")
    get_ok(s, "/attendance_tasks")
    get_ok(s, "/courses")
    get_ok(s, "/students")
    get_ok(s, "/users")
    get_ok(s, "/audit")
    get_ok(s, "/settings")

    cls = "SMOKE_CLASS"
    stu, created_stu = ensure_smoke_student(cls)

    start = (datetime.now() + timedelta(minutes=5)).replace(second=0, microsecond=0)
    end = start + timedelta(minutes=90)
    course_payload = [
        ("code", f"SMOKE-WEB-{int(time.time())}"),
        ("name", "SMOKE_WEB_COURSE"),
        ("teacher", "SMOKE_WEB_TEACHER"),
        ("start_time", start.strftime("%Y-%m-%dT%H:%M")),
        ("end_time", end.strftime("%Y-%m-%dT%H:%M")),
        ("location", "SMOKE_WEB_ROOM"),
        ("class_names", cls),
    ]
    post_redirect(s, "/api/courses", course_payload)

    db = SessionLocal()
    try:
        course = db.query(Course).filter(Course.course_no.like("SMOKE-WEB-%")).order_by(Course.course_id.desc()).first()
        must(course is not None, "created course not found in db")
        course_id = int(course.course_id)
    finally:
        db.close()

    task_payload = [
        ("title", "SMOKE_TASK"),
        ("course_id", str(course_id)),
        ("class_names", cls),
    ]
    post_redirect(s, "/api/attendance_tasks", task_payload)

    r = s.get(f"{BASE}/history?course_id={course_id}", timeout=20, allow_redirects=False)
    must(r.status_code == 200, f"history with course_id status={r.status_code}")
    must("缺勤统计" in r.text, "history missing absent stats section")

    teacher_username = f"smoke_teacher_{int(time.time())}"
    post_redirect(
        s,
        "/api/users",
        {"username": teacher_username, "password": "pass12345", "role": "teacher", "full_name": "SMOKE_TEACHER"},
    )

    ts = requests.Session()
    login(ts, teacher_username, "pass12345")
    get_ok(ts, "/")
    get_ok(ts, "/my_courses")
    r = ts.get(f"{BASE}/courses", timeout=10, allow_redirects=False)
    must(r.status_code in (403, 307), f"teacher /courses expected forbidden, got {r.status_code}")

    delete_user_by_username(teacher_username)

    post_redirect(s, "/api/delete_course", {"course_id": str(course_id)})
    if created_stu:
        delete_student_by_no(stu.student_no)

    print("SMOKE_HTTP_OK")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print("SMOKE_HTTP_FAILED:", str(e))
        raise
