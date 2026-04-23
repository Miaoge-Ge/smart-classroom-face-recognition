import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENV_PY = ROOT / ".venv" / "Scripts" / "python.exe"
if VENV_PY.exists() and Path(sys.executable).resolve() != VENV_PY.resolve():
    os.execv(str(VENV_PY), [str(VENV_PY), *sys.argv])
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.database import SessionLocal
from core.models import Attendance, CourseStudent, AttendanceTaskStudent, Student, UserAccount


def main() -> int:
    db = SessionLocal()
    try:
        students = db.query(Student).all()
        if not students:
            print("NO_STUDENTS")
            return 0

        ids = [int(s.student_id) for s in students if s.student_id is not None]
        face_paths = [str(s.face_image_path) for s in students if getattr(s, "face_image_path", None)]

        if ids:
            db.query(UserAccount).filter(UserAccount.student_id.in_(ids)).update({"student_id": None}, synchronize_session=False)
            db.query(Attendance).filter(Attendance.student_id.in_(ids)).delete(synchronize_session=False)
            db.query(CourseStudent).filter(CourseStudent.student_id.in_(ids)).delete(synchronize_session=False)
            db.query(AttendanceTaskStudent).filter(AttendanceTaskStudent.student_id.in_(ids)).delete(synchronize_session=False)
            db.query(Student).filter(Student.student_id.in_(ids)).delete(synchronize_session=False)
        else:
            db.query(Student).delete(synchronize_session=False)

        db.commit()

        removed_files = 0
        for p in face_paths:
            try:
                if p and os.path.exists(p):
                    os.remove(p)
                    removed_files += 1
            except Exception:
                pass

        print(f"DELETED_STUDENTS {len(students)}")
        print(f"DELETED_FACE_FILES {removed_files}")
        return 0
    except Exception as e:
        db.rollback()
        print(f"ERROR {e}")
        return 1
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())

