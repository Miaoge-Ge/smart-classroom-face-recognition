import sys
import os
import cv2
import numpy as np
import base64
import json
import asyncio
import hashlib
import logging
import time
import pandas as pd
import io
import yaml
import glob
from datetime import datetime, timedelta, timezone
from fastapi import FastAPI, WebSocket, UploadFile, File, Form, Depends, HTTPException, status
from fastapi.responses import StreamingResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from fastapi.responses import RedirectResponse
from contextlib import asynccontextmanager
from sqlalchemy.orm import Session
from sqlalchemy import func, or_, text
from sqlalchemy.exc import IntegrityError, OperationalError

# Security imports
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from starlette.websockets import WebSocketDisconnect
from core.security import verify_password, get_password_hash

# Add root to path FIRST before importing local modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.face_service import FaceRecognitionService
from core.database import engine, Base, get_db, SessionLocal
from core.models import (
    Student,
    Attendance,
    AdminUser,
    Course,
    UserAccount,
    AuditLog,
    CourseStudent,
    AttendanceTask,
    AttendanceTaskStudent,
)
from core.config_manager import Config, global_config
from core.runtime_settings import get_runtime_settings, load_raw_config, save_raw_config
from core.audit import write_audit_log
from core.crypto_manager import encrypt_bytes, encrypt_to_b64, decrypt_bytes
from core.data_access import resolve_course_id, get_teacher_course_ids, build_attendance_query, query_attendance_joined

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Create Tables
Base.metadata.create_all(bind=engine)

def _repair_sqlite_course_students_table():
    try:
        if engine.url.get_backend_name() != "sqlite":
            return
        with engine.connect() as conn:
            row = conn.execute(text("SELECT sql FROM sqlite_master WHERE type='table' AND name='course_students'")).fetchone()
            ddl = row[0] if row and row[0] else ""
        if not ddl:
            return
        if "courses_old" not in ddl and "students_old" not in ddl:
            return
        with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
            conn.exec_driver_sql("PRAGMA foreign_keys=OFF")
            conn.exec_driver_sql("DROP TABLE IF EXISTS course_students_broken")
            conn.exec_driver_sql("ALTER TABLE course_students RENAME TO course_students_broken")
            conn.exec_driver_sql(
                """
                CREATE TABLE course_students (
                    id INTEGER NOT NULL,
                    course_id INTEGER NOT NULL,
                    student_id INTEGER NOT NULL,
                    created_at DATETIME,
                    PRIMARY KEY (id),
                    FOREIGN KEY(course_id) REFERENCES courses (course_id) ON DELETE CASCADE,
                    FOREIGN KEY(student_id) REFERENCES students (student_id) ON DELETE CASCADE
                )
                """
            )
            conn.exec_driver_sql(
                """
                INSERT INTO course_students (id, course_id, student_id, created_at)
                SELECT id, course_id, student_id, created_at FROM course_students_broken
                """
            )
            conn.exec_driver_sql("DROP TABLE course_students_broken")
            conn.exec_driver_sql("CREATE INDEX IF NOT EXISTS ix_course_students_course_id ON course_students (course_id)")
            conn.exec_driver_sql("CREATE INDEX IF NOT EXISTS ix_course_students_student_id ON course_students (student_id)")
            conn.exec_driver_sql("PRAGMA foreign_keys=ON")
    except Exception:
        return

_repair_sqlite_course_students_table()

# Config
import secrets as _secrets
_secret_key_env = os.getenv("SECRET_KEY")
if not _secret_key_env:
    SECRET_KEY = _secrets.token_urlsafe(32)
    logging.warning("SECRET_KEY 未设置，已自动生成随机密钥。重启后所有已签发的 JWT 将失效。请通过环境变量 SECRET_KEY 设置固定密钥。")
else:
    SECRET_KEY = _secret_key_env
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Global Service
service = None
service_init_error = None
logger = logging.getLogger("systems.web")
_RECOGNITION_MODEL_DIR_ALIASES = {
    "fastcontextface": "nexnet",
}


def _normalize_recognition_model_choice(model_choice: str | None) -> str | None:
    if not model_choice:
        return model_choice
    choice = str(model_choice).replace("\\", "/")
    prefix = "models/weights/recognition/"
    relative_choice = choice[len(prefix):] if choice.lower().startswith(prefix) else choice
    lowered = relative_choice.lower()
    for legacy_dir, canonical_dir in _RECOGNITION_MODEL_DIR_ALIASES.items():
        legacy_prefix = f"{legacy_dir}/"
        if lowered.startswith(legacy_prefix):
            suffix = relative_choice[len(legacy_prefix):]
            normalized = f"{canonical_dir}/{suffix}"
            return f"{prefix}{normalized}" if choice.lower().startswith(prefix) else normalized
    return choice

def _write_attendance_batch_sync(items: list[dict]):
    db = SessionLocal()
    try:
        rows = []
        for it in items:
            rows.append(
                Attendance(
                    student_id=int(it["student_id"]),
                    course_id=int(it["course_id"]),
                    task_id=int(it["task_id"]) if it.get("task_id") is not None else None,
                    check_time=it.get("check_time") or datetime.now(),
                    created_at=it.get("created_at") or datetime.now(),
                    confidence=it.get("confidence"),
                    status=it.get("status") or "已签到",
                )
            )
        db.add_all(rows)
        for i in range(3):
            try:
                db.commit()
                return
            except OperationalError as e:
                db.rollback()
                if "locked" in str(e).lower() and i < 2:
                    time.sleep(0.05 * (2**i))
                    continue
                raise
    finally:
        db.close()


async def _attendance_writer_worker(queue: asyncio.Queue):
    batch: list[dict] = []
    last_flush = time.monotonic()

    async def flush_batch() -> None:
        nonlocal batch, last_flush
        if not batch:
            return
        pending = list(batch)
        batch = []
        try:
            await asyncio.to_thread(_write_attendance_batch_sync, pending)
        except Exception:
            logger.exception("Failed to flush attendance batch", extra={"batch_size": len(pending)})
        last_flush = time.monotonic()

    try:
        while True:
            try:
                item = await asyncio.wait_for(queue.get(), timeout=0.2)
                batch.append(item)
            except asyncio.TimeoutError:
                item = None

            now = time.monotonic()
            if batch and (len(batch) >= 50 or (now - last_flush) >= 0.5):
                await flush_batch()
    except asyncio.CancelledError:
        try:
            while True:
                batch.append(queue.get_nowait())
        except asyncio.QueueEmpty:
            pass
        await flush_batch()
        raise


async def _dedup_cleanup_worker(app: FastAPI):
    while True:
        await asyncio.sleep(30)
        cache = getattr(app.state, "dedup_cache", None)
        lock = getattr(app.state, "dedup_lock", None)
        if cache is None or lock is None:
            continue
        now = datetime.now()
        max_keep_seconds = int(get_runtime_settings().get("attendance", {}).get("dedup_seconds", 60)) * 20
        cutoff = now.timestamp() - max(600, max_keep_seconds)
        async with lock:
            app.state.dedup_cache = {k: v for k, v in cache.items() if v.timestamp() >= cutoff}


def ensure_schema():
    expected = {
        "students": {"student_id", "student_no", "name", "class_name", "college", "gender", "face_image_path", "face_embedding_enc", "face_embedding_model_sig", "created_at"},
        "courses": {"course_id", "course_no", "course_name", "teacher", "schedule", "start_time", "end_time", "location", "class_names", "created_at"},
        "attendances": {"record_id", "student_id", "course_id", "task_id", "check_time", "confidence", "status", "created_at"},
        "attendance_tasks": {"task_id", "title", "course_id", "class_name", "status", "start_time", "end_time", "created_by", "created_at"},
        "attendance_task_students": {"id", "task_id", "student_id", "created_at"},
    }

    def table_cols(conn, table: str) -> set[str]:
        rows = conn.execute(text(f"PRAGMA table_info({table})")).fetchall()
        return {r[1] for r in rows}

    def has_table(conn, table: str) -> bool:
        row = conn.execute(
            text("SELECT 1 FROM sqlite_master WHERE type='table' AND name=:t LIMIT 1"),
            {"t": table},
        ).fetchone()
        return bool(row)

    def user_accounts_fk_ok(conn) -> bool:
        if not has_table(conn, "user_accounts"):
            return True
        try:
            rows = conn.execute(text("PRAGMA foreign_key_list(user_accounts)")).fetchall()
        except Exception:
            return True
        for r in rows:
            if len(r) >= 3 and r[2] == "students_old":
                return False
            if len(r) >= 3 and r[2] and not has_table(conn, str(r[2])):
                return False
            if len(r) >= 7 and r[2] == "students" and str(r[6]).upper() != "SET NULL":
                return False
        return True

    def fk_ok(conn, table: str, bad_refs: set[str]) -> bool:
        if not has_table(conn, table):
            return True
        try:
            rows = conn.execute(text(f"PRAGMA foreign_key_list({table})")).fetchall()
        except Exception:
            return True
        for r in rows:
            if len(r) >= 3 and r[2] in bad_refs:
                return False
            if len(r) >= 3 and r[2] and not has_table(conn, str(r[2])):
                return False
        return True

    def create_target_tables(conn):
        conn.execute(text("PRAGMA foreign_keys=OFF"))
        conn.execute(
            text(
                "CREATE TABLE IF NOT EXISTS students ("
                "  student_id INTEGER PRIMARY KEY AUTOINCREMENT,"
                "  student_no VARCHAR NOT NULL,"
                "  name VARCHAR NOT NULL,"
                "  class_name VARCHAR,"
                "  college VARCHAR,"
                "  gender VARCHAR,"
                "  face_image_path VARCHAR,"
                "  face_embedding_enc TEXT,"
                "  face_embedding_model_sig VARCHAR,"
                "  created_at DATETIME"
                ")"
            )
        )
        conn.execute(
            text(
                "CREATE TABLE IF NOT EXISTS courses ("
                "  course_id INTEGER PRIMARY KEY AUTOINCREMENT,"
                "  course_no VARCHAR NOT NULL,"
                "  course_name VARCHAR NOT NULL,"
                "  teacher VARCHAR,"
                "  schedule VARCHAR,"
                "  start_time DATETIME,"
                "  end_time DATETIME,"
                "  location VARCHAR,"
                "  class_names VARCHAR,"
                "  created_at DATETIME"
                ")"
            )
        )
        conn.execute(
            text(
                "CREATE TABLE IF NOT EXISTS attendance_tasks ("
                "  task_id INTEGER PRIMARY KEY AUTOINCREMENT,"
                "  title VARCHAR NOT NULL,"
                "  course_id INTEGER NOT NULL,"
                "  class_name VARCHAR,"
                "  status VARCHAR NOT NULL,"
                "  start_time DATETIME,"
                "  end_time DATETIME,"
                "  created_by VARCHAR,"
                "  created_at DATETIME,"
                "  FOREIGN KEY(course_id) REFERENCES courses(course_id) ON DELETE CASCADE"
                ")"
            )
        )
        conn.execute(
            text(
                "CREATE TABLE IF NOT EXISTS attendances ("
                "  record_id INTEGER PRIMARY KEY AUTOINCREMENT,"
                "  student_id INTEGER NOT NULL,"
                "  course_id INTEGER NOT NULL,"
                "  task_id INTEGER,"
                "  check_time DATETIME,"
                "  confidence FLOAT,"
                "  status VARCHAR,"
                "  created_at DATETIME,"
                "  FOREIGN KEY(student_id) REFERENCES students(student_id) ON DELETE CASCADE,"
                "  FOREIGN KEY(course_id) REFERENCES courses(course_id) ON DELETE CASCADE,"
                "  FOREIGN KEY(task_id) REFERENCES attendance_tasks(task_id) ON DELETE SET NULL"
                ")"
            )
        )
        conn.execute(
            text(
                "CREATE TABLE IF NOT EXISTS attendance_task_students ("
                "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
                "  task_id INTEGER NOT NULL,"
                "  student_id INTEGER NOT NULL,"
                "  created_at DATETIME,"
                "  FOREIGN KEY(task_id) REFERENCES attendance_tasks(task_id) ON DELETE CASCADE,"
                "  FOREIGN KEY(student_id) REFERENCES students(student_id) ON DELETE CASCADE"
                ")"
            )
        )
        conn.execute(text("CREATE UNIQUE INDEX IF NOT EXISTS ux_students_student_no ON students(student_no)"))
        conn.execute(text("CREATE UNIQUE INDEX IF NOT EXISTS ux_courses_course_no ON courses(course_no)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS ix_task_course_id ON attendance_tasks(course_id)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS ix_task_status ON attendance_tasks(status)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS ix_task_class_name ON attendance_tasks(class_name)"))
        conn.execute(text("CREATE UNIQUE INDEX IF NOT EXISTS ux_task_student ON attendance_task_students(task_id, student_id)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS ix_task_students_task ON attendance_task_students(task_id)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS ix_att_student_id ON attendances(student_id)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS ix_att_course_id ON attendances(course_id)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS ix_att_task_id ON attendances(task_id)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS ix_att_check_time ON attendances(check_time)"))
        conn.execute(text("PRAGMA foreign_keys=ON"))

    with engine.connect() as conn:
        need_rebuild = False
        for t, cols in expected.items():
            if not has_table(conn, t):
                need_rebuild = True
                break
            actual = table_cols(conn, t)
            if actual != cols:
                need_rebuild = True
                break

        fix_user_accounts = not user_accounts_fk_ok(conn)
        fix_fk = not fk_ok(conn, "attendance_tasks", {"courses_old"}) or not fk_ok(
            conn, "attendance_task_students", {"attendance_tasks_old", "students_old"}
        ) or not fk_ok(conn, "attendances", {"attendance_tasks_old", "students_old", "courses_old"})

        if not need_rebuild and not fix_user_accounts and not fix_fk:
            try:
                with conn.begin():
                    if has_table(conn, "user_accounts") and has_table(conn, "students"):
                        conn.execute(
                            text(
                                "UPDATE user_accounts SET student_id=NULL "
                                "WHERE student_id IN (SELECT student_id FROM students WHERE student_no='UNKNOWN')"
                            )
                        )
                    if has_table(conn, "courses"):
                        conn.execute(text("DELETE FROM courses WHERE course_no='UNKNOWN'"))
                    if has_table(conn, "students"):
                        conn.execute(text("DELETE FROM students WHERE student_no='UNKNOWN'"))
            except Exception:
                pass
            return

        db_path = None
        try:
            db_path = os.path.abspath(engine.url.database) if engine.url.database else None
        except Exception:
            db_path = None
        if db_path and os.path.exists(db_path):
            try:
                import shutil

                backup_pattern = db_path + ".bak.*"
                existing_backups = []
                try:
                    existing_backups = [p for p in glob.glob(backup_pattern) if os.path.isfile(p)]
                    existing_backups.sort(key=lambda p: os.path.getmtime(p), reverse=True)
                except Exception:
                    existing_backups = []

                should_backup = True
                if existing_backups:
                    try:
                        last_mtime = os.path.getmtime(existing_backups[0])
                        should_backup = (time.time() - float(last_mtime)) > (6 * 3600)
                    except Exception:
                        should_backup = True

                if should_backup:
                    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
                    shutil.copy2(db_path, db_path + f".bak.{ts}")

                try:
                    keep = 3
                    for p in existing_backups[keep:]:
                        try:
                            os.remove(p)
                        except Exception:
                            pass
                except Exception:
                    pass
            except Exception:
                pass

        existing_tables = {
            r[0] for r in conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'")).fetchall()
        }

        conn.execute(text("PRAGMA foreign_keys=OFF"))
        conn.execute(text("BEGIN IMMEDIATE"))
        try:
            if "attendance_task_students" in existing_tables:
                conn.execute(text("ALTER TABLE attendance_task_students RENAME TO attendance_task_students_old"))
            if "attendance_tasks" in existing_tables:
                conn.execute(text("ALTER TABLE attendance_tasks RENAME TO attendance_tasks_old"))
            if "students" in existing_tables:
                conn.execute(text("ALTER TABLE students RENAME TO students_old"))
            if "courses" in existing_tables:
                conn.execute(text("ALTER TABLE courses RENAME TO courses_old"))
            if "attendances" in existing_tables:
                conn.execute(text("ALTER TABLE attendances RENAME TO attendances_old"))
            if "user_accounts" in existing_tables:
                conn.execute(text("ALTER TABLE user_accounts RENAME TO user_accounts_old"))

            create_target_tables(conn)
            conn.execute(
                text(
                    "CREATE TABLE IF NOT EXISTS user_accounts ("
                    "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
                    "  username VARCHAR NOT NULL,"
                    "  hashed_password VARCHAR,"
                    "  role VARCHAR NOT NULL,"
                    "  full_name VARCHAR,"
                    "  student_id INTEGER,"
                    "  created_at DATETIME,"
                    "  FOREIGN KEY(student_id) REFERENCES students(student_id) ON DELETE SET NULL"
                    ")"
                )
            )
            conn.execute(text("CREATE UNIQUE INDEX IF NOT EXISTS ux_user_accounts_username ON user_accounts(username)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS ix_user_accounts_role ON user_accounts(role)"))

            if "students_old" in {r[0] for r in conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'")).fetchall()}:
                old_cols = table_cols(conn, "students_old")
                expr_student_id = "id" if "id" in old_cols else ("student_id" if "student_id" in old_cols else "NULL")
                parts = []
                if "student_no" in old_cols:
                    parts.append("student_no")
                if "student_id" in old_cols:
                    parts.append("CAST(student_id AS TEXT)")
                if "id" in old_cols:
                    parts.append("CAST(id AS TEXT)")
                expr_student_no = f"COALESCE({', '.join(parts)})" if len(parts) > 1 else (parts[0] if parts else "CAST(rowid AS TEXT)")
                expr_name = "name" if "name" in old_cols else "''"
                expr_class = (
                    "class" if "class" in old_cols else ("class_name" if "class_name" in old_cols else "NULL")
                )
                expr_college = "college" if "college" in old_cols else "NULL"
                expr_gender = "gender" if "gender" in old_cols else "NULL"
                expr_face_path = "face_image_path" if "face_image_path" in old_cols else "NULL"
                expr_face_emb = "face_embedding_enc" if "face_embedding_enc" in old_cols else "NULL"
                expr_face_sig = "face_embedding_model_sig" if "face_embedding_model_sig" in old_cols else "NULL"
                expr_created = "created_at" if "created_at" in old_cols else "NULL"

                conn.execute(
                    text(
                        "INSERT INTO students(student_id, student_no, name, class_name, college, gender, face_image_path, face_embedding_enc, face_embedding_model_sig, created_at) "
                        f"SELECT {expr_student_id}, {expr_student_no}, {expr_name}, {expr_class}, {expr_college}, {expr_gender}, {expr_face_path}, {expr_face_emb}, {expr_face_sig}, {expr_created} "
                        "FROM students_old"
                    )
                )
                conn.execute(
                    text(
                        "WITH dup AS ("
                        "  SELECT student_id, student_no,"
                        "         ROW_NUMBER() OVER (PARTITION BY student_no ORDER BY student_id) AS rn"
                        "  FROM students"
                        ") "
                        "UPDATE students "
                        "SET student_no = student_no || '-' || student_id "
                        "WHERE student_id IN (SELECT student_id FROM dup WHERE rn > 1)"
                    )
                )

            if "courses_old" in {r[0] for r in conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'")).fetchall()}:
                old_cols = table_cols(conn, "courses_old")
                expr_course_id = "id" if "id" in old_cols else ("course_id" if "course_id" in old_cols else "NULL")
                parts = []
                if "course_no" in old_cols:
                    parts.append("course_no")
                if "code" in old_cols:
                    parts.append("code")
                if "id" in old_cols:
                    parts.append("CAST(id AS TEXT)")
                expr_course_no = f"COALESCE({', '.join(parts)})" if len(parts) > 1 else (parts[0] if parts else "CAST(rowid AS TEXT)")
                if "course_name" in old_cols and "name" in old_cols:
                    expr_course_name = "COALESCE(course_name, name)"
                elif "course_name" in old_cols:
                    expr_course_name = "course_name"
                elif "name" in old_cols:
                    expr_course_name = "name"
                else:
                    expr_course_name = "''"
                expr_teacher = "teacher" if "teacher" in old_cols else "NULL"
                if "schedule" in old_cols and "schedule_time" in old_cols:
                    expr_schedule = "COALESCE(schedule, schedule_time)"
                elif "schedule" in old_cols:
                    expr_schedule = "schedule"
                elif "schedule_time" in old_cols:
                    expr_schedule = "schedule_time"
                else:
                    expr_schedule = "NULL"
                expr_location = "location" if "location" in old_cols else "NULL"
                expr_created = "created_at" if "created_at" in old_cols else "NULL"

                conn.execute(
                    text(
                        "INSERT INTO courses(course_id, course_no, course_name, teacher, schedule, start_time, end_time, location, class_names, created_at) "
                        f"SELECT {expr_course_id}, {expr_course_no}, {expr_course_name}, {expr_teacher}, {expr_schedule}, NULL, NULL, {expr_location}, NULL, {expr_created} "
                        "FROM courses_old"
                    )
                )
                conn.execute(
                    text(
                        "WITH dup AS ("
                        "  SELECT course_id, course_no,"
                        "         ROW_NUMBER() OVER (PARTITION BY course_no ORDER BY course_id) AS rn"
                        "  FROM courses"
                        ") "
                        "UPDATE courses "
                        "SET course_no = course_no || '-' || course_id "
                        "WHERE course_id IN (SELECT course_id FROM dup WHERE rn > 1)"
                    )
                )

            if "attendances_old" in {r[0] for r in conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'")).fetchall()}:
                old_cols = table_cols(conn, "attendances_old")
                expr_record_id = "id" if "id" in old_cols else ("record_id" if "record_id" in old_cols else "NULL")
                expr_student_id = "student_id" if "student_id" in old_cols else "NULL"
                expr_course_id = (
                    "course_id"
                    if "course_id" in old_cols
                    else (
                        "(SELECT course_id FROM courses WHERE course_name = attendances_old.course_name LIMIT 1)"
                        if "course_name" in old_cols
                        else "NULL"
                    )
                )
                if "check_time" in old_cols and "timestamp" in old_cols:
                    expr_check_time = "COALESCE(check_time, timestamp)"
                elif "check_time" in old_cols:
                    expr_check_time = "check_time"
                elif "timestamp" in old_cols:
                    expr_check_time = "timestamp"
                else:
                    expr_check_time = "NULL"

                if "created_at" in old_cols and "timestamp" in old_cols:
                    expr_created = "COALESCE(created_at, timestamp)"
                elif "created_at" in old_cols:
                    expr_created = "created_at"
                elif "timestamp" in old_cols:
                    expr_created = "timestamp"
                else:
                    expr_created = "NULL"
                expr_conf = "confidence" if "confidence" in old_cols else "NULL"
                expr_status = "status" if "status" in old_cols else "'已签到'"

                conn.execute(
                    text(
                        "INSERT INTO attendances(record_id, student_id, course_id, task_id, check_time, confidence, status, created_at) "
                        f"SELECT {expr_record_id}, {expr_student_id}, {expr_course_id}, NULL, {expr_check_time}, {expr_conf}, {expr_status}, {expr_created} "
                        "FROM attendances_old "
                        f"WHERE ({expr_student_id}) IS NOT NULL AND ({expr_course_id}) IS NOT NULL "
                        f"AND EXISTS(SELECT 1 FROM students WHERE students.student_id = ({expr_student_id})) "
                        f"AND EXISTS(SELECT 1 FROM courses WHERE courses.course_id = ({expr_course_id}))"
                    )
                )

            if "user_accounts_old" in {r[0] for r in conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'")).fetchall()}:
                old_cols = table_cols(conn, "user_accounts_old")
                expr_id = "id" if "id" in old_cols else "NULL"
                expr_username = "username" if "username" in old_cols else "''"
                expr_hp = "hashed_password" if "hashed_password" in old_cols else "NULL"
                expr_role = "role" if "role" in old_cols else "'student'"
                expr_full = "full_name" if "full_name" in old_cols else "NULL"
                if "student_id" in old_cols:
                    expr_sid = (
                        "CASE "
                        "WHEN student_id IS NOT NULL AND EXISTS(SELECT 1 FROM students WHERE students.student_id = user_accounts_old.student_id) "
                        "THEN user_accounts_old.student_id ELSE NULL END"
                    )
                else:
                    expr_sid = "NULL"
                expr_created = "created_at" if "created_at" in old_cols else "NULL"
                conn.execute(
                    text(
                        "INSERT INTO user_accounts(id, username, hashed_password, role, full_name, student_id, created_at) "
                        f"SELECT {expr_id}, {expr_username}, {expr_hp}, {expr_role}, {expr_full}, {expr_sid}, {expr_created} "
                        "FROM user_accounts_old"
                    )
                )

            conn.execute(text("UPDATE user_accounts SET student_id=NULL WHERE student_id IN (SELECT student_id FROM students WHERE student_no='UNKNOWN')"))
            conn.execute(text("DELETE FROM courses WHERE course_no='UNKNOWN'"))
            conn.execute(text("DELETE FROM students WHERE student_no='UNKNOWN'"))

            conn.execute(text("DROP TABLE IF EXISTS attendances_old"))
            conn.execute(text("DROP TABLE IF EXISTS students_old"))
            conn.execute(text("DROP TABLE IF EXISTS courses_old"))
            conn.execute(text("DROP TABLE IF EXISTS user_accounts_old"))
            conn.execute(text("DROP TABLE IF EXISTS attendance_task_students_old"))
            conn.execute(text("DROP TABLE IF EXISTS attendance_tasks_old"))
            conn.execute(text("COMMIT"))
        except Exception:
            conn.execute(text("ROLLBACK"))
            raise
        finally:
            conn.execute(text("PRAGMA foreign_keys=ON"))


ensure_schema()

def init_service():
    global service, service_init_error
    logger.info("Initializing Face Service...")
    try:
        new_config = Config()
        service = FaceRecognitionService(config=new_config)
        service_init_error = None
        logger.info("Face Service Initialized Successfully.")
    except Exception as e:
        service = None
        service_init_error = str(e)
        logger.exception("Error initializing face service: %s", e)

@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_schema()
    app.state.ws_lock = asyncio.Lock()
    app.state.active_ws = 0
    app.state.inference_semaphore = None
    app.state.inference_semaphore_limit = None
    app.state.attendance_queue = asyncio.Queue(maxsize=5000)
    app.state.attendance_worker = asyncio.create_task(_attendance_writer_worker(app.state.attendance_queue))
    app.state.dedup_lock = asyncio.Lock()
    app.state.dedup_cache = {}
    app.state.dedup_cleanup = asyncio.create_task(_dedup_cleanup_worker(app))
    # Init Default Admin if not exists
    db = SessionLocal()
    admin_account = db.query(UserAccount).filter(UserAccount.username == "admin").first()
    if not admin_account:
        hashed = get_password_hash("admin123")
        admin_account = UserAccount(username="admin", hashed_password=hashed, role="admin", full_name="System Admin")
        db.add(admin_account)
        db.commit()
        print("Default admin created (admin/admin123)")
    else:
        if not admin_account.hashed_password:
            admin_account.hashed_password = get_password_hash("admin123")
            db.add(admin_account)
            db.commit()
            print("Default admin password reset (admin/admin123)")

    legacy_admin = db.query(AdminUser).filter(AdminUser.username == "admin").first()
    if not legacy_admin:
        hashed = get_password_hash("admin123")
        legacy_admin = AdminUser(username="admin", hashed_password=hashed, full_name="System Admin")
        db.add(legacy_admin)
        db.commit()
    db.close()
    
    init_service()
    
    yield
    print("Shutdown: Cleaning up...")
    try:
        app.state.attendance_worker.cancel()
    except Exception:
        pass
    try:
        app.state.dedup_cleanup.cancel()
    except Exception:
        pass

app = FastAPI(lifespan=lifespan, title="智慧课堂考勤管理系统")

app.mount("/static", StaticFiles(directory="web/static"), name="static")
templates = Jinja2Templates(directory="web/templates")


@app.middleware("http")
async def https_redirect_middleware(request: Request, call_next):
    settings = get_runtime_settings()
    forwarded_proto = request.headers.get("x-forwarded-proto")
    scheme = forwarded_proto or request.url.scheme
    if settings.get("security", {}).get("force_https"):
        if scheme != "https":
            url = request.url.replace(scheme="https")
            return RedirectResponse(url=str(url), status_code=status.HTTP_307_TEMPORARY_REDIRECT)
    response = await call_next(request)
    if scheme == "https":
        response.headers.setdefault("Strict-Transport-Security", "max-age=31536000; includeSubDomains")
    return response

# --- Auth Helpers ---

def create_access_token(
    username: str,
    role: str,
    full_name: str | None = None,
    student_id: int | None = None,
    expires_delta: timedelta | None = None,
):
    to_encode = {"sub": username, "role": role, "full_name": full_name, "student_id": student_id}
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_principal(request: Request, db: Session = Depends(get_db)):
    token = request.cookies.get("access_token")
    if not token:
        return None
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str | None = payload.get("sub")
        role: str | None = payload.get("role")
        if not username:
            return None

        if not role:
            legacy_admin = db.query(AdminUser).filter(AdminUser.username == username).first()
            if not legacy_admin:
                return None
            return {"username": legacy_admin.username, "role": "admin", "full_name": legacy_admin.full_name}

        account = db.query(UserAccount).filter(UserAccount.username == username).first()
        if not account:
            legacy_admin = db.query(AdminUser).filter(AdminUser.username == username).first()
            if legacy_admin and role == "admin":
                return {"username": legacy_admin.username, "role": "admin", "full_name": legacy_admin.full_name}
            return None

        return {
            "username": account.username,
            "role": account.role,
            "full_name": account.full_name,
            "student_id": account.student_id,
        }
    except JWTError:
        return None

async def login_required(principal=Depends(get_current_principal)):
    if not principal:
        raise HTTPException(status_code=status.HTTP_307_TEMPORARY_REDIRECT, headers={"Location": "/login"})
    return principal

def require_roles(*roles: str):
    async def dep(principal=Depends(login_required)):
        if principal.get("role") not in set(roles):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="无权限访问该资源")
        return principal

    return dep

def parse_date_param(value: str | None, *, end: bool) -> datetime | None:
    if not value:
        return None
    s = str(value).strip()
    if not s:
        return None
    s = s.replace("年", "-").replace("月", "-").replace("日", "")
    s = s.replace("/", "-")
    has_time = ":" in s
    dt = None
    try:
        dt = datetime.fromisoformat(s)
    except Exception:
        try:
            dt = datetime.strptime(s, "%Y-%m-%d")
        except Exception:
            return None
    if not has_time:
        if end:
            return dt.replace(hour=23, minute=59, second=59, microsecond=999999)
        return dt.replace(hour=0, minute=0, second=0, microsecond=0)
    return dt


def _normalize_tabular_text(value) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if float(value).is_integer():
            return str(int(value))
    return str(value).strip()

# --- Pages ---

@app.get("/login")
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.post("/login")
async def login_submit(request: Request, db: Session = Depends(get_db), username: str = Form(...), password: str = Form(...)):
    account = db.query(UserAccount).filter(UserAccount.username == username).first()
    if account and verify_password(password, account.hashed_password):
        access_token = create_access_token(
            username=account.username, role=account.role, full_name=account.full_name, student_id=account.student_id
        )
        redirect_url = "/" if account.role in {"admin", "teacher"} else "/my_attendance"
        response = RedirectResponse(url=redirect_url, status_code=status.HTTP_302_FOUND)
        response.set_cookie(
            key="access_token",
            value=access_token,
            httponly=True,
            samesite="lax",
            secure=(request.url.scheme == "https"),
        )
        write_audit_log(
            actor_username=account.username,
            actor_role=account.role,
            action="login",
            resource="/login",
            status="success",
            ip=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent"),
        )
        return response

    legacy_admin = db.query(AdminUser).filter(AdminUser.username == username).first()
    if legacy_admin and verify_password(password, legacy_admin.hashed_password):
        access_token = create_access_token(username=legacy_admin.username, role="admin", full_name=legacy_admin.full_name)
        response = RedirectResponse(url="/", status_code=status.HTTP_302_FOUND)
        response.set_cookie(
            key="access_token",
            value=access_token,
            httponly=True,
            samesite="lax",
            secure=(request.url.scheme == "https"),
        )
        write_audit_log(
            actor_username=legacy_admin.username,
            actor_role="admin",
            action="login",
            resource="/login",
            status="success",
            ip=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent"),
        )
        return response

    write_audit_log(
        actor_username=username,
        actor_role=None,
        action="login",
        resource="/login",
        status="failed",
        ip=request.client.host if request.client else None,
        user_agent=request.headers.get("user-agent"),
    )
    return templates.TemplateResponse("login.html", {"request": request, "error": "用户名或密码错误"})

@app.get("/logout")
async def logout(request: Request, user=Depends(login_required)):
    response = RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)
    response.delete_cookie("access_token")
    write_audit_log(
        actor_username=user.get("username"),
        actor_role=user.get("role"),
        action="logout",
        resource="/logout",
        status="success",
        ip=request.client.host if request.client else None,
        user_agent=request.headers.get("user-agent"),
    )
    return response

@app.get("/")
async def index(request: Request, user=Depends(require_roles("admin", "teacher")), db: Session = Depends(get_db)):
    if user.get("role") == "teacher":
        teacher_key = user.get("full_name") or user.get("username")
        courses = db.query(Course).filter(Course.teacher == teacher_key, Course.course_no != "UNKNOWN").all()
    else:
        courses = db.query(Course).filter(Course.course_no != "UNKNOWN").all()

    return templates.TemplateResponse("index.html", {"request": request, "user": user, "courses": courses, "title": "实时监控"})

@app.get("/my_attendance")
async def my_attendance_page(request: Request, user=Depends(require_roles("student")), db: Session = Depends(get_db)):
    student_id = user.get("student_id")
    records = []
    if student_id:
        rows = (
            db.query(Attendance, Course)
            .outerjoin(Course, Attendance.course_id == Course.course_id)
            .filter(Attendance.student_id == student_id)
            .order_by(Attendance.check_time.desc())
            .limit(200)
            .all()
        )
        for r, c in rows:
            course_label = c.course_name if c else "-"
            ts = r.check_time
            records.append(
                {
                    "course_name": course_label,
                    "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S") if ts else "-",
                    "status": r.status,
                }
            )

    return templates.TemplateResponse(
        "my_attendance.html",
        {"request": request, "user": user, "records": records, "title": "我的考勤"},
    )

@app.get("/my_courses")
async def my_courses_page(request: Request, user=Depends(require_roles("teacher")), db: Session = Depends(get_db)):
    teacher_key = user.get("full_name") or user.get("username")
    courses = db.query(Course).filter(Course.teacher == teacher_key).all()
    course_rows = []
    for c in courses:
        total_records = db.query(Attendance).filter(Attendance.course_id == c.course_id).count()
        unique_students = (
            db.query(Attendance.student_id).filter(Attendance.course_id == c.course_id).distinct().count()
        )
        course_rows.append(
            {
                "id": c.course_id,
                "name": c.course_name,
                "code": c.course_no,
                "schedule_time": c.schedule or "-",
                "total_records": total_records,
                "unique_students": unique_students,
            }
        )

    return templates.TemplateResponse(
        "my_courses.html",
        {"request": request, "user": user, "courses": course_rows, "title": "我的课程"},
    )

@app.get("/students")
async def students_page(
    request: Request,
    class_name: str = None,
    search_query: str = None,
    page: int = 1,
    page_size: int = 20,
    user=Depends(require_roles("admin")),
    db: Session = Depends(get_db),
):
    try:
        page = int(page)
    except Exception:
        page = 1
    page = max(1, page)
    try:
        page_size = int(page_size)
    except Exception:
        page_size = 20
    page_size = min(200, max(10, page_size))

    query = db.query(Student)
    query = query.filter(Student.student_no != "UNKNOWN")
    if class_name:
        query = query.filter(Student.class_name == class_name)
    if search_query:
        q = search_query.strip()
        if q:
            query = query.filter(or_(Student.name.like(f"%{q}%"), Student.student_no.like(f"%{q}%")))

    total_count = query.count()
    total_pages = max(1, (total_count + page_size - 1) // page_size)
    page = min(page, total_pages)

    students = (
        query.order_by(Student.created_at.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )
    
    classes = db.query(Student.class_name).distinct().all()
    classes = [c[0] for c in classes if c[0]]
    
    return templates.TemplateResponse(
        "students.html",
        {
            "request": request,
            "students": students,
            "classes": classes,
            "current_class": class_name,
            "current_search": search_query,
            "page": page,
            "page_size": page_size,
            "total_count": total_count,
            "total_pages": total_pages,
            "user": user,
            "title": "学生数据管理",
        },
    )

@app.get("/history")
async def history_page(
    request: Request, 
    college: str = None, 
    search_query: str = None, 
    course_id: int = None,
    course_name: str = None,
    start_date: str = None,
    end_date: str = None,
    page: int = 1,
    page_size: int = 20,
    user=Depends(require_roles("admin", "teacher")),
    db: Session = Depends(get_db)
):
    if not course_id and course_name:
        course_id = resolve_course_id(db, course_name)

    teacher_course_ids = None
    if user.get("role") == "teacher":
        teacher_key = user.get("full_name") or user.get("username")
        teacher_course_ids = get_teacher_course_ids(db, teacher_key)

    try:
        page = int(page)
    except Exception:
        page = 1
    page = max(1, page)
    try:
        page_size = int(page_size)
    except Exception:
        page_size = 20
    page_size = min(200, max(10, page_size))

    start_time = None
    end_time = None
    start_time = parse_date_param(start_date, end=False)
    end_time = parse_date_param(end_date, end=True)

    base_q = build_attendance_query(
        db,
        course_id=course_id,
        teacher_course_ids=teacher_course_ids,
        college=college,
        search_query=search_query,
        start_time=start_time,
        end_time=end_time,
    )
    total_count = base_q.count()
    avg_confidence = base_q.with_entities(func.avg(Attendance.confidence)).scalar()

    roster_ids: set[int] = set()
    absent_list: list[dict] = []
    if course_id:
        course = db.query(Course).filter(Course.course_id == int(course_id)).first()
        if course and course.class_names:
            class_names = [x.strip() for x in str(course.class_names).split(",") if x.strip()]
            if class_names:
                roster_q = db.query(Student.student_id).filter(Student.student_no != "UNKNOWN", Student.class_name.in_(class_names))
                if college:
                    roster_q = roster_q.filter(Student.college == college)
                roster_ids = {int(r[0]) for r in roster_q.all() if r[0] is not None}

    expected_count = len(roster_ids) if roster_ids else db.query(Student.student_id).filter(Student.student_no != "UNKNOWN").count()

    stats_q = build_attendance_query(
        db,
        course_id=course_id,
        teacher_course_ids=teacher_course_ids,
        college=college,
        search_query=None,
        start_time=start_time,
        end_time=end_time,
    )
    if roster_ids:
        stats_q = stats_q.filter(Attendance.student_id.in_(list(roster_ids)))
    actual_ids_rows = stats_q.with_entities(Attendance.student_id).distinct().all()
    actual_ids = {int(r[0]) for r in actual_ids_rows if r[0] is not None}
    actual_count = len(actual_ids)
    attendance_rate = int((actual_count / expected_count) * 100) if expected_count else 0

    absent_count = 0
    if roster_ids:
        absent_ids = roster_ids - actual_ids
        absent_count = len(absent_ids)
        if absent_ids:
            rows = (
                db.query(Student.student_no, Student.name, Student.class_name)
                .filter(Student.student_id.in_(list(absent_ids)))
                .order_by(Student.class_name, Student.name)
                .limit(80)
                .all()
            )
            absent_list = [{"student_no": r[0] or "-", "name": r[1] or "-", "class_name": r[2] or "-"} for r in rows]

    records = (
        base_q.offset((page - 1) * page_size).limit(page_size).all()
    )
    
    history_data = []
    for a, s, c in records:
        ts = a.check_time
        history_data.append({
            "record_id": a.record_id,
            "name": s.name,
            "student_id": s.student_no or "-",
            "class_name": s.class_name or "-",
            "college": s.college or "-",
            "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S") if ts else "-",
            "status": a.status,
            "course_name": c.course_name if c else "-",
            "confidence": a.confidence,
        })
        
    colleges = db.query(Student.college).distinct().all()
    colleges = [c[0] for c in colleges if c[0]]
    courses = (
        db.query(Course).filter(Course.course_no != "UNKNOWN").all()
        if user.get("role") == "admin"
        else db.query(Course)
        .filter(Course.teacher == (user.get("full_name") or user.get("username")), Course.course_no != "UNKNOWN")
        .all()
    )
    course_map = {c.course_id: c.course_name for c in courses}

    return templates.TemplateResponse("history.html", {
        "request": request, 
        "records": history_data, 
        "colleges": colleges,
        "courses": courses,
        "current_college": college,
        "current_course_id": course_id,
        "current_search": search_query,
        "current_start_date": start_date,
        "current_end_date": end_date,
        "stats": {
            "total_count": total_count,
            "attendance_rate": attendance_rate,
            "avg_confidence": float(avg_confidence) if avg_confidence is not None else None,
            "expected_count": expected_count,
            "actual_count": actual_count,
            "absent_count": absent_count,
            "absent_list": absent_list,
            "course_name": course_map.get(int(course_id)) if course_id else None,
        },
        "page": page,
        "page_size": page_size,
        "total_pages": max(1, (total_count + page_size - 1) // page_size),
        "user": user,
        "title": "考勤记录"
    })


@app.get("/attendance_tasks")
async def attendance_tasks_page(request: Request, user=Depends(require_roles("admin", "teacher")), db: Session = Depends(get_db)):
    teacher_key = user.get("full_name") or user.get("username")
    if user.get("role") == "teacher":
        courses = db.query(Course).filter(Course.teacher == teacher_key, Course.course_no != "UNKNOWN").all()
        course_ids = [c.course_id for c in courses]
        tasks = (
            db.query(AttendanceTask)
            .filter(AttendanceTask.course_id.in_(course_ids))
            .order_by(AttendanceTask.created_at.desc())
            .limit(500)
            .all()
        )
    else:
        courses = db.query(Course).filter(Course.course_no != "UNKNOWN").all()
        tasks = db.query(AttendanceTask).order_by(AttendanceTask.created_at.desc()).limit(500).all()

    class_rows = db.query(Student.class_name).distinct().all()
    classes = [r[0] for r in class_rows if r[0]]

    course_map = {c.course_id: c.course_name for c in courses}
    items = []
    for t in tasks:
        items.append(
            {
                "task_id": t.task_id,
                "title": t.title,
                "course_id": t.course_id,
                "course_name": course_map.get(t.course_id) or "-",
                "class_name": t.class_name,
                "status": t.status,
                "start_time": t.start_time.strftime("%Y-%m-%d %H:%M:%S") if t.start_time else None,
                "end_time": t.end_time.strftime("%Y-%m-%d %H:%M:%S") if t.end_time else None,
            }
        )

    return templates.TemplateResponse(
        "attendance_tasks.html",
        {"request": request, "user": user, "courses": courses, "classes": classes, "tasks": items, "title": "考勤任务"},
    )


@app.get("/api/attendance_tasks")
async def attendance_tasks_api(
    status: str = None,
    user=Depends(require_roles("admin", "teacher")),
    db: Session = Depends(get_db),
):
    teacher_key = user.get("full_name") or user.get("username")
    q = db.query(AttendanceTask)
    if user.get("role") == "teacher":
        course_ids = [r[0] for r in db.query(Course.course_id).filter(Course.teacher == teacher_key).all()]
        q = q.filter(AttendanceTask.course_id.in_(course_ids)) if course_ids else q.filter(False)
    if status:
        q = q.filter(AttendanceTask.status == status)
    tasks = q.order_by(AttendanceTask.created_at.desc()).limit(500).all()
    course_rows = db.query(Course.course_id, Course.course_name).all()
    course_map = {r[0]: r[1] for r in course_rows}
    data = []
    for t in tasks:
        data.append(
            {
                "task_id": t.task_id,
                "title": t.title,
                "course_id": t.course_id,
                "course_name": course_map.get(t.course_id),
                "class_name": t.class_name,
                "status": t.status,
                "start_time": t.start_time.isoformat() if t.start_time else None,
                "end_time": t.end_time.isoformat() if t.end_time else None,
            }
        )
    return {"tasks": data}


@app.post("/api/attendance_tasks")
async def create_attendance_task(
    request: Request,
    title: str = Form(...),
    course_id: int = Form(...),
    user=Depends(require_roles("admin", "teacher")),
    db: Session = Depends(get_db),
):
    teacher_key = user.get("full_name") or user.get("username")
    course = db.query(Course).filter(Course.course_id == course_id).first()
    if not course:
        return RedirectResponse(url="/attendance_tasks", status_code=303)
    if user.get("role") == "teacher" and course.teacher != teacher_key:
        raise HTTPException(status_code=403, detail="无权限创建该课程的任务")

    form = await request.form()
    class_names = [str(x).strip() for x in (form.getlist("class_names") if form else []) if str(x).strip()]
    title = (title or "").strip()
    if not title or not class_names:
        return RedirectResponse(url="/attendance_tasks", status_code=303)

    students = db.query(Student.student_id).filter(Student.class_name.in_(class_names)).all()
    student_ids = [int(r[0]) for r in students if r[0] is not None]
    if not student_ids:
        return RedirectResponse(url="/attendance_tasks", status_code=303)

    class_label = ",".join(class_names)
    task = AttendanceTask(
        title=title,
        course_id=int(course_id),
        class_name=class_label,
        status="draft",
        created_by=user.get("username"),
        created_at=datetime.now(),
    )
    db.add(task)
    db.flush()
    for sid in student_ids:
        db.add(AttendanceTaskStudent(task_id=task.task_id, student_id=int(sid), created_at=datetime.now()))
    db.commit()
    write_audit_log(
        actor_username=user.get("username"),
        actor_role=user.get("role"),
        action="create_attendance_task",
        resource="/api/attendance_tasks",
        status="success",
        ip=request.client.host if request.client else None,
        user_agent=request.headers.get("user-agent"),
        details={"task_id": task.task_id, "course_id": course_id, "class_names": class_names},
    )
    return RedirectResponse(url="/attendance_tasks", status_code=303)


@app.post("/api/attendance_tasks/{task_id}/start")
async def start_attendance_task(
    request: Request,
    task_id: int,
    user=Depends(require_roles("admin", "teacher")),
    db: Session = Depends(get_db),
):
    teacher_key = user.get("full_name") or user.get("username")
    task = db.query(AttendanceTask).filter(AttendanceTask.task_id == task_id).first()
    if not task:
        return RedirectResponse(url="/attendance_tasks", status_code=303)
    if user.get("role") == "teacher":
        course = db.query(Course).filter(Course.course_id == task.course_id).first()
        if not course or course.teacher != teacher_key:
            raise HTTPException(status_code=403, detail="无权限")
    if task.status != "running":
        task.status = "running"
        task.start_time = datetime.now()
        task.end_time = None
        db.add(task)
        db.commit()
    return RedirectResponse(url="/attendance_tasks", status_code=303)


@app.post("/api/attendance_tasks/{task_id}/close")
async def close_attendance_task(
    request: Request,
    task_id: int,
    user=Depends(require_roles("admin", "teacher")),
    db: Session = Depends(get_db),
):
    teacher_key = user.get("full_name") or user.get("username")
    task = db.query(AttendanceTask).filter(AttendanceTask.task_id == task_id).first()
    if not task:
        return RedirectResponse(url="/attendance_tasks", status_code=303)
    if user.get("role") == "teacher":
        course = db.query(Course).filter(Course.course_id == task.course_id).first()
        if not course or course.teacher != teacher_key:
            raise HTTPException(status_code=403, detail="无权限")
    if task.status == "running":
        task.status = "closed"
        task.end_time = datetime.now()
        db.add(task)
        db.commit()
    return RedirectResponse(url="/attendance_tasks", status_code=303)


@app.post("/api/attendance_tasks/{task_id}/delete")
async def delete_attendance_task(
    request: Request,
    task_id: int,
    user=Depends(require_roles("admin", "teacher")),
    db: Session = Depends(get_db),
):
    teacher_key = user.get("full_name") or user.get("username")
    task = db.query(AttendanceTask).filter(AttendanceTask.task_id == task_id).first()
    if not task:
        return RedirectResponse(url="/attendance_tasks", status_code=303)
    if user.get("role") == "teacher":
        course = db.query(Course).filter(Course.course_id == task.course_id).first()
        if not course or course.teacher != teacher_key:
            raise HTTPException(status_code=403, detail="无权限")
    if task.status == "running":
        return RedirectResponse(url="/attendance_tasks", status_code=303)
    deleted = {"task_id": task.task_id, "course_id": task.course_id, "class_name": task.class_name, "status": task.status}
    db.delete(task)
    db.commit()
    write_audit_log(
        actor_username=user.get("username"),
        actor_role=user.get("role"),
        action="delete_attendance_task",
        resource="/api/attendance_tasks/{task_id}/delete",
        status="success",
        ip=request.client.host if request.client else None,
        user_agent=request.headers.get("user-agent"),
        details=deleted,
    )
    return RedirectResponse(url="/attendance_tasks", status_code=303)

@app.get("/courses")
async def courses_page(request: Request, user=Depends(require_roles("admin")), db: Session = Depends(get_db)):
    courses = db.query(Course).filter(Course.course_no != "UNKNOWN").all()
    class_rows = db.query(Student.class_name).distinct().all()
    classes = [r[0] for r in class_rows if r[0]]
    return templates.TemplateResponse("courses.html", {
        "request": request, 
        "courses": courses,
        "classes": classes,
        "user": user,
        "title": "课程数据管理"
    })

@app.post("/api/courses")
async def add_course(
    request: Request,
    name: str = Form(...),
    code: str = Form(...),
    teacher: str = Form(...),
    start_time: str = Form(None),
    end_time: str = Form(None),
    schedule: str = Form(None),
    location: str = Form(None),
    db: Session = Depends(get_db),
    user=Depends(require_roles("admin"))
):
    try:
        form = await request.form()
        class_names = [str(x).strip() for x in (form.getlist("class_names") if form else []) if str(x).strip()]
        class_label = ",".join(class_names) if class_names else None

        start_dt = None
        end_dt = None
        if start_time:
            try:
                start_dt = datetime.fromisoformat(str(start_time))
            except Exception:
                start_dt = None
        if end_time:
            try:
                end_dt = datetime.fromisoformat(str(end_time))
            except Exception:
                end_dt = None

        schedule_text = (schedule or "").strip() if schedule else None
        if start_dt:
            if end_dt:
                schedule_text = f"{start_dt.strftime('%Y-%m-%d %H:%M')} - {end_dt.strftime('%Y-%m-%d %H:%M')}"
            else:
                schedule_text = start_dt.strftime("%Y-%m-%d %H:%M")

        course = Course(
            course_name=name,
            course_no=code,
            schedule=schedule_text,
            start_time=start_dt,
            end_time=end_dt,
            location=location,
            teacher=teacher,
            class_names=class_label,
        )
        db.add(course)
        db.commit()
        write_audit_log(
            actor_username=user.get("username"),
            actor_role=user.get("role"),
            action="create_course",
            resource="/api/courses",
            status="success",
            ip=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent"),
            details={"name": name, "code": code, "teacher": teacher, "schedule": schedule_text, "location": location, "class_names": class_names, "start_time": start_time, "end_time": end_time},
        )
        return RedirectResponse(url="/courses", status_code=303)
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/api/delete_course")
async def delete_course(request: Request, course_id: int = Form(...), db: Session = Depends(get_db), user=Depends(require_roles("admin"))):
    course = db.query(Course).filter(Course.course_id == course_id).first()
    if not course:
        return RedirectResponse(url="/courses", status_code=303)
    deleted = {"id": course.course_id, "name": course.course_name, "code": course.course_no}
    try:
        for i in range(3):
            try:
                db.query(Attendance).filter(Attendance.course_id == int(course_id)).delete(synchronize_session=False)
                db.query(CourseStudent).filter(CourseStudent.course_id == int(course_id)).delete(synchronize_session=False)
                db.query(AttendanceTask).filter(AttendanceTask.course_id == int(course_id)).delete(synchronize_session=False)
                db.delete(course)
                db.commit()
                break
            except OperationalError as oe:
                try:
                    db.rollback()
                except Exception:
                    pass
                if i < 2 and "locked" in str(oe).lower():
                    time.sleep(0.15 * (i + 1))
                    continue
                raise
        write_audit_log(
            actor_username=user.get("username"),
            actor_role=user.get("role"),
            action="delete_course",
            resource="/api/delete_course",
            status="success",
            ip=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent"),
            details=deleted,
        )
    except Exception as e:
        try:
            db.rollback()
        except Exception:
            pass
        write_audit_log(
            actor_username=user.get("username"),
            actor_role=user.get("role"),
            action="delete_course",
            resource="/api/delete_course",
            status="error",
            ip=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent"),
            details={"course": deleted, "error": str(e)},
        )
    return RedirectResponse(url="/courses", status_code=303)

@app.get("/api/export_attendance")
async def export_attendance(
    request: Request,
    college: str = None,
    search_query: str = None,
    task_id: int = None,
    course_id: int = None,
    course_name: str = None,
    start_date: str = None,
    end_date: str = None,
    user=Depends(require_roles("admin", "teacher")),
    db: Session = Depends(get_db),
):
    if not course_id and course_name:
        course_id = resolve_course_id(db, course_name)
    try:
        task_id = int(task_id) if task_id not in (None, "") else None
    except Exception:
        task_id = None

    teacher_course_ids = None
    if user.get("role") == "teacher":
        teacher_key = user.get("full_name") or user.get("username")
        teacher_course_ids = get_teacher_course_ids(db, teacher_key)

    start_time = parse_date_param(start_date, end=False)
    end_time = parse_date_param(end_date, end=True)

    records = query_attendance_joined(
        db,
        task_id=task_id,
        course_id=course_id,
        teacher_course_ids=teacher_course_ids,
        college=college,
        search_query=search_query,
        start_time=start_time,
        end_time=end_time,
        limit=200000,
    )
    
    data = []
    for a, s, c in records:
        course_label = c.course_name if c else "-"
        ts = a.check_time
        data.append({
            "姓名": s.name,
            "学号": s.student_no or "-",
            "学院": s.college,
            "班级": s.class_name,
            "课程": course_label,
            "打卡时间": ts.strftime("%Y-%m-%d %H:%M:%S") if ts else "-",
            "置信度": a.confidence,
            "状态": a.status
        })
        
    df = pd.DataFrame(data)
    
    stream = io.BytesIO()
    df.to_excel(stream, index=False)
    stream.seek(0)
    
    response = StreamingResponse(stream, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    response.headers["Content-Disposition"] = "attachment; filename=attendance_report.xlsx"
    write_audit_log(
        actor_username=user.get("username"),
        actor_role=user.get("role"),
        action="export_attendance",
        resource="/api/export_attendance",
        status="success",
        ip=request.client.host if request.client else None,
        user_agent=request.headers.get("user-agent"),
        details={
            "college": college,
            "search_query": search_query,
            "course_id": course_id,
            "course_name": course_name,
            "start_date": start_date,
            "end_date": end_date,
        },
    )
    return response


@app.post("/api/delete_attendance")
async def delete_attendance(
    request: Request,
    user=Depends(require_roles("admin")),
    db: Session = Depends(get_db),
):
    form = await request.form()
    raw_ids = form.getlist("ids") if form else []
    ids = []
    for x in raw_ids:
        try:
            ids.append(int(x))
        except Exception:
            pass
    if not ids:
        return RedirectResponse(url="/history", status_code=303)
    deleted_count = db.query(Attendance).filter(Attendance.record_id.in_(ids)).delete(synchronize_session=False)
    db.commit()
    write_audit_log(
        actor_username=user.get("username"),
        actor_role=user.get("role"),
        action="delete_attendance",
        resource="/api/delete_attendance",
        status="success",
        ip=request.client.host if request.client else None,
        user_agent=request.headers.get("user-agent"),
        details={"deleted_count": int(deleted_count), "ids": ids[:50]},
    )
    return RedirectResponse(url="/history", status_code=303)


@app.get("/users")
async def users_page(request: Request, user=Depends(require_roles("admin")), db: Session = Depends(get_db)):
    error = request.query_params.get("error")
    success = request.query_params.get("success")
    accounts = db.query(UserAccount).order_by(UserAccount.created_at.desc()).all()
    items = []
    for a in accounts:
        student_display = None
        if a.student_id:
            stu = db.query(Student).filter(Student.student_id == a.student_id).first()
            if stu:
                student_display = f"{stu.name}({stu.student_no or '-'})"
        items.append(
            {
                "id": a.id,
                "username": a.username,
                "role": a.role,
                "full_name": a.full_name,
                "student_display": student_display,
            }
        )
    return templates.TemplateResponse(
        "users.html",
        {"request": request, "user": user, "accounts": items, "error": error, "success": success, "title": "权限管理"},
    )

@app.get("/audit")
async def audit_page(request: Request, user=Depends(require_roles("admin")), db: Session = Depends(get_db)):
    rows = db.query(AuditLog).order_by(AuditLog.created_at.desc()).limit(500).all()
    items = []
    for r in rows:
        items.append(
            {
                "id": r.id,
                "created_at": r.created_at.strftime("%Y-%m-%d %H:%M:%S") if r.created_at else "-",
                "actor_username": r.actor_username,
                "actor_role": r.actor_role,
                "action": r.action,
                "resource": r.resource,
                "status": r.status,
                "ip": r.ip,
            }
        )
    return templates.TemplateResponse(
        "audit.html",
        {"request": request, "user": user, "rows": items, "title": "审计日志"},
    )


@app.get("/api/audit/export")
async def export_audit_logs(
    request: Request,
    ids: str = None,
    user=Depends(require_roles("admin")),
    db: Session = Depends(get_db),
):
    query = db.query(AuditLog)
    selected_ids = []
    if ids:
        try:
            selected_ids = [int(x) for x in ids.split(",") if x.strip().isdigit()]
        except Exception:
            selected_ids = []
    if selected_ids:
        query = query.filter(AuditLog.id.in_(selected_ids))
    rows = query.order_by(AuditLog.created_at.desc()).limit(5000).all()

    data = []
    for r in rows:
        data.append(
            {
                "时间": r.created_at.strftime("%Y-%m-%d %H:%M:%S") if r.created_at else None,
                "用户名": r.actor_username,
                "角色": r.actor_role,
                "动作": r.action,
                "资源": r.resource,
                "状态": r.status,
                "IP": r.ip,
                "User-Agent": r.user_agent,
                "详情": r.details,
            }
        )
    df = pd.DataFrame(data)
    stream = io.BytesIO()
    df.to_excel(stream, index=False)
    stream.seek(0)

    response = StreamingResponse(stream, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    response.headers["Content-Disposition"] = "attachment; filename=audit_logs.xlsx"
    return response


@app.post("/api/audit/delete")
async def delete_audit_logs(
    request: Request,
    user=Depends(require_roles("admin")),
    db: Session = Depends(get_db),
):
    form = await request.form()
    raw_ids = form.getlist("ids") if form else []
    ids = []
    for x in raw_ids:
        try:
            ids.append(int(x))
        except Exception:
            pass
    if not ids:
        return RedirectResponse(url="/audit", status_code=303)
    deleted_count = db.query(AuditLog).filter(AuditLog.id.in_(ids)).delete(synchronize_session=False)
    db.commit()
    return RedirectResponse(url="/audit", status_code=303)

@app.post("/api/users")
async def create_user(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    role: str = Form(...),
    full_name: str = Form(None),
    student_no: str = Form(None),
    user=Depends(require_roles("admin")),
    db: Session = Depends(get_db),
):
    role = role.strip().lower()
    if role not in {"admin", "teacher", "student"}:
        write_audit_log(
            actor_username=user.get("username"),
            actor_role=user.get("role"),
            action="create_user",
            resource="/api/users",
            status="failed",
            ip=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent"),
            details={"username": username, "role": role, "reason": "invalid_role"},
        )
        return RedirectResponse(url="/users?error=角色无效", status_code=303)
    if db.query(UserAccount).filter(UserAccount.username == username).first():
        write_audit_log(
            actor_username=user.get("username"),
            actor_role=user.get("role"),
            action="create_user",
            resource="/api/users",
            status="failed",
            ip=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent"),
            details={"username": username, "role": role, "reason": "username_exists"},
        )
        return RedirectResponse(url="/users?error=用户名已存在", status_code=303)
    if username == "admin" and role != "admin":
        write_audit_log(
            actor_username=user.get("username"),
            actor_role=user.get("role"),
            action="create_user",
            resource="/api/users",
            status="failed",
            ip=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent"),
            details={"username": username, "role": role, "reason": "admin_reserved"},
        )
        return RedirectResponse(url="/users?error=admin账号仅允许管理员角色", status_code=303)

    student_id = None
    if role == "student":
        if not student_no:
            write_audit_log(
                actor_username=user.get("username"),
                actor_role=user.get("role"),
                action="create_user",
                resource="/api/users",
                status="failed",
                ip=request.client.host if request.client else None,
                user_agent=request.headers.get("user-agent"),
                details={"username": username, "role": role, "reason": "missing_student_no"},
            )
            return RedirectResponse(url="/users?error=学生账号必须填写学号", status_code=303)
        stu = (
            db.query(Student)
            .filter(Student.student_no == student_no)
            .first()
        )
        if not stu:
            write_audit_log(
                actor_username=user.get("username"),
                actor_role=user.get("role"),
                action="create_user",
                resource="/api/users",
                status="failed",
                ip=request.client.host if request.client else None,
                user_agent=request.headers.get("user-agent"),
                details={"username": username, "role": role, "student_no": student_no, "reason": "student_not_found"},
            )
            return RedirectResponse(url="/users?error=未找到对应学号的学生信息", status_code=303)
        student_id = stu.student_id

    account = UserAccount(
        username=username,
        hashed_password=get_password_hash(password),
        role=role,
        full_name=full_name,
        student_id=student_id,
    )
    db.add(account)
    db.commit()
    write_audit_log(
        actor_username=user.get("username"),
        actor_role=user.get("role"),
        action="create_user",
        resource="/api/users",
        status="success",
        ip=request.client.host if request.client else None,
        user_agent=request.headers.get("user-agent"),
        details={"username": username, "role": role, "student_no": student_no},
    )
    return RedirectResponse(url="/users?success=账号创建成功", status_code=303)

@app.post("/api/delete_user")
async def delete_user(request: Request, user_id: int = Form(...), user=Depends(require_roles("admin")), db: Session = Depends(get_db)):
    account = db.query(UserAccount).filter(UserAccount.id == user_id).first()
    if not account:
        write_audit_log(
            actor_username=user.get("username"),
            actor_role=user.get("role"),
            action="delete_user",
            resource="/api/delete_user",
            status="failed",
            ip=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent"),
            details={"user_id": user_id, "reason": "not_found"},
        )
        return RedirectResponse(url="/users?error=账号不存在", status_code=303)
    if account.username == "admin":
        write_audit_log(
            actor_username=user.get("username"),
            actor_role=user.get("role"),
            action="delete_user",
            resource="/api/delete_user",
            status="failed",
            ip=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent"),
            details={"user_id": user_id, "reason": "admin_protected"},
        )
        return RedirectResponse(url="/users?error=不能删除admin账号", status_code=303)
    deleted_username = account.username
    deleted_role = account.role
    db.delete(account)
    db.commit()
    write_audit_log(
        actor_username=user.get("username"),
        actor_role=user.get("role"),
        action="delete_user",
        resource="/api/delete_user",
        status="success",
        ip=request.client.host if request.client else None,
        user_agent=request.headers.get("user-agent"),
        details={"user_id": user_id, "username": deleted_username, "role": deleted_role},
    )
    return RedirectResponse(url="/users?success=账号已删除", status_code=303)

@app.get("/settings")
async def settings_page(request: Request, user=Depends(require_roles("admin"))):
    # Scan for available models
    det_weights_dir = "models/weights/detection"
    rec_weights_dir = "models/weights/recognition"
    
    det_models = []
    if os.path.exists(det_weights_dir):
        # Scan .pt files
        files = glob.glob(os.path.join(det_weights_dir, "*.pt"))
        det_models = [os.path.basename(f) for f in files]
        
    rec_models = []
    if os.path.exists(rec_weights_dir):
        # Scan directories (AdaFace, ArcFace, etc)
        subdirs = [d for d in os.listdir(rec_weights_dir) if os.path.isdir(os.path.join(rec_weights_dir, d))]
        for d in subdirs:
            # Check if .pth exists inside
            pth_files = glob.glob(os.path.join(rec_weights_dir, d, "*.pth"))
            for pth in pth_files:
                # e.g. AdaFace/best.pth
                rec_models.append(f"{d}/{os.path.basename(pth)}")
    rec_models.sort()
                
    current_config = load_raw_config()
    current_config.setdefault("recognition", {})
    current_config["recognition"]["weights_path"] = _normalize_recognition_model_choice(
        current_config["recognition"].get("weights_path")
    )
            
    return templates.TemplateResponse("settings.html", {
        "request": request,
        "user": user,
        "det_models": det_models,
        "rec_models": rec_models,
        "config": current_config,
        "runtime_settings": get_runtime_settings(),
        "title": "系统设置"
    })

@app.post("/api/settings")
async def update_settings(
    request: Request,
    det_model: str = Form(...),
    rec_model: str = Form(...),
    similarity_threshold: float = Form(...),
    dedup_seconds: int = Form(60),
    capture_width: int = Form(640),
    capture_height: int = Form(480),
    frame_interval_ms: int = Form(200),
    jpeg_quality: float = Form(0.6),
    max_inference_concurrency: int = Form(2),
    max_ws_connections: int = Form(32),
    force_https: bool = Form(False),
    user=Depends(require_roles("admin"))
):
    try:
        config = load_raw_config()
        rec_model = _normalize_recognition_model_choice(rec_model)
        rec_model = str(rec_model or "").replace("\\", "/")
        if rec_model.lower().startswith("models/weights/recognition/"):
            rec_model = rec_model.split("models/weights/recognition/", 1)[1]
            
        # Update Config
        config.setdefault("detector", {})
        config.setdefault("recognition", {})
        config.setdefault("attendance", {})
        config.setdefault("capture", {})
        config.setdefault("performance", {})
        config.setdefault("security", {})
        similarity_threshold = round(float(similarity_threshold), 2)
        config["detector"]["model_path"] = f"models/weights/detection/{det_model}"
        config["recognition"]["weights_path"] = f"models/weights/recognition/{rec_model}"
        config["recognition"]["similarity_threshold"] = float(similarity_threshold)
        config["attendance"]["dedup_seconds"] = int(dedup_seconds)
        config["capture"]["width"] = int(capture_width)
        config["capture"]["height"] = int(capture_height)
        config["capture"]["frame_interval_ms"] = int(frame_interval_ms)
        config["capture"]["jpeg_quality"] = float(jpeg_quality)
        config["performance"]["max_inference_concurrency"] = int(max_inference_concurrency)
        config["performance"]["max_ws_connections"] = int(max_ws_connections)
        config["security"]["force_https"] = bool(force_https)
        
        save_raw_config(config)
            
        # Reload Service
        init_service()

        write_audit_log(
            actor_username=user.get("username"),
            actor_role=user.get("role"),
            action="update_settings",
            resource="/api/settings",
            status="success",
            ip=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent"),
            details={
                "det_model": det_model,
                "rec_model": rec_model,
                "similarity_threshold": float(similarity_threshold),
                "dedup_seconds": int(dedup_seconds),
                "capture": {
                    "width": int(capture_width),
                    "height": int(capture_height),
                    "frame_interval_ms": int(frame_interval_ms),
                    "jpeg_quality": float(jpeg_quality),
                },
                "performance": {
                    "max_inference_concurrency": int(max_inference_concurrency),
                    "max_ws_connections": int(max_ws_connections),
                },
                "security": {"force_https": bool(force_https)},
            },
        )
        
        return RedirectResponse(url="/settings?success=1", status_code=303)
        
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/api/runtime_settings")
async def runtime_settings_api(user=Depends(login_required)):
    return get_runtime_settings()

# ... (Keep existing websocket and other APIs) ...


# --- APIs ---

@app.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    settings = get_runtime_settings()
    max_ws = int(settings.get("performance", {}).get("max_ws_connections", 32))

    async with app.state.ws_lock:
        if app.state.active_ws >= max_ws:
            await websocket.close(code=1013)
            return
        app.state.active_ws += 1

    try:
        token = websocket.cookies.get("access_token")
        principal = None
        if token:
            try:
                payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
                username = payload.get("sub")
                role = payload.get("role")
                full_name = payload.get("full_name")
                if username and role:
                    principal = {"username": username, "role": role, "full_name": full_name}
            except JWTError:
                principal = None

        if not principal or principal.get("role") not in {"admin", "teacher"}:
            await websocket.close(code=1008)
            return

        async def get_inference_semaphore():
            cur = get_runtime_settings()
            limit = int(cur.get("performance", {}).get("max_inference_concurrency", 2))
            if not app.state.inference_semaphore or app.state.inference_semaphore_limit != limit:
                app.state.inference_semaphore = asyncio.Semaphore(limit)
                app.state.inference_semaphore_limit = limit
            return app.state.inference_semaphore

        async def load_teacher_courses() -> set[int] | None:
            if principal.get("role") != "teacher":
                return None
            teacher_name = principal.get("full_name") or principal.get("username")

            def sync():
                db = SessionLocal()
                try:
                    rows = db.query(Course.course_id).filter(Course.teacher == teacher_name).all()
                    return {r[0] for r in rows}
                finally:
                    db.close()

            return await asyncio.to_thread(sync)

        def resolve_course_id_sync(course_name: str) -> int | None:
            db = SessionLocal()
            try:
                row = (
                    db.query(Course.course_id)
                    .filter(Course.course_name == course_name)
                    .first()
                )
                return row[0] if row else None
            finally:
                db.close()

        teacher_course_ids = await load_teacher_courses()
        roster: dict[int, dict] = {}
        roster_ids: set[int] = set()
        current_course_id: int | None = None
        current_task_id: int | None = None
        last_stats_refresh = 0.0
        cached_stats = {"expected": 0, "actual": 0, "absent_count": 0, "absent_list": []}

        def load_course_students_sync(cid: int):
            db = SessionLocal()
            try:
                rows = (
                    db.query(Student.student_id, Student.student_no, Student.name)
                    .join(CourseStudent, CourseStudent.student_id == Student.student_id)
                    .filter(CourseStudent.course_id == cid)
                    .order_by(Student.name)
                    .all()
                )
                return rows
            finally:
                db.close()

        def load_course_roster_sync(cid: int):
            db = SessionLocal()
            try:
                course = db.query(Course).filter(Course.course_id == cid).first()
                if course and course.class_names:
                    class_names = [x.strip() for x in str(course.class_names).split(",") if x.strip()]
                    if class_names:
                        rows = (
                            db.query(Student.student_id, Student.student_no, Student.name)
                            .filter(Student.class_name.in_(class_names), Student.student_no != "UNKNOWN")
                            .order_by(Student.name)
                            .all()
                        )
                        if rows:
                            return rows
                return (
                    db.query(Student.student_id, Student.student_no, Student.name)
                    .join(CourseStudent, CourseStudent.student_id == Student.student_id)
                    .filter(CourseStudent.course_id == cid)
                    .order_by(Student.name)
                    .all()
                )
            finally:
                db.close()

        def load_task_roster_sync(tid: int):
            db = SessionLocal()
            try:
                rows = (
                    db.query(Student.student_id, Student.student_no, Student.name)
                    .join(AttendanceTaskStudent, AttendanceTaskStudent.student_id == Student.student_id)
                    .filter(AttendanceTaskStudent.task_id == tid)
                    .order_by(Student.name)
                    .all()
                )
                return rows
            finally:
                db.close()

        def resolve_task_sync(tid: int) -> AttendanceTask | None:
            db = SessionLocal()
            try:
                return db.query(AttendanceTask).filter(AttendanceTask.task_id == tid).first()
            finally:
                db.close()

        while True:
            raw = await websocket.receive_text()
            payload = None
            if raw.strip().startswith("{"):
                try:
                    payload = json.loads(raw)
                except Exception:
                    continue

            data = payload.get("image") if isinstance(payload, dict) else raw
            course_id = payload.get("course_id") if isinstance(payload, dict) else None
            task_id = payload.get("task_id") if isinstance(payload, dict) else None
            client_ts = payload.get("client_ts") if isinstance(payload, dict) else None

            try:
                course_id = int(course_id) if course_id not in (None, "") else None
            except Exception:
                course_id = None
            try:
                task_id = int(task_id) if task_id not in (None, "") else None
            except Exception:
                task_id = None
            try:
                client_ts = int(client_ts) if client_ts not in (None, "") else None
            except Exception:
                client_ts = None

            if task_id and task_id != current_task_id:
                t = await asyncio.to_thread(resolve_task_sync, int(task_id))
                allowed = True
                if not t:
                    allowed = False
                elif teacher_course_ids is not None and int(t.course_id) not in teacher_course_ids:
                    allowed = False
                if allowed:
                    current_task_id = int(task_id)
                    current_course_id = int(t.course_id)
                    roster = {}
                    roster_ids = set()
                    for sid, sno, sname in await asyncio.to_thread(load_task_roster_sync, int(task_id)):
                        roster[int(sid)] = {"student_no": sno, "name": sname}
                        roster_ids.add(int(sid))
                    cached_stats["expected"] = len(roster_ids)
                    last_stats_refresh = 0.0
                else:
                    current_task_id = None
                    current_course_id = None
                    roster = {}
                    roster_ids = set()
                    cached_stats = {"expected": 0, "actual": 0, "absent_count": 0, "absent_list": []}

            if course_id and course_id != current_course_id and not task_id:
                cid = int(course_id)
                allowed = teacher_course_ids is None or int(cid) in teacher_course_ids
                if allowed:
                    current_course_id = cid
                    current_task_id = None
                    roster = {}
                    roster_ids = set()
                    for sid, sno, sname in await asyncio.to_thread(load_course_roster_sync, cid):
                        roster[int(sid)] = {"student_no": sno, "name": sname}
                        roster_ids.add(int(sid))
                    cached_stats["expected"] = len(roster_ids)
                    last_stats_refresh = 0.0
                else:
                    current_task_id = None
                    current_course_id = None
                    roster = {}
                    roster_ids = set()
                    cached_stats = {"expected": 0, "actual": 0, "absent_count": 0, "absent_list": []}

            if teacher_course_ids is not None and current_course_id is not None and int(current_course_id) not in teacher_course_ids:
                current_course_id = None
                current_task_id = None
                roster = {}
                roster_ids = set()
                cached_stats = {"expected": 0, "actual": 0, "absent_count": 0, "absent_list": []}

            if not data:
                continue

            if "," in data:
                _, encoded = data.split(",", 1)
            else:
                encoded = data

            def decode_frame_sync(encoded_str: str):
                image_data = base64.b64decode(encoded_str)
                nparr = np.frombuffer(image_data, np.uint8)
                return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            try:
                frame = await asyncio.to_thread(decode_frame_sync, encoded)
            except Exception:
                continue

            if frame is None:
                continue

            t0 = time.monotonic()
            sem = await get_inference_semaphore()
            try:
                async with sem:
                    results = await asyncio.to_thread(service.recognize_frame, frame) if service else []
            except Exception:
                results = []
            processing_ms = (time.monotonic() - t0) * 1000.0

            current_faces = []
            now = datetime.now()
            dedup_seconds = int(get_runtime_settings().get("attendance", {}).get("dedup_seconds", 60))

            for res in results:
                name = res.get("name")
                sid = res.get("student_id")
                if sid is not None:
                    try:
                        sid = int(sid)
                    except Exception:
                        sid = None
                if sid and current_course_id:
                    enforce_roster = bool(roster_ids)
                    if enforce_roster and int(sid) not in roster_ids:
                        continue
                    key = (int(current_course_id), int(current_task_id or 0), int(sid))
                    async with app.state.dedup_lock:
                        last = app.state.dedup_cache.get(key)
                        if not last or (now - last).total_seconds() > dedup_seconds:
                            try:
                                app.state.attendance_queue.put_nowait(
                                    {
                                        "student_id": int(sid),
                                        "course_id": int(current_course_id),
                                        "task_id": int(current_task_id) if current_task_id else None,
                                        "confidence": res.get("score"),
                                        "check_time": now,
                                        "created_at": now,
                                        "status": "已签到",
                                    }
                                )
                                app.state.dedup_cache[key] = now
                            except Exception:
                                pass

                current_faces.append(
                    {
                        "box": res.get("box"),
                        "name": name,
                        "score": res.get("score"),
                    }
                )

            course_att = 0
            if current_course_id:
                async with app.state.dedup_lock:
                    course_att = sum(1 for (cid, tid, _), _t in app.state.dedup_cache.items() if cid == int(current_course_id) and int(tid) == int(current_task_id or 0))

                if time.monotonic() - last_stats_refresh >= 2.0:
                    start_at = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

                    def load_att_ids_sync(cid: int, tid: int | None, start: datetime):
                        db = SessionLocal()
                        try:
                            q = db.query(Attendance.student_id).filter(
                                Attendance.course_id == cid, 
                                Attendance.check_time >= start
                            )
                            if tid:
                                q = q.filter(Attendance.task_id == tid)
                            rows = q.distinct().all()
                            return [r[0] for r in rows]
                        finally:
                            db.close()

                    ids = await asyncio.to_thread(load_att_ids_sync, current_course_id, current_task_id, start_at)
                    actual_ids = {int(x) for x in ids if x is not None}
                    if roster_ids:
                        actual_ids = actual_ids & roster_ids
                    absent_ids = roster_ids - actual_ids
                    absent_list = []
                    for sid in list(absent_ids)[:50]:
                        info = roster.get(int(sid))
                        if info:
                            absent_list.append(info)
                    cached_stats = {
                        "expected": len(roster_ids),
                        "actual": len(actual_ids),
                        "absent_count": len(absent_ids),
                        "absent_list": absent_list,
                    }
                    last_stats_refresh = time.monotonic()

            try:
                await websocket.send_json(
                    {
                        "faces": current_faces,
                        "attendance_count": course_att,
                        "expected_count": cached_stats["expected"],
                        "actual_count": cached_stats["actual"],
                        "absent_count": cached_stats["absent_count"],
                        "absent_list": cached_stats["absent_list"],
                        "client_ts": client_ts,
                        "server_ts": int(time.time() * 1000),
                        "processing_ms": float(processing_ms),
                    }
                )
            except WebSocketDisconnect:
                break
            except Exception:
                break
    finally:
        async with app.state.ws_lock:
            app.state.active_ws = max(0, app.state.active_ws - 1)
        try:
            await websocket.close()
        except Exception:
            pass

@app.post("/api/register")
async def register_face(
    request: Request,
    name: str = Form(...), 
    student_no: str = Form(...),
    college: str = Form(None),
    gender: str = Form(None),
    class_name: str = Form(None),
    file: UploadFile = File(...),
    user=Depends(require_roles("admin"))
):
    if not service:
        return RedirectResponse(url="/students?error=人脸服务未初始化", status_code=303)
        
    # Ensure data/faces exists
    faces_dir = "data/faces"
    os.makedirs(faces_dir, exist_ok=True)
    
    try:
        name = (name or "").strip()
        student_no = (student_no or "").strip()
        if not name:
            return RedirectResponse(url="/students?error=姓名不能为空", status_code=303)
        if not student_no:
            return RedirectResponse(url="/students?error=学号不能为空", status_code=303)
        raw_bytes = await file.read()
        nparr = np.frombuffer(raw_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            return RedirectResponse(url="/students?error=图片解码失败", status_code=303)

        feature, detections = service.extract_primary_feature(image)
        if feature is None:
            return RedirectResponse(url="/students?error=未检测到清晰人脸，请正对镜头重试", status_code=303)
        if len(detections) > 1:
            logger.info(
                "Multiple faces detected during registration; using primary face",
                extra={"student_no": student_no, "face_count": len(detections)},
            )

        embedding_raw = feature.detach().cpu().numpy().astype(np.float32).tobytes()
        embedding_enc = encrypt_to_b64(embedding_raw)

        db = SessionLocal()
        try:
            student = db.query(Student).filter(Student.student_no == student_no).first()
            if not student:
                student = Student(
                    name=name, 
                    student_no=student_no,
                    college=college,
                    gender=gender,
                    class_name=class_name
                )
                db.add(student)
                db.flush()
            else:
                student.name = name
            if college:
                student.college = college
            if gender:
                student.gender = gender
            if class_name:
                student.class_name = class_name

            file_path = os.path.join(faces_dir, f"{student.student_id}.enc")
            with open(file_path, "wb") as f:
                f.write(encrypt_bytes(raw_bytes))
            student.face_image_path = file_path
            student.face_embedding_enc = embedding_enc
            if getattr(service, "model_sig", None):
                student.face_embedding_model_sig = service.model_sig
            service.upsert_known_face(student.student_id, student.name, feature, student.student_no)
            
            db.commit()
            write_audit_log(
                actor_username=user.get("username"),
                actor_role=user.get("role"),
                action="register_face",
                resource="/api/register",
                status="success",
                ip=request.client.host if request.client else None,
                user_agent=request.headers.get("user-agent"),
                details={"name": name, "student_no": student_no, "college": college, "gender": gender, "class_name": class_name},
            )
        finally:
            db.close()
        
        return RedirectResponse(url="/students", status_code=303)
            
    except Exception as e:
        logger.exception("register_face failed: %s", e)
        return RedirectResponse(url=f"/students?error={str(e)}", status_code=303)

@app.post("/api/delete_student")
async def delete_student(request: Request, student_id: int = Form(...), db: Session = Depends(get_db), user=Depends(require_roles("admin"))):
    try:
        stu = db.query(Student).filter(Student.student_id == student_id).first()
        if stu:
            deleted = {"student_id": stu.student_id, "name": stu.name, "student_no": stu.student_no}
            db.query(UserAccount).filter(UserAccount.student_id == student_id).update({"student_id": None})
            db.query(Attendance).filter(Attendance.student_id == student_id).delete(synchronize_session=False)
            # Delete file
            if stu.face_image_path and os.path.exists(stu.face_image_path):
                os.remove(stu.face_image_path)
            
            # Remove from DB
            db.delete(stu)
            db.commit()
            
            # Remove from memory cache
            if service:
                service.remove_known_face(stu.student_id)
            write_audit_log(
                actor_username=user.get("username"),
                actor_role=user.get("role"),
                action="delete_student",
                resource="/api/delete_student",
                status="success",
                ip=request.client.host if request.client else None,
                user_agent=request.headers.get("user-agent"),
                details=deleted,
            )
                
        return RedirectResponse(url="/students", status_code=303)
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/api/student_face")
async def student_face(student_id: int, db: Session = Depends(get_db), user=Depends(require_roles("admin"))):
    stu = db.query(Student).filter(Student.student_id == student_id).first()
    if not stu or not stu.face_image_path:
        raise HTTPException(status_code=404, detail="未找到照片")
    if not os.path.exists(stu.face_image_path):
        raise HTTPException(status_code=404, detail="照片文件不存在")
    try:
        token = open(stu.face_image_path, "rb").read()
        raw = decrypt_bytes(token)
    except Exception:
        raise HTTPException(status_code=500, detail="照片解密失败")
    media_type = "application/octet-stream"
    if raw.startswith(b"\xff\xd8"):
        media_type = "image/jpeg"
    elif raw.startswith(b"\x89PNG\r\n\x1a\n"):
        media_type = "image/png"
    resp = Response(content=raw, media_type=media_type)
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.post("/api/update_student")
async def update_student(
    request: Request,
    student_id: int = Form(...),
    student_no: str = Form(...),
    name: str = Form(...),
    gender: str = Form(None),
    class_name: str = Form(None),
    college: str = Form(None),
    file: UploadFile = File(None),
    user=Depends(require_roles("admin")),
):
    if not service:
        return {"status": "error", "message": "服务未初始化"}

    db = SessionLocal()
    try:
        stu = db.query(Student).filter(Student.student_id == student_id).first()
        if not stu:
            return RedirectResponse(url="/students?error=学生不存在", status_code=303)

        old_name = stu.name
        new_student_no = (student_no or "").strip()
        new_name = (name or "").strip()
        if not new_student_no:
            return RedirectResponse(url="/students?error=学号不能为空", status_code=303)
        if not new_name:
            return RedirectResponse(url="/students?error=姓名不能为空", status_code=303)
        exists = (
            db.query(Student.student_id)
            .filter(Student.student_no == new_student_no, Student.student_id != student_id)
            .first()
        )
        if exists:
            return RedirectResponse(url="/students?error=学号已存在", status_code=303)
        stu.student_no = new_student_no
        stu.name = new_name
        stu.gender = (gender or "").strip() or None
        stu.class_name = (class_name or "").strip() or None
        stu.college = (college or "").strip() or None

        if file and getattr(file, "filename", None):
            raw_bytes = await file.read()
            nparr = np.frombuffer(raw_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if image is None:
                return RedirectResponse(url="/students?error=图片解码失败", status_code=303)
            feature, detections = service.extract_primary_feature(image)
            if feature is None:
                return RedirectResponse(url="/students?error=未检测到清晰人脸，请正对镜头重试", status_code=303)
            if len(detections) > 1:
                logger.info(
                    "Multiple faces detected during student update; using primary face",
                    extra={"student_id": student_id, "face_count": len(detections)},
                )

            faces_dir = "data/faces"
            os.makedirs(faces_dir, exist_ok=True)
            file_path = os.path.join(faces_dir, f"{stu.student_id}.enc")
            with open(file_path, "wb") as f:
                f.write(encrypt_bytes(raw_bytes))

            stu.face_image_path = file_path
            embedding_raw = feature.detach().cpu().numpy().astype(np.float32).tobytes()
            stu.face_embedding_enc = encrypt_to_b64(embedding_raw)
            if getattr(service, "model_sig", None):
                stu.face_embedding_model_sig = service.model_sig
            service.upsert_known_face(stu.student_id, stu.name, feature, stu.student_no)

        if service:
            existing_feature = service.known_faces.get(int(stu.student_id))
            if existing_feature is not None:
                service.upsert_known_face(stu.student_id, stu.name, existing_feature, stu.student_no)

        db.commit()
        write_audit_log(
            actor_username=user.get("username"),
            actor_role=user.get("role"),
            action="update_student",
            resource="/api/update_student",
            status="success",
            ip=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent"),
            details={"student_id": student_id, "student_no": stu.student_no, "name": stu.name},
        )
        return RedirectResponse(url="/students", status_code=303)
    except Exception as e:
        db.rollback()
        logger.exception("update_student failed: %s", e)
        return RedirectResponse(url=f"/students?error={str(e)}", status_code=303)
    finally:
        db.close()


@app.get("/api/export_students")
async def export_students(
    request: Request,
    class_name: str = None,
    search_query: str = None,
    user=Depends(require_roles("admin")),
    db: Session = Depends(get_db),
):
    query = db.query(Student)
    if class_name:
        query = query.filter(Student.class_name == class_name)
    if search_query:
        q = search_query.strip()
        if q:
            query = query.filter(or_(Student.name.like(f"%{q}%"), Student.student_no.like(f"%{q}%")))
    rows = query.order_by(Student.created_at.desc()).all()

    data = []
    for s in rows:
        data.append(
            {
                "学号": s.student_no,
                "姓名": s.name,
                "班级": s.class_name,
                "性别": s.gender,
                "学院": s.college,
                "注册时间": s.created_at.strftime("%Y-%m-%d %H:%M:%S") if s.created_at else None,
            }
        )
    df = pd.DataFrame(data)
    stream = io.BytesIO()
    df.to_excel(stream, index=False)
    stream.seek(0)

    response = StreamingResponse(stream, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    response.headers["Content-Disposition"] = "attachment; filename=students.xlsx"
    write_audit_log(
        actor_username=user.get("username"),
        actor_role=user.get("role"),
        action="export_students",
        resource="/api/export_students",
        status="success",
        ip=request.client.host if request.client else None,
        user_agent=request.headers.get("user-agent"),
        details={"class_name": class_name, "search_query": search_query},
    )
    return response

@app.post("/api/import_students")
async def import_students(request: Request, file: UploadFile = File(...), db: Session = Depends(get_db), user=Depends(require_roles("admin"))):
    try:
        contents = await file.read()
        df = pd.read_excel(io.BytesIO(contents))
        
        # Check columns
        required_columns = ['姓名', '学号', '性别', '班级', '学院']
        if not all(col in df.columns for col in required_columns):
            return {"status": "error", "message": f"Excel必须包含列: {','.join(required_columns)}"}
            
        count = 0
        for _, row in df.iterrows():
            name = _normalize_tabular_text(row.get('姓名'))
            student_id = _normalize_tabular_text(row.get('学号'))
            if not name or not student_id:
                continue
            
            # Skip if exists (by student_id)
            if db.query(Student).filter(Student.student_no == student_id).first():
                continue
                
            new_student = Student(
                name=name,
                student_no=student_id,
                gender=_normalize_tabular_text(row.get('性别')) or None,
                class_name=_normalize_tabular_text(row.get('班级')) or None,
                college=_normalize_tabular_text(row.get('学院')) or None,
                face_image_path=None # No face initially
            )
            db.add(new_student)
            count += 1
            
        db.commit()
        write_audit_log(
            actor_username=user.get("username"),
            actor_role=user.get("role"),
            action="import_students",
            resource="/api/import_students",
            status="success",
            ip=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent"),
            details={"imported_count": count},
        )
        return RedirectResponse(url="/students", status_code=303)
    except Exception as e:
        return {"status": "error", "message": f"导入失败: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
