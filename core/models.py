from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Boolean, Text, Float
from sqlalchemy.orm import relationship
from datetime import datetime
from .database import Base

class AdminUser(Base):
    __tablename__ = "admin_users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    full_name = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.now)

class UserAccount(Base):
    __tablename__ = "user_accounts"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=True)
    role = Column(String, index=True, nullable=False)  # admin | teacher | student
    full_name = Column(String, nullable=True)
    student_id = Column(Integer, ForeignKey("students.student_id", ondelete="SET NULL"), nullable=True)
    created_at = Column(DateTime, default=datetime.now)

    student = relationship("Student")

class Course(Base):
    __tablename__ = "courses"
    course_id = Column(Integer, primary_key=True, index=True)
    course_no = Column(String, unique=True, index=True, nullable=False)
    course_name = Column(String, index=True, nullable=False)
    teacher = Column(String, nullable=True)
    schedule = Column(String, nullable=True)
    start_time = Column(DateTime, nullable=True)
    end_time = Column(DateTime, nullable=True)
    location = Column(String, nullable=True)
    class_names = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.now)

    attendances = relationship("Attendance", back_populates="course")
    course_students = relationship("CourseStudent", back_populates="course", cascade="all, delete-orphan")
    attendance_tasks = relationship("AttendanceTask", back_populates="course", cascade="all, delete-orphan")

class CourseStudent(Base):
    __tablename__ = "course_students"

    id = Column(Integer, primary_key=True, index=True)
    course_id = Column(Integer, ForeignKey("courses.course_id", ondelete="CASCADE"), index=True, nullable=False)
    student_id = Column(Integer, ForeignKey("students.student_id", ondelete="CASCADE"), index=True, nullable=False)
    created_at = Column(DateTime, default=datetime.now)

    course = relationship("Course", back_populates="course_students")
    student = relationship("Student", back_populates="course_students")

class Student(Base):
    __tablename__ = "students"

    student_id = Column(Integer, primary_key=True, index=True)
    student_no = Column(String, unique=True, index=True, nullable=False)
    name = Column(String, index=True, nullable=False)
    class_name = Column(String, nullable=True)
    college = Column(String, nullable=True) # 学院
    gender = Column(String, nullable=True) # 性别
    
    face_image_path = Column(String) # Path to the original registered image
    face_embedding_enc = Column(Text, nullable=True)
    face_embedding_model_sig = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.now)

    attendances = relationship("Attendance", back_populates="student", cascade="all, delete-orphan")
    course_students = relationship("CourseStudent", back_populates="student", cascade="all, delete-orphan")

class AttendanceTask(Base):
    __tablename__ = "attendance_tasks"

    task_id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=False)
    course_id = Column(Integer, ForeignKey("courses.course_id", ondelete="CASCADE"), index=True, nullable=False)
    class_name = Column(String, nullable=True)
    status = Column(String, nullable=False, index=True)
    start_time = Column(DateTime, nullable=True)
    end_time = Column(DateTime, nullable=True)
    created_by = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.now, index=True)

    course = relationship("Course", back_populates="attendance_tasks")
    task_students = relationship("AttendanceTaskStudent", back_populates="task", cascade="all, delete-orphan")
    attendances = relationship("Attendance", back_populates="task")

class AttendanceTaskStudent(Base):
    __tablename__ = "attendance_task_students"

    id = Column(Integer, primary_key=True, index=True)
    task_id = Column(Integer, ForeignKey("attendance_tasks.task_id", ondelete="CASCADE"), index=True, nullable=False)
    student_id = Column(Integer, ForeignKey("students.student_id", ondelete="CASCADE"), index=True, nullable=False)
    created_at = Column(DateTime, default=datetime.now)

    task = relationship("AttendanceTask", back_populates="task_students")
    student = relationship("Student")

class Attendance(Base):
    __tablename__ = "attendances"

    record_id = Column(Integer, primary_key=True, index=True)
    student_id = Column(Integer, ForeignKey("students.student_id", ondelete="CASCADE"), index=True, nullable=False)
    course_id = Column(Integer, ForeignKey("courses.course_id", ondelete="CASCADE"), index=True, nullable=False)
    task_id = Column(Integer, ForeignKey("attendance_tasks.task_id", ondelete="SET NULL"), index=True, nullable=True)
    check_time = Column(DateTime, default=datetime.now, index=True)
    confidence = Column(Float, nullable=True)
    status = Column(String, default="Present") # Present, Late, etc.
    created_at = Column(DateTime, default=datetime.now, index=True)
    
    student = relationship("Student", back_populates="attendances")
    course = relationship("Course", back_populates="attendances")
    task = relationship("AttendanceTask", back_populates="attendances")


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    actor_username = Column(String, index=True, nullable=True)
    actor_role = Column(String, index=True, nullable=True)
    action = Column(String, index=True, nullable=False)
    resource = Column(String, index=True, nullable=True)
    status = Column(String, nullable=True)
    ip = Column(String, nullable=True)
    user_agent = Column(String, nullable=True)
    details = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.now, index=True)
