import json
from datetime import datetime
import logging

from core.database import SessionLocal
from core.models import AuditLog

logger = logging.getLogger("systems.audit")


def write_audit_log(
    *,
    actor_username: str | None,
    actor_role: str | None,
    action: str,
    resource: str | None = None,
    status: str | None = None,
    ip: str | None = None,
    user_agent: str | None = None,
    details: dict | None = None,
) -> None:
    db = None
    try:
        db = SessionLocal()
        details_json = None
        if details is not None:
            try:
                details_json = json.dumps(details, ensure_ascii=False)
            except (TypeError, ValueError) as e:
                logger.warning(f"Failed to serialize audit details: {e}")
                details_json = json.dumps({"raw": str(details)}, ensure_ascii=False)

        row = AuditLog(
            actor_username=actor_username,
            actor_role=actor_role,
            action=action,
            resource=resource,
            status=status,
            ip=ip,
            user_agent=user_agent,
            details=details_json,
            created_at=datetime.now(),
        )
        db.add(row)
        db.commit()
    except Exception as e:
        if db is not None:
            try:
                db.rollback()
            except Exception:
                logger.exception("Audit log rollback failed")
        logger.exception("Failed to write audit log: %s", e)
    finally:
        if db is not None:
            try:
                db.close()
            except Exception:
                logger.exception("Audit log session close failed")

