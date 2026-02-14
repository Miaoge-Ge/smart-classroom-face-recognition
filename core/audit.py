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
    try:
        db = SessionLocal()
        try:
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
            db.rollback()
            logger.error(f"Failed to write audit log: {e}")
            raise
        finally:
            db.close()
    except Exception as e:
        logger.error(f"Audit log error: {e}")

