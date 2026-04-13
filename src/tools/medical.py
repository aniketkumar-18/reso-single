"""Medical domain tools.

Exact Supabase schema (verified via PostgREST OpenAPI):
  medical_conditions: id, account_id, condition (NOT NULL), severity (NOT NULL default 'moderate'),
                      notes (NOT NULL default ''), diagnosed_at (NOT NULL default now())
  medications:        id, account_id, name (NOT NULL), dosage (NOT NULL), frequency (NOT NULL),
                      notes (NOT NULL default ''), started_at (NOT NULL default now()),
                      active (NOT NULL default true)
"""

from __future__ import annotations

from typing import Any, Literal

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from pydantic import BaseModel, Field

import src.infra.db as db
from src.infra.logger import setup_logger

logger = setup_logger(__name__)


def _ctx(config: RunnableConfig) -> tuple[str, str]:
    c = config.get("configurable", {})
    return c.get("user_id", ""), c.get("conversation_id", "")


# ── Medical conditions ─────────────────────────────────────────────────────────

class LogMedicalConditionInput(BaseModel):
    condition: str = Field(..., description="Name of the condition, e.g. 'Type 2 Diabetes'.")
    severity: Literal["mild", "moderate", "severe"] = Field("moderate")
    notes: str = Field("", description="Any extra context or management notes.")
    diagnosed_at: str | None = Field(None, description="ISO-8601 datetime. Null = now.")


@tool("log_medical_condition", args_schema=LogMedicalConditionInput)
async def log_medical_condition(
    condition: str, severity: str, notes: str, diagnosed_at: str | None,
    config: RunnableConfig,
) -> dict[str, Any]:
    """Log a medical condition for the user."""
    user_id, _ = _ctx(config)
    data: dict[str, Any] = {"condition": condition, "severity": severity, "notes": notes}
    if diagnosed_at:
        data["diagnosed_at"] = diagnosed_at
    result = await db.log_medical_condition(user_id, data)
    return {"status": "logged", "condition": result, "refresh": "medical-conditions"}


class UpdateMedicalConditionInput(BaseModel):
    condition: str = Field(..., description="Exact condition name to update.")
    updates: dict[str, Any] = Field(..., description="Fields to update (severity, notes, etc.).")


@tool("update_medical_condition", args_schema=UpdateMedicalConditionInput)
async def update_medical_condition(
    condition: str, updates: dict[str, Any], config: RunnableConfig,
) -> dict[str, Any]:
    """Update an existing medical condition record."""
    user_id, _ = _ctx(config)
    client = await db.get_client()
    if client is None:
        return {"status": "error", "message": "database not configured", "refresh": "medical-conditions"}
    try:
        result = (
            await client.table("medical_conditions")
            .update(updates)
            .eq("account_id", user_id)
            .eq("condition", condition)
            .execute()
        )
        return {"status": "updated", "condition": result.data[0] if result.data else {}, "refresh": "medical-conditions"}
    except Exception as exc:
        return {"status": "error", "message": str(exc), "refresh": "medical-conditions"}


# ── Medications ────────────────────────────────────────────────────────────────

class LogMedicationInput(BaseModel):
    name: str = Field(..., description="Drug name, e.g. 'Metformin'.")
    dosage: str = Field(..., description="Dose with units, e.g. '500mg'.")
    frequency: str = Field(..., description="How often taken, e.g. 'twice daily with meals'.")
    notes: str = Field("", description="Optional extra context.")
    started_at: str | None = Field(None, description="ISO-8601 datetime. Null = now.")


@tool("log_medication", args_schema=LogMedicationInput)
async def log_medication(
    name: str, dosage: str, frequency: str, notes: str, started_at: str | None,
    config: RunnableConfig,
) -> dict[str, Any]:
    """Log a medication for the user."""
    user_id, _ = _ctx(config)
    data: dict[str, Any] = {"name": name, "dosage": dosage, "frequency": frequency, "notes": notes}
    if started_at:
        data["started_at"] = started_at
    result = await db.log_medication(user_id, data)
    return {"status": "logged", "medication": result, "refresh": "medications"}


class UpdateMedicationInput(BaseModel):
    name: str = Field(..., description="Exact medication name to update.")
    updates: dict[str, Any] = Field(..., description="Fields to update (dosage, frequency, active, notes).")


@tool("update_medication", args_schema=UpdateMedicationInput)
async def update_medication(
    name: str, updates: dict[str, Any], config: RunnableConfig,
) -> dict[str, Any]:
    """Update an existing medication record."""
    user_id, _ = _ctx(config)
    client = await db.get_client()
    if client is None:
        return {"status": "error", "message": "database not configured", "refresh": "medications"}
    try:
        result = (
            await client.table("medications")
            .update(updates)
            .eq("account_id", user_id)
            .eq("name", name)
            .execute()
        )
        return {"status": "updated", "medication": result.data[0] if result.data else {}, "refresh": "medications"}
    except Exception as exc:
        return {"status": "error", "message": str(exc), "refresh": "medications"}


