"""Fitness domain tools."""

from __future__ import annotations

import math
from datetime import datetime, timezone
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


class LogWorkoutInput(BaseModel):
    activity: str = Field(..., description="Type of exercise, e.g. 'running', 'yoga', 'cycling'.")
    duration_minutes: int = Field(..., description="Duration in minutes.", ge=1, le=600)
    intensity: Literal["low", "moderate", "high"] = Field("moderate")
    notes: str | None = Field(None)
    logged_at: str | None = Field(None, description="ISO-8601 timestamp. Null = now.")


@tool("log_workout", args_schema=LogWorkoutInput)
async def log_workout(activity: str, duration_minutes: int, intensity: str,
                      notes: str | None, logged_at: str | None,
                      config: RunnableConfig) -> dict[str, Any]:
    """Log a workout session for the user."""
    user_id, conversation_id = _ctx(config)
    # Exact workouts schema: type, duration_min, exercises, intensity, notes NOT NULL, logged_at NOT NULL
    workout_data: dict[str, Any] = {
        "type": activity,
        "duration_min": duration_minutes,
        "intensity": intensity,
        "exercises": [],
        "notes": notes or "",
        "logged_at": logged_at or datetime.now(timezone.utc).isoformat(),
    }
    result = await db.log_workout(user_id, conversation_id, workout_data)
    return {"status": "logged", "workout": result, "refresh": "workouts"}


class LogBodyMetricsInput(BaseModel):
    weight_kg: float | None = Field(None, description="Body weight in kg.")
    body_fat_percent: float | None = Field(None)
    waist_cm: float | None = Field(None)
    logged_at: str | None = Field(None)


@tool("log_body_metrics", args_schema=LogBodyMetricsInput)
async def log_body_metrics(weight_kg: float | None, body_fat_percent: float | None,
                           waist_cm: float | None, logged_at: str | None,
                           config: RunnableConfig) -> dict[str, Any]:
    """Log body measurements for the user."""
    user_id, _ = _ctx(config)
    client = await db.get_client()
    if client is None:
        return {"status": "error", "message": "database not configured", "refresh": "body-metrics"}
    # Exact body_metrics schema: weight_kg, body_fat_pct, waist_cm, notes NOT NULL, logged_at NOT NULL
    metrics: dict[str, Any] = {
        "account_id": user_id,
        "notes": "",
        "logged_at": logged_at or datetime.now(timezone.utc).isoformat(),
    }
    if weight_kg is not None: metrics["weight_kg"] = weight_kg
    if body_fat_percent is not None: metrics["body_fat_pct"] = body_fat_percent
    if waist_cm is not None: metrics["waist_cm"] = waist_cm
    try:
        result = await client.table("body_metrics").insert(metrics).execute()
        return {"status": "logged", "metrics": result.data[0] if result.data else {}, "refresh": "body-metrics"}
    except Exception as exc:
        return {"status": "error", "message": str(exc), "refresh": "body-metrics"}


class UpdateFitnessGoalsInput(BaseModel):
    goal_type: str = Field(..., description="E.g. 'weight_loss', 'muscle_gain', 'maintenance'.")
    target_weight_kg: float | None = Field(None)
    weekly_workout_days: int | None = Field(None, ge=1, le=7)
    daily_step_goal: int | None = Field(None)
    notes: str | None = Field(None)


@tool("update_fitness_goals", args_schema=UpdateFitnessGoalsInput)
async def update_fitness_goals(goal_type: str, target_weight_kg: float | None,
                               weekly_workout_days: int | None, daily_step_goal: int | None,
                               notes: str | None, config: RunnableConfig) -> dict[str, Any]:
    """Update or set the user's fitness goals."""
    user_id, _ = _ctx(config)
    client = await db.get_client()
    if client is None:
        return {"status": "error", "message": "database not configured", "refresh": "fitness-goals"}
    goals: dict[str, Any] = {"account_id": user_id, "goal": goal_type}  # existing column is "goal"
    if target_weight_kg is not None: goals["target_weight_kg"] = target_weight_kg
    if weekly_workout_days is not None: goals["weekly_workout_days"] = weekly_workout_days
    if daily_step_goal is not None: goals["daily_step_goal"] = daily_step_goal
    if notes: goals["notes"] = notes
    try:
        result = await client.table("fitness_profiles").upsert(goals, on_conflict="account_id").execute()
        return {"status": "updated", "goals": result.data[0] if result.data else {}, "refresh": "fitness-goals"}
    except Exception as exc:
        return {"status": "error", "message": str(exc), "refresh": "fitness-goals"}


class CalculateMacroTargetsInput(BaseModel):
    weight_kg: float = Field(..., description="Current body weight in kg.")
    height_cm: float = Field(..., description="Height in cm.")
    age: int = Field(..., description="Age in years.")
    sex: Literal["male", "female"] = Field(...)
    activity_level: Literal["sedentary", "light", "moderate", "active", "very_active"] = Field("moderate")
    goal: Literal["weight_loss", "maintenance", "muscle_gain"] = Field("maintenance")


_ACTIVITY = {"sedentary": 1.2, "light": 1.375, "moderate": 1.55, "active": 1.725, "very_active": 1.9}


@tool("calculate_macro_targets", args_schema=CalculateMacroTargetsInput)
def calculate_macro_targets(weight_kg: float, height_cm: float, age: int, sex: str,
                             activity_level: str, goal: str,
                             config: RunnableConfig) -> dict[str, Any]:
    """Calculate TDEE and macro targets using the Mifflin-St Jeor formula."""
    bmr = (10 * weight_kg + 6.25 * height_cm - 5 * age + (5 if sex == "male" else -161))
    tdee = bmr * _ACTIVITY.get(activity_level, 1.55)
    if goal == "weight_loss":
        cals, pr, fr, cr = tdee - 500, 0.30, 0.30, 0.40
    elif goal == "muscle_gain":
        cals, pr, fr, cr = tdee + 300, 0.30, 0.25, 0.45
    else:
        cals, pr, fr, cr = tdee, 0.25, 0.30, 0.45
    return {
        "tdee_kcal": round(tdee), "target_calories": round(cals),
        "protein_g": math.ceil(cals * pr / 4), "fat_g": math.ceil(cals * fr / 9),
        "carbs_g": math.ceil(cals * cr / 4), "refresh": "macro-targets",
    }


from src.tools.profile import PROFILE_TOOLS

FITNESS_TOOLS = [log_workout, log_body_metrics, update_fitness_goals, calculate_macro_targets, *PROFILE_TOOLS]
