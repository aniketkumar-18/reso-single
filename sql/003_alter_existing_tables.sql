-- ============================================================
-- Additive migrations for existing tables (safe to re-run).
-- Already applied to Supabase — verified via PostgREST OpenAPI.
-- ============================================================

-- workouts: intensity + conversation_id (already applied)
-- body_metrics: waist_cm (already applied)
-- fitness_profiles: daily_step_goal + notes (already applied)
-- meal_items: description + conversation_id (already applied)
-- messages: metadata jsonb (already applied)

-- No further changes needed — all columns verified present.
