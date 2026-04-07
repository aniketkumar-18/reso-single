-- ============================================================
-- Zone 1 — Core Supabase schema
-- NOTE: conversations table already exists with account_id column — skipped here
-- ============================================================

-- ── User profiles ─────────────────────────────────────────────────────────────
create table if not exists user_profiles (
    user_id         uuid primary key references auth.users(id) on delete cascade,
    date_of_birth   date,
    sex             text check (sex in ('male', 'female', 'other')),
    height_cm       numeric(5,1),
    weight_kg       numeric(5,1),
    conditions      text[],
    medications     text[],
    allergies       text[],
    goals           text[],
    created_at      timestamptz default now(),
    updated_at      timestamptz default now()
);

alter table user_profiles enable row level security;

drop policy if exists "users_see_own_profile" on user_profiles;
create policy "users_see_own_profile"
    on user_profiles for all
    using (auth.uid() = user_id)
    with check (auth.uid() = user_id);

-- ── Messages ──────────────────────────────────────────────────────────────────
create table if not exists messages (
    id                  uuid primary key default gen_random_uuid(),
    conversation_id     uuid not null references conversations(id) on delete cascade,
    role                text not null check (role in ('user', 'assistant', 'system', 'tool')),
    content             text not null,
    metadata            jsonb,
    created_at          timestamptz default now()
);

alter table messages enable row level security;

drop policy if exists "users_see_own_messages" on messages;
create policy "users_see_own_messages"
    on messages for all
    using (
        exists (
            select 1 from conversations c
            where c.id = messages.conversation_id
              and c.account_id = auth.uid()
        )
    );

create index if not exists idx_messages_conversation_id_created
    on messages(conversation_id, created_at desc);

-- ── User memories ─────────────────────────────────────────────────────────────
create table if not exists user_memories (
    id          uuid primary key default gen_random_uuid(),
    user_id     uuid not null references auth.users(id) on delete cascade,
    fact        text not null,
    created_at  timestamptz default now()
);

alter table user_memories enable row level security;

drop policy if exists "users_see_own_memories" on user_memories;
create policy "users_see_own_memories"
    on user_memories for all
    using (auth.uid() = user_id)
    with check (auth.uid() = user_id);

create index if not exists idx_user_memories_user_id on user_memories(user_id);
