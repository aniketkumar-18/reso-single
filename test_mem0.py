"""Memory system test — run with server stopped.

Usage:
    .venv/bin/python test_mem0.py
"""

import asyncio
import sys

TEST_USER = "f3ba959b-62ac-4dd6-9417-845cfe61a15a"  # must exist in auth.users


async def main() -> None:
    from src.infra.mem0_client import add_memory, search_memories

    print("=== Adding memories ===")
    facts = [
        ("User prefers eating dinner before 7pm.", "nutrition"),
        ("User dislikes spicy food and is vegetarian.", "nutrition"),
        ("User works as a software engineer — sedentary desk job.", "general"),
        ("User prefers morning workouts before 9am.", "fitness"),
    ]
    for fact, domain in facts:
        result = await add_memory(fact, user_id=TEST_USER, domain=domain)
        print(f"  [{result['status']}] ({domain}) {fact}")

    print()
    print("=== Dedup test — inserting same fact again ===")
    result = await add_memory("User dislikes spicy food and is vegetarian.", user_id=TEST_USER, domain="nutrition")
    print(f"  [{result['status']}] expected: duplicate")

    print()
    print("=== Retrieving memories ===")
    memories = await search_memories("dinner and workout preferences", user_id=TEST_USER)
    print(f"  total retrieved: {len(memories)}")
    for m in memories:
        print(f"  • {m}")

    print()
    print("=== Verifying in Supabase ===")
    from src.infra import db
    client = await db.get_client()
    if client:
        result = await client.table("user_memories").select("fact, domain, created_at").eq("user_id", TEST_USER).execute()
        print(f"  rows in user_memories: {len(result.data or [])}")
        for row in (result.data or []):
            print(f"  • [{row.get('domain','?')}] {row['fact']}")
    else:
        print("  DB not configured")

    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
