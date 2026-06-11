"""Regression tests for auto-reset continuity handoffs."""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest

from gateway.config import GatewayConfig, Platform, SessionResetPolicy
from gateway.run import GatewayRunner
from gateway.platforms.base import MessageEvent
from gateway.session import SessionEntry, SessionSource, SessionStore, build_session_key
from hermes_state import SessionDB


@pytest.fixture()
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    db = SessionDB(db_path=tmp_path / "state.db")
    config = GatewayConfig(
        default_reset_policy=SessionResetPolicy(
            mode="idle",
            idle_minutes=1,
            notify=True,
        )
    )
    session_store = SessionStore(sessions_dir=tmp_path / "sessions", config=config)
    session_store._db = db
    return session_store


def _source() -> SessionSource:
    return SessionSource(
        platform=Platform.DISCORD,
        user_id="joel",
        chat_id="rocky-channel",
        chat_type="group",
        user_name="Joel",
        chat_name="Rocky",
    )


def _telegram_topic_source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        user_id="6667473435",
        chat_id="6667473435",
        chat_type="dm",
        user_name="Joel",
        chat_name="₿rill",
        thread_id="21389",
    )


def _seed_expired_session(
    store: SessionStore,
    sid: str = "old-session",
    source: SessionSource | None = None,
) -> tuple[SessionSource, str]:
    source = source or _source()
    session_key = build_session_key(
        source,
        group_sessions_per_user=store.config.group_sessions_per_user,
        thread_sessions_per_user=store.config.thread_sessions_per_user,
    )
    db = cast(SessionDB, store._db)
    db.create_session(sid, source=source.platform.value, user_id=source.user_id)
    db.set_session_title(sid, "Hermes reset continuity incident")
    db.append_message(sid, "user", "Implement the reset handoff and prove it with tests.")
    db.append_message(sid, "assistant", "I will patch the auto-reset path and verify it.")
    db.append_message(
        sid,
        "tool",
        '{"todos":[{"content":"Patch reset handoff","status":"in_progress"},{"content":"Run focused tests","status":"pending"}]}',
        tool_name="todo",
    )
    store._entries[session_key] = SessionEntry(
        session_key=session_key,
        session_id=sid,
        created_at=datetime.now() - timedelta(hours=2),
        updated_at=datetime.now() - timedelta(minutes=10),
        origin=source,
        display_name=source.chat_name,
        platform=source.platform,
        chat_type=source.chat_type,
        total_tokens=0,
    )
    store._loaded = True
    return source, session_key


def test_auto_reset_creates_child_session_with_handoff(store):
    source, _session_key = _seed_expired_session(store)

    entry = store.get_or_create_session(source)

    assert entry.session_id != "old-session"
    assert entry.was_auto_reset is True
    assert entry.auto_reset_reason == "idle"
    assert entry.reset_had_activity is True
    assert entry.parent_session_id == "old-session"
    assert entry.reset_handoff is not None
    assert "[SESSION RESET HANDOFF]" in entry.reset_handoff
    assert "Previous session id: old-session" in entry.reset_handoff
    assert "Previous title: Hermes reset continuity incident" in entry.reset_handoff
    assert "Implement the reset handoff and prove it with tests." in entry.reset_handoff
    assert "Patch reset handoff" in entry.reset_handoff

    db = cast(SessionDB, store._db)
    old_row = db.get_session("old-session")
    new_row = db.get_session(entry.session_id)
    assert old_row is not None
    assert new_row is not None
    assert old_row["end_reason"] == "session_reset"
    assert new_row["parent_session_id"] == "old-session"


def test_session_entry_serializes_reset_handoff(store):
    source, session_key = _seed_expired_session(store)
    entry = store.get_or_create_session(source)

    round_tripped = SessionEntry.from_dict(entry.to_dict())

    assert round_tripped.session_key == session_key
    assert round_tripped.parent_session_id == "old-session"
    assert round_tripped.reset_handoff == entry.reset_handoff


def test_gateway_prepends_handoff_not_fresh_context_note():
    session_entry = SimpleNamespace(
        was_auto_reset=True,
        auto_reset_reason="daily",
        parent_session_id="previous-session",
        reset_handoff=(
            "[SESSION RESET HANDOFF]\n"
            "Previous session id: previous-session\n"
            "Recent exact excerpts:\n"
            "- User: keep working on the reset handoff"
        ),
    )

    prompt = GatewayRunner._prepend_auto_reset_handoff("platform context", session_entry)

    assert prompt.startswith("[SESSION RESET HANDOFF]")
    assert "Previous session id: previous-session" in prompt
    assert "platform context" in prompt
    assert "fresh conversation with no prior context" not in prompt


def test_gateway_fallback_handoff_still_not_context_free():
    session_entry = SimpleNamespace(
        was_auto_reset=True,
        auto_reset_reason="daily",
        parent_session_id="previous-session",
        reset_handoff=None,
    )

    prompt = GatewayRunner._prepend_auto_reset_handoff("platform context", session_entry)

    assert "[SESSION RESET HANDOFF]" in prompt
    assert "Previous session id: previous-session" in prompt
    assert "not context-free" in prompt
    assert "fresh conversation with no prior context" not in prompt


@pytest.mark.asyncio
async def test_gateway_handler_triggers_reset_and_injects_handoff_into_agent_call(store):
    """Exercise the gateway path, not just the SessionStore/helper units."""
    source, session_key = _seed_expired_session(store)
    event = MessageEvent(text="keep going please", source=source, message_id="m-1")
    runner = object.__new__(GatewayRunner)
    runner_any = cast(Any, runner)
    runner_any.config = store.config
    runner_any.session_store = store
    runner_any._session_db = store._db
    runner_any.hooks = SimpleNamespace(emit=AsyncMock())
    adapter = SimpleNamespace(send=AsyncMock(), stop_typing=AsyncMock())
    runner_any.adapters = {Platform.DISCORD: adapter}
    runner_any._session_model_overrides = {}
    runner_any._session_reasoning_overrides = {}
    runner_any._pending_model_notes = {}
    runner_any._background_tasks = set()
    runner_any._show_reasoning = False

    runner_any._cache_session_source = lambda *_args, **_kwargs: None
    runner_any._is_telegram_topic_lane = lambda *_args, **_kwargs: False
    runner_any._set_session_reasoning_override = lambda *_args, **_kwargs: None
    runner_any._format_session_info = lambda: ""
    runner_any._thread_metadata_for_source = lambda *_args, **_kwargs: {}
    runner_any._reply_anchor_for_event = lambda event: event.message_id
    runner_any._set_session_env = lambda context: []
    runner_any._clear_session_env = lambda tokens: None
    runner_any._bind_adapter_run_generation = lambda *_args, **_kwargs: None
    runner_any._is_session_run_current = lambda *_args, **_kwargs: True
    runner_any._should_send_voice_reply = lambda *_args, **_kwargs: False
    runner_any._get_guild_id = lambda event: None
    runner_any._deliver_platform_notice = AsyncMock()

    async def _prepare_inbound_message_text(**kwargs: Any) -> str:
        return kwargs["event"].text

    captured: dict[str, Any] = {}

    async def _run_agent(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {
            "final_response": "ok",
            "messages": [],
            "api_calls": 0,
            "tools": [],
            "last_prompt_tokens": 0,
        }

    runner_any._prepare_inbound_message_text = _prepare_inbound_message_text
    runner_any._run_agent = _run_agent

    response = await runner._handle_message_with_agent(
        event,
        source,
        _quick_key=session_key,
        run_generation=1,
    )

    assert response == "ok"
    assert captured["message"] == "keep going please"
    assert captured["session_id"] != "old-session"
    assert captured["history"] == []
    context_prompt = captured["context_prompt"]
    assert context_prompt.startswith("[SESSION RESET HANDOFF]")
    assert "Previous session id: old-session" in context_prompt
    assert "Previous title: Hermes reset continuity incident" in context_prompt
    assert "Implement the reset handoff and prove it with tests." in context_prompt
    assert "fresh conversation with no prior context" not in context_prompt

    new_row = cast(SessionDB, store._db).get_session(captured["session_id"])
    assert new_row is not None
    assert new_row["parent_session_id"] == "old-session"
    adapter.send.assert_awaited()
    assert "continuity handoff was preserved" in adapter.send.await_args.args[1]


@pytest.mark.asyncio
async def test_telegram_topic_binding_rebinds_to_auto_reset_child(store):
    """A stale Telegram topic binding must not switch away from an auto-reset child."""
    source, session_key = _seed_expired_session(store, source=_telegram_topic_source())
    assert source.user_id is not None
    assert source.chat_id is not None
    assert source.thread_id is not None
    db = cast(SessionDB, store._db)
    db.enable_telegram_topic_mode(chat_id=source.chat_id, user_id=source.user_id)
    db.bind_telegram_topic(
        chat_id=source.chat_id,
        thread_id=source.thread_id,
        user_id=source.user_id,
        session_key=session_key,
        session_id="old-session",
    )

    event = MessageEvent(text="keep going please", source=source, message_id="m-telegram")
    runner = object.__new__(GatewayRunner)
    runner_any = cast(Any, runner)
    runner_any.config = store.config
    runner_any.session_store = store
    runner_any._session_db = db
    runner_any.hooks = SimpleNamespace(emit=AsyncMock())
    adapter = SimpleNamespace(send=AsyncMock(), stop_typing=AsyncMock())
    runner_any.adapters = {Platform.TELEGRAM: adapter}
    runner_any._session_model_overrides = {}
    runner_any._session_reasoning_overrides = {}
    runner_any._pending_model_notes = {}
    runner_any._background_tasks = set()
    runner_any._show_reasoning = False

    runner_any._cache_session_source = lambda *_args, **_kwargs: None
    runner_any._set_session_reasoning_override = lambda *_args, **_kwargs: None
    runner_any._format_session_info = lambda: ""
    runner_any._thread_metadata_for_source = lambda *_args, **_kwargs: {}
    runner_any._reply_anchor_for_event = lambda event: event.message_id
    runner_any._set_session_env = lambda context: []
    runner_any._clear_session_env = lambda tokens: None
    runner_any._bind_adapter_run_generation = lambda *_args, **_kwargs: None
    runner_any._is_session_run_current = lambda *_args, **_kwargs: True
    runner_any._should_send_voice_reply = lambda *_args, **_kwargs: False
    runner_any._deliver_platform_notice = AsyncMock()

    async def _prepare_inbound_message_text(**kwargs: Any) -> str:
        return kwargs["event"].text

    captured: dict[str, Any] = {}

    async def _run_agent(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {
            "final_response": "ok",
            "messages": [],
            "api_calls": 0,
            "tools": [],
            "last_prompt_tokens": 0,
        }

    runner_any._prepare_inbound_message_text = _prepare_inbound_message_text
    runner_any._run_agent = _run_agent

    response = await runner._handle_message_with_agent(
        event,
        source,
        _quick_key=session_key,
        run_generation=1,
    )

    assert response == "ok"
    assert captured["session_id"] != "old-session"
    context_prompt = captured["context_prompt"]
    assert context_prompt.startswith("[SESSION RESET HANDOFF]")
    assert "Previous session id: old-session" in context_prompt
    assert "fresh conversation with no prior context" not in context_prompt

    child = db.get_session(captured["session_id"])
    assert child is not None
    assert child["parent_session_id"] == "old-session"
    assert child["end_reason"] is None
    old = db.get_session("old-session")
    assert old is not None
    assert old["end_reason"] == "session_reset"

    binding = db.get_telegram_topic_binding(
        chat_id=source.chat_id,
        thread_id=source.thread_id,
    )
    assert binding is not None
    assert binding["session_id"] == captured["session_id"]
    entry = store._entries[session_key]
    assert entry.session_id == captured["session_id"]
