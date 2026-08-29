"""Local conversation persistence tests."""

from __future__ import annotations

from isaac.interfaces.conversation_store import ConversationStore


def test_conversation_store_persists_messages_and_title(tmp_path) -> None:
    path = tmp_path / "conversations.sqlite3"
    store = ConversationStore(path)
    conversation_id = store.create()
    store.add(conversation_id, "user", "Organize os documentos da semana com segurança")
    store.add(conversation_id, "assistant", "Vou primeiro analisar os arquivos.")

    reopened = ConversationStore(path)

    assert reopened.load(conversation_id) == [
        {"role": "user", "content": "Organize os documentos da semana com segurança"},
        {"role": "assistant", "content": "Vou primeiro analisar os arquivos."},
    ]
    assert reopened.list()[0]["title"].startswith("Organize os documentos")


def test_ensure_latest_reuses_most_recent_conversation(tmp_path) -> None:
    store = ConversationStore(tmp_path / "conversations.sqlite3")
    conversation_id = store.create("Teste")

    assert store.ensure_latest() == conversation_id
