import json
from pathlib import Path

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, MessageType, SendResult
from gateway.session import SessionSource
from gateway.session_context import clear_session_vars, get_session_env, set_session_vars


class _StubAdapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="test"), Platform.MATRIX)
        self.sent_voice_path = None
        self.sent_text = None

    async def connect(self):
        return True

    async def disconnect(self):
        pass

    async def send(self, chat_id, text=None, reply_to=None, metadata=None, content=None):
        self.sent_text = content if content is not None else text
        return SendResult(success=True, message_id="$text")

    async def send_voice(self, chat_id, audio_path, caption=None, reply_to=None, metadata=None):
        self.sent_voice_path = audio_path
        return SendResult(success=True, message_id="$voice")

    async def get_chat_info(self, chat_id):
        return {}

    async def _keep_typing(self, chat_id, stop_event=None, metadata=None):
        if stop_event is not None:
            await stop_event.wait()

    def _should_auto_tts_for_chat(self, chat_id):
        return True


@pytest.mark.asyncio
async def test_voice_auto_tts_restores_platform_context_after_runner_clears_it(monkeypatch, tmp_path):
    adapter = _StubAdapter()

    async def handler(event):
        return "Matrix voice reply"

    adapter.set_message_handler(handler)

    source = SessionSource(
        platform=Platform.MATRIX,
        chat_id="!room:example.org",
        user_id="@daniele:example.org",
        user_name="daniele",
    )
    event = MessageEvent(
        text="voice-message-recording.ogg",
        message_type=MessageType.VOICE,
        source=source,
        message_id="$incoming",
    )

    tokens = set_session_vars(platform="matrix", chat_id=source.chat_id)
    clear_session_vars(tokens)
    assert get_session_env("HERMES_SESSION_PLATFORM") == ""

    audio_path = tmp_path / "reply.ogg"
    audio_path.write_bytes(b"ogg")
    seen = {}

    def fake_tts_tool(*, text, output_path=None):
        seen["platform"] = get_session_env("HERMES_SESSION_PLATFORM")
        seen["chat_id"] = get_session_env("HERMES_SESSION_CHAT_ID")
        seen["message_id"] = get_session_env("HERMES_SESSION_MESSAGE_ID")
        return json.dumps({"file_path": str(audio_path)})

    monkeypatch.setattr("tools.tts_tool.check_tts_requirements", lambda: True)
    monkeypatch.setattr("tools.tts_tool.text_to_speech_tool", fake_tts_tool)

    await adapter._process_message_background(event, "agent:main:matrix:dm:!room:example.org")

    assert seen == {
        "platform": "matrix",
        "chat_id": "!room:example.org",
        "message_id": "$incoming",
    }
    assert adapter.sent_voice_path == str(audio_path)
    assert adapter.sent_text == "Matrix voice reply"
    assert get_session_env("HERMES_SESSION_PLATFORM") == ""