from __future__ import annotations

import unittest
from unittest.mock import patch
import urllib.error

from assistant.local_chat import (
    LocalChatError,
    OllamaChatClient,
    chat_with_tools,
)


class _FakeClient:
    def __init__(self, messages: list[dict[str, object]]) -> None:
        self.messages = iter(messages)
        self.requests: list[dict[str, object]] = []

    def chat(self, **request: object) -> dict[str, object]:
        self.requests.append(request)
        return next(self.messages)


class LocalChatTests(unittest.TestCase):
    def test_chat_uses_project_context_and_returns_answer(self) -> None:
        client = _FakeClient([{"role": "assistant", "content": "Grounded answer."}])
        answer, conversation, tool_results = chat_with_tools(
            client,  # type: ignore[arg-type]
            model="qwen2.5:3b",
            conversation=[{"role": "user", "content": "What does the CNN do?"}],
            project_context={"cnn": {"model_loaded": True}},
            tool_executor=lambda _name, _arguments: {},
        )

        self.assertEqual(answer, "Grounded answer.")
        self.assertEqual(conversation[-1]["content"], "Grounded answer.")
        self.assertEqual(tool_results, [])
        self.assertIn("model_loaded", client.requests[0]["messages"][0]["content"])

    def test_recommendation_tool_result_is_returned_to_model(self) -> None:
        first_response = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "function": {
                        "name": "run_simulator_verified_recommendation",
                        "arguments": {"modulation_index": 0.75},
                    }
                }
            ],
        }
        client = _FakeClient(
            [
                first_response,
                {"role": "assistant", "content": "The simulator verified the result."},
            ]
        )
        tool_invocations: list[tuple[str, dict[str, object]]] = []

        def execute(name: str, arguments: dict[str, object]) -> dict[str, object]:
            tool_invocations.append((name, arguments))
            return {"verified_thd_percent": 12.5}

        answer, conversation, tool_results = chat_with_tools(
            client,  # type: ignore[arg-type]
            model="qwen2.5:3b",
            conversation=[{"role": "user", "content": "Recommend phases"}],
            project_context={},
            tool_executor=execute,
        )

        self.assertEqual(answer, "The simulator verified the result.")
        self.assertEqual(
            tool_invocations,
            [
                (
                    "run_simulator_verified_recommendation",
                    {"modulation_index": 0.75},
                )
            ],
        )
        self.assertEqual(tool_results, [{"verified_thd_percent": 12.5}])
        self.assertEqual(conversation[-1]["content"], answer)
        self.assertEqual(len(client.requests), 2)

    def test_unsupported_tool_is_rejected(self) -> None:
        client = _FakeClient(
            [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {"function": {"name": "write_file", "arguments": {}}}
                    ],
                }
            ]
        )

        with self.assertRaisesRegex(LocalChatError, "Unsupported assistant tool"):
            chat_with_tools(
                client,  # type: ignore[arg-type]
                model="qwen2.5:3b",
                conversation=[],
                project_context={},
                tool_executor=lambda _name, _arguments: {},
            )

    def test_tool_call_rounds_are_bounded(self) -> None:
        tool_call = {
            "role": "assistant",
            "tool_calls": [
                {
                    "function": {
                        "name": "run_simulator_verified_recommendation",
                        "arguments": {},
                    }
                }
            ],
        }
        client = _FakeClient([tool_call, tool_call])

        with self.assertRaisesRegex(LocalChatError, "tool call limit"):
            chat_with_tools(
                client,  # type: ignore[arg-type]
                model="qwen2.5:3b",
                conversation=[],
                project_context={},
                tool_executor=lambda _name, _arguments: {},
                max_tool_rounds=1,
            )

    def test_empty_model_name_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "model name is required"):
            OllamaChatClient().chat(model=" ", messages=[], tools=[])

    def test_unavailable_local_service_has_actionable_error(self) -> None:
        with patch(
            "assistant.local_chat.urllib.request.urlopen",
            side_effect=urllib.error.URLError("connection refused"),
        ):
            with self.assertRaisesRegex(LocalChatError, "Start Ollama"):
                OllamaChatClient(timeout_seconds=0.1).list_models()


if __name__ == "__main__":
    unittest.main()
