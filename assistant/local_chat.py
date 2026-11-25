"""Local Ollama chat with one explicitly scoped inverter recommendation tool."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from typing import Any


OLLAMA_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_CHAT_MODEL = "qwen2.5:3b"
MAX_TOOL_ROUNDS = 2

SYSTEM_PROMPT = """You are the local project assistant for a seven-level cascaded
H-bridge inverter research project. Answer using the supplied project context
and verified tool results. The CNN predicts a reference target waveform; it does
not predict phases, switching gates, or inverter-output THD. Do not invent
measurements, claim unmeasured improvement, or describe this software as
hardware control, real-time actuation, or protection.

When the user asks for recommended carrier phases, gate/switching states, or a
simulator-verified THD comparison, call run_simulator_verified_recommendation.
Use operating conditions from the current project context when the user has not
specified them. Ask a concise clarification if required values are unavailable.
Never write or invent gate states yourself: the tool returns phases and metrics
from the existing optimizer and simulator. For a recommendation, report the
selected objective, its baseline and simulator-verified THD, selected phases,
and whether the baseline fallback was used. Do not reinterpret, relabel, or
infer extra metrics. Explain event Parseval total THD separately from sampled
H2-100 THD; they are not interchangeable. Dataset and checkpoint are read-only.
You may discuss CNN status and metrics only when present in the supplied
context. You have no tools for hardware, file writes, training, network access,
or protection actions."""

RECOMMENDATION_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "run_simulator_verified_recommendation",
        "description": (
            "Run the existing bounded Hybrid POA-PFO phase optimizer and inverter "
            "simulator. Returns recommended carrier phases, verified THD metrics, "
            "and safe simulator-generated gate states."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "modulation_index": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "reference_phase_degrees": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 360.0,
                },
                "dc_voltage_per_cell_v": {"type": "number", "exclusiveMinimum": 0.0},
                "load_resistance_ohm": {"type": "number", "exclusiveMinimum": 0.0},
                "load_inductance_h": {"type": "number", "exclusiveMinimum": 0.0},
                "objective": {
                    "type": "string",
                    "enum": ["voltage", "current"],
                },
            },
            "required": [],
            "additionalProperties": False,
        },
    },
}


class LocalChatError(RuntimeError):
    """Raised when the local Ollama service or its response is unavailable."""


class OllamaChatClient:
    """Small standard-library client restricted to the local Ollama service."""

    def __init__(self, *, timeout_seconds: float = 120.0) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive.")
        self.timeout_seconds = float(timeout_seconds)

    def _request(self, endpoint: str, payload: Mapping[str, Any] | None = None) -> dict[str, Any]:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{OLLAMA_BASE_URL}{endpoint}",
            data=data,
            headers={"Content-Type": "application/json"},
            method="GET" if data is None else "POST",
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=self.timeout_seconds,
            ) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise LocalChatError(
                f"Ollama returned HTTP {exc.code}: {detail or exc.reason}"
            ) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise LocalChatError(
                "Cannot reach local Ollama at 127.0.0.1:11434. "
                "Start Ollama and install a chat model first."
            ) from exc
        try:
            result = json.loads(body)
        except json.JSONDecodeError as exc:
            raise LocalChatError("Ollama returned invalid JSON.") from exc
        if not isinstance(result, dict):
            raise LocalChatError("Ollama returned an unexpected response.")
        return result

    def list_models(self) -> tuple[str, ...]:
        response = self._request("/api/tags")
        models = response.get("models")
        if not isinstance(models, list):
            raise LocalChatError("Ollama model-list response is missing its models array.")
        names = tuple(
            str(model["name"])
            for model in models
            if isinstance(model, dict) and isinstance(model.get("name"), str)
        )
        return names

    def chat(
        self,
        *,
        model: str,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        selected_model = model.strip()
        if not selected_model:
            raise ValueError("A local chat model name is required.")
        response = self._request(
            "/api/chat",
            {
                "model": selected_model,
                "messages": list(messages),
                "tools": list(tools),
                "stream": False,
            },
        )
        message = response.get("message")
        if not isinstance(message, dict):
            raise LocalChatError("Ollama chat response is missing its message.")
        return message


def _tool_arguments(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise LocalChatError("The model returned invalid tool arguments.") from exc
    if not isinstance(value, dict):
        raise LocalChatError("The model returned tool arguments in an invalid format.")
    return value


def chat_with_tools(
    client: OllamaChatClient,
    *,
    model: str,
    conversation: Sequence[Mapping[str, Any]],
    project_context: Mapping[str, Any],
    tool_executor: Callable[[str, dict[str, Any]], Mapping[str, Any]],
    max_tool_rounds: int = MAX_TOOL_ROUNDS,
) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
    """Answer one turn, allowing bounded calls to the approved simulator tool."""
    if max_tool_rounds < 0:
        raise ValueError("max_tool_rounds cannot be negative.")
    context_json = json.dumps(project_context, ensure_ascii=True, default=str)
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT + "\n\nCurrent project context:\n" + context_json,
        },
        *[dict(message) for message in conversation],
    ]
    tool_results: list[dict[str, Any]] = []
    tool_rounds = 0

    while True:
        message = client.chat(
            model=model,
            messages=messages,
            tools=(RECOMMENDATION_TOOL,),
        )
        tool_calls = message.get("tool_calls", [])
        if not isinstance(tool_calls, list):
            raise LocalChatError("The model returned tool calls in an invalid format.")
        messages.append(dict(message))
        if not tool_calls:
            content = message.get("content")
            if not isinstance(content, str) or not content.strip():
                raise LocalChatError("The local model returned an empty answer.")
            return content.strip(), messages[1:], tool_results
        if tool_rounds >= max_tool_rounds:
            raise LocalChatError(
                "The local model exceeded the allowed simulator-tool call limit."
            )
        tool_rounds += 1

        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                raise LocalChatError("The model returned a malformed tool call.")
            function = tool_call.get("function")
            if not isinstance(function, dict):
                raise LocalChatError("The model returned a malformed tool function.")
            name = function.get("name")
            if name != "run_simulator_verified_recommendation":
                raise LocalChatError(f"Unsupported assistant tool: {name!r}.")
            arguments = _tool_arguments(function.get("arguments"))
            result = dict(tool_executor(name, arguments))
            tool_results.append(result)
            messages.append(
                {
                    "role": "tool",
                    "tool_name": name,
                    "content": json.dumps(result, ensure_ascii=True, default=str),
                }
            )


__all__ = [
    "DEFAULT_CHAT_MODEL",
    "LocalChatError",
    "OLLAMA_BASE_URL",
    "OllamaChatClient",
    "RECOMMENDATION_TOOL",
    "chat_with_tools",
]
