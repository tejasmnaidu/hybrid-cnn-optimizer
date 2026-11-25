"""Local Ollama chat with one explicitly scoped inverter recommendation tool."""

from __future__ import annotations

import json
import math
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

Write clear, complete sentences, not bare numbers. For a metric question,
include the value, unit, metric name, and harmonic range or measurement method
when those details are available in context. If the requested fact is absent
from project context and tool results, say exactly that it is not available in
the provided project data; do not fill the gap with a guess or generic advice.
Answer the latest user request only; do not append unrelated facts from earlier
turns.

When the user asks for recommended carrier phases, gate/switching states, or a
simulator-verified THD comparison, call run_simulator_verified_recommendation.
Use operating conditions from the current project context when the user has not
specified them. Ask a concise clarification if required values are unavailable.
Never write or invent gate states yourself: the tool returns phases and metrics
from the existing optimizer and simulator. For a recommendation, report the
selected objective, its baseline and simulator-verified THD, selected phases,
and whether the baseline fallback was used. The application renders the actual
simulator-generated gate table separately; do not describe array dimensions or
metadata as gate values. Do not reinterpret, relabel, or infer extra metrics.
Explain event Parseval total THD separately from sampled H2-100 THD; they are
not interchangeable. Dataset and checkpoint are read-only.
You may discuss CNN status and metrics only when present in the supplied
context. You have no tools for hardware, file writes, training, network access,
or protection actions."""

RECOMMENDATION_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "run_simulator_verified_recommendation",
        "description": (
            "Run the existing bounded Hybrid POA-PFO phase optimizer and inverter "
            "simulator. Returns selected carrier phases, verified THD metrics, "
            "and baseline-fallback status."
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


def _recommendation_answer(result: Mapping[str, Any]) -> str:
    objective = result.get("objective")
    if objective == "voltage":
        metric_label = "event-based Parseval total voltage THD (orders 2–infinity)"
    elif objective == "current":
        metric_label = "event-based load-current THD (orders 2–100)"
    else:
        raise LocalChatError("The simulator recommendation returned an unknown objective.")

    phases = result.get("carrier_phase_offsets_degrees")
    if (
        not isinstance(phases, (list, tuple))
        or len(phases) != 3
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0.0 <= float(value) < 360.0
            for value in phases
        )
    ):
        raise LocalChatError(
            "The simulator recommendation returned invalid carrier phase offsets."
        )
    baseline = result.get("baseline_objective_thd_percent")
    verified = result.get("simulator_verified_objective_thd_percent")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
        for value in (baseline, verified)
    ):
        raise LocalChatError(
            "The simulator recommendation returned invalid THD measurements."
        )
    fallback = result.get("fallback_to_baseline")
    if not isinstance(fallback, bool):
        raise LocalChatError(
            "The simulator recommendation returned invalid fallback status."
        )

    formatted_phases = ", ".join(f"{float(value):.2f}°" for value in phases)
    fallback_note = (
        "No feasible improvement was found, so the baseline phases were retained."
        if fallback
        else "The optimizer selected a candidate rather than the baseline fallback."
    )
    return (
        f"The simulator-recommended carrier phase offsets are "
        f"[{formatted_phases}]. The baseline {metric_label} is "
        f"{float(baseline):.6f}%; the independently simulator-verified "
        f"recommended value is {float(verified):.6f}%. {fallback_note} "
        "The application displays the simulator-generated gate table separately."
    )


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
        if len(tool_calls) != 1:
            raise LocalChatError(
                "Only one simulator recommendation may be requested per chat turn."
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
        answer = _recommendation_answer(tool_results[-1])
        messages.append({"role": "assistant", "content": answer})
        return answer, messages[1:], tool_results


__all__ = [
    "DEFAULT_CHAT_MODEL",
    "LocalChatError",
    "OLLAMA_BASE_URL",
    "OllamaChatClient",
    "RECOMMENDATION_TOOL",
    "chat_with_tools",
]
