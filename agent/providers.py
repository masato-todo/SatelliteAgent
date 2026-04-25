"""LLM provider abstraction.

Phase 1: Gemini 2.5 Flash (google-genai SDK) — drives ReAct while the
         LFM2-VL orchestrator adapter is still being trained. Cheap &
         supports function calling + vision.
Phase 3: Swap to LFM2-VL via a local provider (TBD).
"""
from __future__ import annotations

import os
from typing import Any, Protocol


class Provider(Protocol):
    def generate(
        self,
        contents: list[Any],
        tools: list[dict[str, Any]],
        system: str | None = None,
    ) -> Any: ...


class GeminiProvider:
    """Google Gemini wrapper for Phase 1 ReAct driving.

    Uses `google-genai` (the unified SDK).  Defaults to `gemini-2.5-flash`.
    """

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        max_output_tokens: int = 2048,
    ):
        from google import genai
        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY not set")
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.max_output_tokens = max_output_tokens

    def generate(
        self,
        contents: list[Any],
        tools: list[dict[str, Any]],
        system: str | None = None,
    ):
        from google.genai import types
        gemini_tools = [
            types.Tool(function_declarations=[
                {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema") or {"type": "object", "properties": {}},
                }
                for t in tools
            ])
        ] if tools else None

        cfg = types.GenerateContentConfig(
            tools=gemini_tools,
            system_instruction=system,
            max_output_tokens=self.max_output_tokens,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        return self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=cfg,
        )


class LFM2VLProvider:
    """Placeholder for Phase 3 local LFM2-VL orchestrator."""

    def __init__(self, base_path: str, orchestrator_adapter: str):
        raise NotImplementedError("Phase 3: integrate local LFM2-VL here")

    def generate(self, *args, **kwargs):
        raise NotImplementedError
