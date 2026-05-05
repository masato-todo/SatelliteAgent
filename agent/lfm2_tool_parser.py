"""Custom vLLM tool_call parser for LFM2.5 / LFM2.5-VL.

LFM2.5 emits Pythonic function calls inside custom delimiter tokens:
    <|tool_call_start|>[fn(arg1="v", arg2=42)]<|tool_call_end|>

vLLM 0.19+ ships parsers for hermes / pythonic / llama4_pythonic / etc but
none of them handle this Liquid-specific format. This file is a plugin to
register a custom parser; load it via:

    vllm serve <model>                                                \\
        --enable-auto-tool-choice                                     \\
        --tool-parser-plugin /abs/path/to/agent/lfm2_tool_parser.py   \\
        --tool-call-parser lfm2_pythonic

Or import via sitecustomize.py + ToolParserManager.import_tool_parser(path)
when running prime-rl orchestrator (Kaggle GRPO).

Verified against vLLM 0.19.0 in S60c.
"""
from __future__ import annotations

import re
import ast
import json

from vllm.tool_parsers import ToolParser, ToolParserManager
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.engine.protocol import (
    ExtractedToolCallInformation,
    ToolCall,
    FunctionCall,
)


LFM2_RE = re.compile(
    r"<\|tool_call_start\|>\s*(\[.*?\])\s*<\|tool_call_end\|>",
    re.DOTALL,
)


def _ast_value(node):
    """Best-effort literal eval; fall back to source text if not literal."""
    try:
        return ast.literal_eval(node)
    except Exception:
        return ast.unparse(node)


@ToolParserManager.register_module("lfm2_pythonic")
class LFM2PythonicToolParser(ToolParser):
    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        m = LFM2_RE.search(model_output)
        if m is None:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output,
            )
        try:
            tree = ast.parse(m.group(1), mode="eval")
            list_node = tree.body
            if not isinstance(list_node, ast.List):
                raise ValueError("top-level expression is not a list of calls")
            calls = []
            for call in list_node.elts:
                if not isinstance(call, ast.Call):
                    continue
                fname = (
                    call.func.id
                    if isinstance(call.func, ast.Name)
                    else ast.unparse(call.func)
                )
                kwargs = {kw.arg: _ast_value(kw.value) for kw in call.keywords}
                # Positional args (unusual but support it)
                for i, a in enumerate(call.args):
                    kwargs.setdefault(f"arg{i}", _ast_value(a))
                calls.append(ToolCall(
                    function=FunctionCall(name=fname, arguments=json.dumps(kwargs)),
                ))
        except Exception:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output,
            )

        leading = model_output[: m.start()].strip()
        return ExtractedToolCallInformation(
            tools_called=bool(calls),
            tool_calls=calls,
            content=leading or None,
        )

    def extract_tool_calls_streaming(self, *args, **kwargs):
        # Streaming not implemented; fall back to non-streaming.
        return None
