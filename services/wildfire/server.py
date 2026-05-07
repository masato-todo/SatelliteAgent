"""OpenAI-compatible chat-completions server for
LiquidAI/LFM2.5-VL-450M + YujiYamaguchi/lfm2-5-vl-450m-wildfire (LoRA).

Why custom: vLLM doesn't support Lfm2Vl LoRA reliably; llama.cpp can't
convert this LoRA. transformers + peft applies the adapter at runtime.

Endpoints:
  GET  /v1/models               -> {"data":[{"id": MODEL_NAME}, ...]}
  POST /v1/chat/completions     -> OpenAI 1:1 reply
  GET  /health                  -> 200

The chat endpoint expects messages with content blocks:
  [{type:"text", text:"..."},
   {type:"image_url", image_url:{url:"data:image/png;base64,..."}}]
which is exactly what tools/classifier_openai.py sends.
"""
from __future__ import annotations

import base64
import io
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

import torch
from fastapi import FastAPI, HTTPException
from PIL import Image
from peft import PeftModel
from pydantic import BaseModel
from transformers import AutoModelForImageTextToText, AutoProcessor


BASE_DIR    = os.environ.get("LFM_BASE_DIR",    "/models/base")
ADAPTER_DIR = os.environ.get("LFM_ADAPTER_DIR", "/models/adapter")
MODEL_NAME  = os.environ.get("LFM_MODEL_NAME",  "lfm2.5-vl-450m-wildfire")
DEVICE      = os.environ.get("LFM_DEVICE",      "cuda" if torch.cuda.is_available() else "cpu")
DTYPE       = torch.float16 if DEVICE == "cuda" else torch.float32

STATE: dict[str, Any] = {}


def _load() -> None:
    # Prefer the processor packaged with the adapter — it carries the
    # exact image_processor_type the LoRA was fine-tuned with.
    # The base model's processor uses Lfm2VlImageProcessorFast (cosmetic
    # speedup), but our LoRAs were trained with the slow
    # Lfm2VlImageProcessor; mixing them silently shifts image
    # preprocessing and the model output collapses (all "HIGH" / similar).
    proc_dir = ADAPTER_DIR if os.path.exists(os.path.join(ADAPTER_DIR, "processor_config.json")) else BASE_DIR
    print(f"[lfm-serve] loading processor from {proc_dir}", flush=True)
    processor = AutoProcessor.from_pretrained(proc_dir, trust_remote_code=True)
    print(f"[lfm-serve]   image_processor: {type(processor.image_processor).__name__}", flush=True)
    print(f"[lfm-serve] loading base model on {DEVICE} ({DTYPE})", flush=True)
    base = AutoModelForImageTextToText.from_pretrained(
        BASE_DIR, trust_remote_code=True, dtype=DTYPE,
    ).to(DEVICE)
    print(f"[lfm-serve] applying LoRA adapter from {ADAPTER_DIR}", flush=True)
    model = PeftModel.from_pretrained(base, ADAPTER_DIR, is_trainable=False)
    model.eval()
    STATE["processor"] = processor
    STATE["model"] = model
    print(f"[lfm-serve] ready: {MODEL_NAME}", flush=True)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    _load()
    yield


app = FastAPI(lifespan=lifespan)


# -- Pydantic models (loose) --
class ChatMessage(BaseModel):
    role: str
    content: Any  # str OR list[{type, text/image_url}]


class ChatReq(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    max_tokens: int = 512
    temperature: float = 0.1
    top_p: float | None = None
    response_format: dict | None = None
    stream: bool = False


# -- Helpers --
def _decode_data_url(url: str) -> Image.Image:
    if not url.startswith("data:"):
        raise HTTPException(400, f"only data: URLs supported, got {url[:30]}")
    head, b64 = url.split(",", 1)
    raw = base64.b64decode(b64)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def _to_processor_messages(messages: list[ChatMessage]) -> list[dict]:
    """Convert OpenAI messages (image_url data URLs) to processor format
    (PIL.Image objects under {type: image, image: ...}).
    """
    out: list[dict] = []
    for m in messages:
        content = m.content
        if isinstance(content, str):
            out.append({"role": m.role, "content": [{"type": "text", "text": content}]})
            continue
        new_blocks: list[dict] = []
        for part in content:
            t = part.get("type")
            if t == "text":
                new_blocks.append({"type": "text", "text": part.get("text", "")})
            elif t == "image_url":
                img = _decode_data_url(part["image_url"]["url"])
                new_blocks.append({"type": "image", "image": img})
            elif t == "image":
                # Pass-through for unusual clients
                new_blocks.append(part)
            else:
                # Unknown block: drop silently
                continue
        out.append({"role": m.role, "content": new_blocks})
    return out


@torch.inference_mode()
def _generate(processor, model, messages: list[dict],
              max_tokens: int, temperature: float, top_p: float | None) -> str:
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v
              for k, v in inputs.items()}

    gen_kwargs = {
        "max_new_tokens": max_tokens,
        "do_sample": temperature > 0.0,
    }
    if temperature > 0.0:
        gen_kwargs["temperature"] = temperature
    if top_p is not None:
        gen_kwargs["top_p"] = top_p

    out = model.generate(**inputs, **gen_kwargs)
    # Strip the prompt prefix
    in_len = inputs["input_ids"].shape[-1]
    new_tokens = out[0, in_len:]
    text = processor.decode(new_tokens, skip_special_tokens=True)
    return text.strip()


# -- Endpoints --
@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME, "device": DEVICE}


@app.get("/v1/models")
def models():
    return {
        "object": "list",
        "data": [
            {"id": MODEL_NAME, "object": "model", "owned_by": "local",
             "created": int(time.time())}
        ],
    }


@app.post("/v1/chat/completions")
def chat_completions(req: ChatReq):
    if "processor" not in STATE:
        raise HTTPException(503, "model still loading")
    proc_msgs = _to_processor_messages(req.messages)
    try:
        text = _generate(
            STATE["processor"], STATE["model"], proc_msgs,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
        )
    except Exception as e:
        raise HTTPException(500, f"generate failed: {type(e).__name__}: {e}")

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model or MODEL_NAME,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }
