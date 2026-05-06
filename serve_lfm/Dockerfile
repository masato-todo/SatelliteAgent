# Reuse the existing LFM2.5-VL TRL fine-tune image (aarch64-compatible,
# already has torch + transformers + peft + accelerate + torchvision).
# We only need to add the OpenAI-compat server framework.
FROM lfm25-vl-trl-ft:latest

ENV PYTHONUNBUFFERED=1 \
    HF_HUB_DISABLE_TELEMETRY=1

WORKDIR /app

RUN pip install --no-cache-dir fastapi 'uvicorn[standard]'

COPY server.py /app/server.py

EXPOSE 8000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
