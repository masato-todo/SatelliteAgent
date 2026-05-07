# Reuse the existing LFM2.5-VL TRL fine-tune image (aarch64-compatible,
# already has torch + transformers + peft + accelerate + torchvision).
# We only need to add the OpenAI-compat server framework.
FROM lfm25-vl-trl-ft:latest

ENV PYTHONUNBUFFERED=1 \
    HF_HUB_DISABLE_TELEMETRY=1

WORKDIR /app

RUN pip install --no-cache-dir fastapi 'uvicorn[standard]'

# The base image ships transformers 5.5.x / peft 0.18.x — these match what
# the wildfire LoRA was fine-tuned with (recall=0.933 reproducible).
# The wildfire-precursor LoRA, however, needs transformers >=5.7 and
# peft >=0.19 to apply correctly (with the older versions it collapses
# to "always HIGH"). Set UPGRADE_LIBS=1 at build time when serving the
# precursor adapter to install the newer versions on top.
ARG UPGRADE_LIBS=0
RUN if [ "$UPGRADE_LIBS" = "1" ]; then \
        pip install --no-cache-dir --upgrade \
            "transformers>=5.7,<5.8" \
            "peft>=0.19,<0.20"; \
    fi

COPY server.py /app/server.py

EXPOSE 8000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
