FROM python:3.10-slim

# System deps for OpenCV and scikit-image
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# PyTorch CPU wheel lives at a custom index — install before other deps
COPY requirements-docker.txt .
RUN pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        torch==2.2.2+cpu torchvision==0.17.2+cpu \
    && pip install --no-cache-dir -r requirements-docker.txt

# Application code and model checkpoint
COPY config.py app.py run_inference.py find_path.py run_batch_pathfinding.py ./
COPY src/ src/
COPY models/ models/

# Runtime output directories
RUN mkdir -p data/masks/predicted data/masks/cleaned data/masks/skeletons data/paths

EXPOSE 7860

ENV PYTHONUNBUFFERED=1

CMD ["python", "app.py"]
