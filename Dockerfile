# Green Wave++ -- single image: backend + built dashboard.
#
#   docker compose up --build          ->  http://localhost:8000/#key=<your key>
#
# Stage 1 builds the React dashboard; stage 2 is a CPU-only Python image
# that serves both the API and the static bundle from port 8000. CPU torch
# is installed first so the requirements.txt 'torch>=...' line is already
# satisfied and pip never pulls the multi-GB CUDA wheels.

FROM node:20-alpine AS ui
WORKDIR /ui
COPY ui/package.json ui/package-lock.json ./
RUN npm ci
COPY ui/ ./
RUN npm run build


FROM python:3.12-slim
WORKDIR /app

RUN pip install --no-cache-dir torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
COPY --from=ui /ui/dist ui/dist

EXPOSE 8000
# Demo mode: synthetic sensors, mock signals -- no GPU, mic, camera or SUMO
# needed inside the container. Swap the command for --virtual when you mount
# media + models.
CMD ["python", "run.py", "--demo", "--no-ui"]
