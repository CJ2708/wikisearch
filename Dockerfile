# Portable container image — works on Hugging Face Spaces, Fly.io, Railway, etc.
# (Render can use this too, but the simpler render.yaml is the recommended path.)
FROM python:3.11-slim

WORKDIR /app

# Install dependencies first so Docker can cache this layer.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code.
COPY . .

# Hosts provide the port via $PORT; default to 8666 for local `docker run`.
ENV PORT=8666
EXPOSE 8666

# Use the cloud LLM by default in containers (set GROQ_API_KEY at runtime).
ENV LLM_PROVIDER=groq

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8666}"]
