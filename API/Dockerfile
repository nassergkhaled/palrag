# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.10
FROM python:${PYTHON_VERSION}-slim as base

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Create a virtual environment
RUN python -m venv /venv
ENV PATH="/venv/bin:$PATH"

ARG UID=10001
RUN mkdir -p /home/appuser && \
    adduser \
    --disabled-password \
    --gecos "" \
    --home "/home/appuser" \
    --shell "/sbin/nologin" \
    --no-create-home \
    --uid "${UID}" \
    appuser

# Change ownership of the /app directory
RUN chown -R appuser:appuser /app

# Copy and install Python dependencies
COPY API/requirements.txt .
RUN /venv/bin/python -m pip install --no-cache-dir -r requirements.txt

USER appuser

# Copy the rest of the application code
COPY . .

EXPOSE 8000

CMD ["/venv/bin/python", "-m", "uvicorn", "app:app", "--host=0.0.0.0", "--port=8000"]
