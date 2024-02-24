# syntax=docker/dockerfile:1

# Use Python 3.10 slim as the base image
ARG PYTHON_VERSION=3.10
FROM python:${PYTHON_VERSION}-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory inside the container
WORKDIR /app

# Create a virtual environment
RUN python -m venv /venv
ENV PATH="/venv/bin:$PATH"

# Add a non-privileged user to run the application
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

# Change ownership and permissions of the /app directory
RUN chown -R appuser:appuser /app

# Install Python dependencies inside the virtual environment
COPY requirements.txt .
RUN /venv/bin/python -m pip install --no-cache-dir -r requirements.txt

# Switch to the non-privileged user to run the application
USER appuser

# Copy the source code into the container
COPY . .

# Expose the port that the application listens on
EXPOSE 8000

# Command to run the application
CMD ["/venv/bin/python", "-m", "uvicorn", "app:app", "--host=0.0.0.0", "--port=8000"]
