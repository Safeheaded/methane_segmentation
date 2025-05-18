FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates

# Download the latest installer
ADD https://astral.sh/uv/install.sh /uv-installer.sh

# Run the installer then remove it
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Ensure the installed binary is on the `PATH`
ENV PATH="/root/.local/bin/:$PATH"

# Set workdir
WORKDIR /app

# Copy project files
COPY . .

# Create virtual environment and install dependencies
RUN uv sync --locked

RUN uv run install_torch.py

# Activate venv by default
ENV VIRTUAL_ENV="/app/.venv"
ENV PATH="/app/.venv/bin:$PATH"

# Default command
CMD ["bash"]
