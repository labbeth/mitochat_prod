FROM python:3.11-slim

WORKDIR /app

# ------------------------------------------------------------------------------------
# Install system dependencies
# ------------------------------------------------------------------------------------
RUN apt-get update && apt-get install -y \
    curl nano \
    && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------------------------------
# Copy UI code
# ------------------------------------------------------------------------------------
COPY scripts/ ./scripts/
COPY assets/ ./assets/
COPY requirements.frontend.txt .

# ------------------------------------------------------------------------------------
# Install Python deps
# ------------------------------------------------------------------------------------
RUN pip install --upgrade pip
RUN pip install -r requirements.frontend.txt

EXPOSE 8501

CMD ["streamlit", "run", "scripts/streamlit_app_frontend.py", "--server.address=0.0.0.0"]
