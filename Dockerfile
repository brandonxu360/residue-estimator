# Use official TensorFlow image as base
FROM tensorflow/tensorflow

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code into the container
COPY src/ src/

# Set DATA_DIR environment variable to data directory in container
ENV DATA_DIR=/app/data
