# Use a lightweight, stable Python base image to keep the container size small
FROM python:3.10-slim

# Prevent Python from writing .pyc files and force stdout to flush immediately (good for live logs)
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory inside the virtual container
WORKDIR /app

# Copy the requirements file first to leverage Docker layer caching
COPY requirements.txt /app/

# Install the required libraries (no-cache-dir keeps the image size down)
RUN pip install --no-cache-dir -r requirements.txt

# Copy all the rest of your project files (server.py, client.py) into the container
COPY . /app/

# Expose the ports for Flower Server (8080) and TensorBoard (6006)
EXPOSE 8080
EXPOSE 6006