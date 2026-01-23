FROM python:3.10-slim

# Set the working directory to /app for copying files
WORKDIR /app

# Copy requirements from the root of your project
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project (including backend folder) to /app
COPY . .

# Create a non-root user for security (Hugging Face requirement)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Switch working directory to /app/backend so imports (like 'from agents...') work
WORKDIR /app/backend

# Expose the port (Hugging Face expects port 7860)
EXPOSE 7860

# Start the application from the backend directory
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]