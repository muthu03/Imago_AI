# Use official Python image as a base
FROM python:3.10

# Set the working directory
WORKDIR /app

# Copy the application files
COPY requirements.txt requirements.txt
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port for FastAPI
EXPOSE 5000

# Run the FastAPI app
CMD ["python", "app.py"]
