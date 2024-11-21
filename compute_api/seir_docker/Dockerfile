# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app
# Set the environment variables to allow metadata server access
ENV GOOGLE_CLOUD_PROJECT="quiet-fusion-440218-i2"
ENV BETA=0.0
ENV EPSILON=0.0
ENV GCS_BUCKET="seir-output-bucket"
ENV DAYS=100
ENV SIMULATIONS=30


# Copy the script and requirements file into the container
COPY requirements.txt requirements.txt
# Install dependencies
RUN pip3 install -r requirements.txt


COPY main.py .


# Run the Python script
CMD ["python", "main.py"]
