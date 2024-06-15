FROM python:3.8.19-bookworm

# Upgrade pip
RUN pip install --upgrade pip

# Set the working directory
WORKDIR /app

# Copy the project files
COPY . /app

# Ensure permissions
RUN chmod -R 777 /app/src

# Install required Python packages
RUN pip install -r /app/src/requirements.txt

# Run the training script
RUN python -m src.train_pipeline

# Expose port 5000 for the application
EXPOSE 5000

# Set environment variables for Flask
ENV PYTHONPATH=${PYTHONPATH}:/app/src
ENV FLASK_APP=src.predict
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=5000

# Command to run the Flask application
CMD ["flask", "run"]
