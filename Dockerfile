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

ENV PYTHONPATH=${PYTHONPATH}:/app/src 

# CMD ["python", "src/predict.py"]
CMD ["bash", "-c", "python src/train_pipeline.py && python src/predict.py  && tail -f /dev/nul"]
