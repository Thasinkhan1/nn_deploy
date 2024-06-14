FROM python:3.10.14-bookworm

RUN pip install --upgrade pip

COPY  src /app/src

WORKDIR /app


RUN chmod -R 777 /app/src

RUN pip install -r /app/src/requirements.txt

RUN python -m src.train_pipeline

EXPOSE 5000

CMD ["uvicorn", "src.predict:app", "--host", "0.0.0.0", "--port", "5000"]

# ENV PYTHONPATH=${PYTHONPATH}:/app/src

# RUN tail -f /var/log/error.log 

# #ENTRYPOINT ["python3"]
# #
# #CMD ["./src/train_pipeline.py"]
