FROM python:3.9-slim

ADD data ./data
ADD client.py .


ADD dataset_utils.py .
ADD model_definition.py .

RUN pip install numpy pandas tensorflow tensorflow-cpu flwr sklearn

CMD ["python", "-u", "client.py"]