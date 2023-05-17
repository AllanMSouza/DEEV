FROM python:3.9-slim


ADD server.py .


RUN pip install numpy pandas tensorflow tensorflow-cpu flwr sklearn

CMD ["python", "-u", "server.py"]