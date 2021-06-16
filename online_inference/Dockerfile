FROM python:3.7-slim-stretch

COPY requirements_docker.txt ./requirements.txt
RUN pip install -r requirements.txt

COPY configs/ ./configs/
COPY models/ ./models/
COPY src/ ./src/

EXPOSE 8000

ENV HOST '0.0.0.0'

CMD ["uvicorn", "src.main_app:app"]
