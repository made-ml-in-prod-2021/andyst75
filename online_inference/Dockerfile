FROM python:3.7-slim
COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

COPY configs/ ./configs/
COPY models/ ./models/
COPY src/ ./src/

CMD ["uvicorn", "src.main_app:app"]
