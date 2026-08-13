FROM python:3.10.4

WORKDIR /app

COPY application/requirements.txt .
RUN pip install -r requirements.txt

COPY application/model_application.py .
COPY feast /app/feast

ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=UTF-8
ENV MPLCONFIGDIR=/tmp/mplconfig
ENV FEAST_REPO_PATH=/app/feast

ENV GRADIO_SERVER_PORT=7860
ENV GRADIO_SERVER_NAME=0.0.0.0

EXPOSE 7860

ENTRYPOINT ["python"]
CMD ["model_application.py"]
