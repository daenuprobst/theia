FROM pytorch/pytorch:latest
ENV PYTHONUNBUFFERED=1
RUN apt-get update \
    && apt-get install -y build-essential \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /usr/src/app
COPY ./src src 
COPY ./MANIFEST.in . 
COPY ./LICENSE .
COPY ./setup.cfg .
COPY ./setup.py .
COPY ./pyproject.toml .
COPY ./README.md .
COPY ./gunicorn_start.sh .
RUN pip install gunicorn
RUN pip install .
RUN theia-download
RUN chmod +x gunicorn_start.sh
ENTRYPOINT ["./gunicorn_start.sh"]
