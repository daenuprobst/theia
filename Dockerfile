FROM continuumio/miniconda3:22.11.1
ENV PYTHONUNBUFFERED=1
RUN apt-get update \
    && apt-get install -y build-essential vim \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /usr/src/app
COPY ./requirements.txt .
RUN conda install pytorch cpuonly pandas=1.4.3 scikit-learn=1.1.1 -c pytorch
RUN pip install gunicorn
RUN pip install sentry-sdk[flask]
RUN pip install -U -r  requirements.txt
ENTRYPOINT ["/usr/src/app/theia/gunicorn_start.sh"]
