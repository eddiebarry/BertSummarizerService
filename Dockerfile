FROM ubuntu:18.04
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git
WORKDIR /usr/src
RUN git clone https://github.com/eddiebarry/BertSummarizerService.git
WORKDIR /usr/src/BertSummarizerService
RUN pip install -r requirements.txt
RUN gunicorn --bind 0.0.0.0:5000 wsgi:app
