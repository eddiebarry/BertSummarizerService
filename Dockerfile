FROM nvidia/cuda:10.2-base
RUN apt-get update && apt-get install --no-install-recommends --no-install-suggests -y curl
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git && \
    apt-get -y install python3 &&\
    apt-get install -y python3-pip

RUN pip3 install --upgrade pip
WORKDIR /usr/src
ADD "https://www.random.org/cgi-bin/randbyte?nbytes=10&format=h" skipcache
RUN git clone https://github.com/eddiebarry/BertSummarizerService.git
WORKDIR /usr/src/BertSummarizerService
RUN pip install -r requirements.txt
#RUN gunicorn --bind 0.0.0.0:5000 wsgi:app --timeout 600
