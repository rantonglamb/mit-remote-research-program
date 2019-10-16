FROM cassandra:latest
WORKDIR /app
COPY . /app
RUN mv /etc/apt/sources.list.d/cassandra.list /etc/apt/sources.list.d/cassandra.list.bak
RUN mv /etc/apt/sources.list /etc/apt/sources.list.bak
RUN set -e; \
        apt-get update;\
        apt-get install -y --no-install-recommends \
        python3-pip \
        python3-setuptools;

RUN pip3 install -r requirements.txt
EXPOSE 80
CMD ./services_script.sh