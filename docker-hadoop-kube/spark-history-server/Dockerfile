ARG IMAGE_TAG=slama-yarn
FROM node2.bdcl:5000/yarn-base:${IMAGE_TAG}

HEALTHCHECK CMD curl -f http://localhost:18080/ || exit 1

RUN curl -O https://archive.apache.org/dist/spark/spark-3.2.0/spark-3.2.0-bin-hadoop3.2.tgz

RUN tar zxvf spark-3.2.0-bin-hadoop3.2.tgz && mv spark-3.2.0-bin-hadoop3.2 /usr/local/lib

ENV SPARK_HOME=/usr/local/lib/spark-3.2.0-bin-hadoop3.2

RUN apt-get update && apt-get install -y procps && rm -rf /var/lib/apt/lists/*

ADD run.sh /run.sh
RUN chmod a+x /run.sh

ADD spark-defaults.conf /usr/local/lib/spark-3.2.0-bin-hadoop3.2/conf/spark-defaults.conf

EXPOSE 18080

CMD ["/run.sh"]
