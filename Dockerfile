FROM python:3.8

RUN apt-get update && \
    apt-get install -y openjdk-11-jre net-tools wget nano iputils-ping curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV SCALA_VERSION=2.12.10
RUN	wget http://scala-lang.org/files/archive/scala-${SCALA_VERSION}.deb && \
    dpkg -i scala-${SCALA_VERSION}.deb

ENV SPARK_VERSION=3.2.1
ENV HADOOP_VERSION=3.2
RUN wget https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz && \
    tar -xvzf spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz && \
    mv spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION} spark && \
    rm spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz

ENV SPARK_HOME=/spark

RUN pip install poetry && poetry config virtualenvs.create false
WORKDIR /lama
COPY . .
