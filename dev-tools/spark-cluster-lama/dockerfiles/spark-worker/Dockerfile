
FROM node2.bdcl:5000/spark-base:3.2.0-hadoop3.2

COPY worker.sh /

ENV SPARK_WORKER_WEBUI_PORT 8081
ENV SPARK_WORKER_LOG /spark/logs
ENV SPARK_MASTER "spark://spark-master:7077"

EXPOSE 8081

COPY requirements.txt requirements.txt
# RUN pip install -r requirements.txt

CMD ["/bin/bash", "/worker.sh"]