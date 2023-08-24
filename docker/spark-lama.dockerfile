FROM python:3.9.9

RUN wget https://download.java.net/openjdk/jdk11/ri/openjdk-11+28_linux-x64_bin.tar.gz
RUN tar -xvf openjdk-11+28_linux-x64_bin.tar.gz
RUN mv jdk-11 /usr/local/lib/jdk-11
RUN ln -s /usr/local/lib/jdk-11/bin/java /usr/local/bin/java

RUN mkdir -p /src
COPY dist/sparklightautoml_dev-0.3.2-py3-none-any.whl /src
RUN pip install /src/sparklightautoml_dev-0.3.2-py3-none-any.whl

RUN python3 -c 'from pyspark.sql import SparkSession; SparkSession.builder.config("spark.jars.packages", "com.microsoft.azure:synapseml_2.12:0.11.1-spark3.3").config("spark.jars.repositories", "https://mmlspark.azureedge.net/maven").getOrCreate()'

RUN mkdir /src/jars
COPY jars/spark-lightautoml_2.12-0.1.1.jar /src/jars/

COPY examples /src/examples
COPY examples/data /opt/spark_data

WORKDIR /src

ENTRYPOINT ["python3"]
