ARG base_image
FROM ${base_image}

ARG SPARK_VER
ARG SYNAPSEML_VER
ARG SLAMA_VER
ARG LIGHTGBM_VER
ARG spark_jars_cache=jars_cache

USER root

RUN pip install pyspark==${SPARK_VER}

#USER ${spark_id}
RUN mkdir -p /src

COPY docker/spark-lama/download_jars.py /src/download_jars.py

# download all jars dependencies once to reuse them on each SLAMA run
RUN python3 /src/download_jars.py

COPY requirements.txt /src

RUN pip install -r /src/requirements.txt
# RUN pip install torchvision==0.9.1

COPY dist/sparklightautoml-${SLAMA_VER}-py3-none-any.whl /tmp/sparklightautoml-${SLAMA_VER}-py3-none-any.whl
RUN pip install /tmp/sparklightautoml-${SLAMA_VER}-py3-none-any.whl

RUN pip install --upgrade lightgbm==${LIGHTGBM_VER}

COPY examples/spark /examples

COPY jars /root/jars
COPY examples /examples

ENV PYSPARK_PYTHON=python3

WORKDIR /root

RUN pip install psutil

# fixing problem with SLF4j
RUN rm /root/.ivy2/jars/org.slf4j_slf4j-api-1.7.25.jar

# uncomment these lines to use with custom synapseml lightgbm build
# RUN rm /root/.ivy2/jars/com.microsoft.azure_synapseml-lightgbm_*.jar
# COPY jars/synapseml-lightgbm_2.12-1.0.8-custom.jar /root/.ivy2/jars/
