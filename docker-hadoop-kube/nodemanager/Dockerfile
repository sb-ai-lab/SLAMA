ARG IMAGE_TAG=slama-yarn
FROM node2.bdcl:5000/yarn-base:${IMAGE_TAG}

MAINTAINER Ivan Ermilov <ivan.s.ermilov@gmail.com>

HEALTHCHECK CMD curl -f http://localhost:8042/ || exit 1

ADD run.sh /run.sh
RUN chmod a+x /run.sh

EXPOSE 8042

RUN apt update

RUN apt install build-essential libncurses5-dev zlib1g-dev libnss3-dev libgdbm-dev libssl-dev libsqlite3-dev libffi-dev libreadline-dev curl libbz2-dev -y

RUN apt install wget

ARG PYTHON_VERSION=3.8.9
ENV PYTHON_VERSION=${PYTHON_VERSION}

RUN wget https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz

RUN tar -xvf Python-${PYTHON_VERSION}.tgz

RUN cd Python-${PYTHON_VERSION} && ./configure --enable-optimizations

RUN cd Python-${PYTHON_VERSION} && make altinstall

RUN ln -s /usr/local/bin/python$(echo ${PYTHON_VERSION} | awk -F '.' '{print $1"."$2}') /usr/bin/python3

#ADD LightAutoML-0.3.0-py3-none-any.whl /LightAutoML-0.3.0-py3-none-any.whl
#
#RUN python3 -m pip install /LightAutoML-0.3.0-py3-none-any.whl

CMD ["/run.sh"]
