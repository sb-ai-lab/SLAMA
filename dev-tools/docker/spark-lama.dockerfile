FROM spark-pyspark-python:3.9-3.2.0

WORKDIR /src

RUN pip install poetry
RUN poetry config virtualenvs.create false --local

# we need star here to make copying of poetry.lock conditional
COPY requirements.txt /src

# workaround to make poetry not so painly slow on dependency resolution
# before this image building: poetry export -f requirements.txt > requirements.txt
RUN pip install -r requirements.txt

COPY poetry.lock /src
COPY pyproject.toml /src
RUN poetry install

COPY . /src
RUN poetry build
RUN pip install dist/LightAutoML-0.3.0-py3-none-any.whl
