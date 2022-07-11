# SLAMA: LightAutoML on Spark

SLAMA is a version of [LightAutoML library](https://github.com/sb-ai-lab/LightAutoML) modified to run in distributed mode with Apache Spark framework.

It requires:
1. Python 3.9
2. PySpark 3.2+ (installed as a dependency)
3. [Synapse ML library](https://microsoft.github.io/SynapseML/)
   (It will be downloaded by Spark automatically)

Currently, only tabular Preset is supported. See demo with spark-based tabular automl
preset in [examples/spark/tabular-preset-automl.py](https://github.com/sb-ai-lab/SLAMA/tree/main/examples/spark/tabular-preset-automl.py).
For further information check docs in the root of the project containing dedicated SLAMA section.

<a name="apache"></a>
# License
This project is licensed under the Apache License, Version 2.0. See [LICENSE](https://github.com/sb-ai-lab/SLAMA/tree/main/LICENSE) file for more details.


# Installation
First of all you need to install [git](https://git-scm.com/downloads) and [poetry](https://python-poetry.org/docs/#installation).

```bash

# Load SLAMA source code
git clone https://github.com/sb-ai-lab/SLAMA.git

cd SLAMA/

# !!!Choose only one item!!!

# 1. Recommended: Create virtual environment inside your project directory
poetry config virtualenvs.in-project true

# 2. Global installation: Don't create virtual environment
poetry config virtualenvs.create false --local

# For more information read poetry docs

# Install SLAMA
poetry lock
poetry install
```

# Cluster preparation

You can found cluster preparation for using SLAMA [here](https://github.com/sb-ai-lab/SLAMA/blob/main/docker-hadoop/README.md)
