FROM python:3.8-slim
 
WORKDIR /app

COPY get-poetry.py ./

RUN python get-poetry.py --yes && \
    rm get-poetry.py && \
    . "$HOME/.poetry/env" && \
    poetry config virtualenvs.create false

COPY pyproject.toml poetry.lock ./

RUN . "$HOME/.poetry/env" && \
    poetry install --no-root

COPY epimodel luigi.cfg logging.conf ./
