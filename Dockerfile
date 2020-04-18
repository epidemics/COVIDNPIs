FROM python:3.8-slim 
ENV PYTHONUNBUFFERED 1

WORKDIR /app
COPY get-poetry.py pyproject.toml poetry.lock ./
RUN \
  python get-poetry.py --yes && \
  . "$HOME/.poetry/env" && \
  poetry config virtualenvs.create false && \
  rm get-poetry.py && \
  pip install --no-cache-dir --upgrade pip && \
  poetry install --no-dev

COPY epimodel epimodel
COPY do pipeline.sh ./

