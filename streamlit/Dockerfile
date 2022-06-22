FROM python:3.10-slim AS builder

RUN apt-get update
RUN apt-get install -y --no-install-recommends build-essential gcc openjdk-17-jdk-headless python3-wheel
RUN mkdir -p /app

ENV VIRTUAL_ENV=/app/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

ARG PYSPARK_VERSION=3.3.0

RUN pip --no-cache-dir install \
        pyspark==${PYSPARK_VERSION} \
        pyspark[sql]==${PYSPARK_VERSION} \
        folium plotly \
        streamlit streamlit-folium neuralprophet[live]

# Image for running
FROM python:3.10-slim
EXPOSE 8501

RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends openjdk-17-jre-headless; \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /app /app

ENV PATH="/app/venv/bin:$PATH"
WORKDIR /app

COPY app.py /app/app.py
COPY accidents_ts.np /app/accidents_ts.np
COPY injured_ts.np /app/injured_ts.np
COPY deadly_ts.np /app/deadly_ts.np

CMD ["streamlit", "run", "app.py"]
