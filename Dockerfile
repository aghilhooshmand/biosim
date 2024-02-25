
ARG PYTHON_VERSION=3.11.7
FROM python:${PYTHON_VERSION}-slim as base
RUN useradd --create-home --shell /bin/bash biosim
WORKDIR /home/biosim
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
USER biosim
COPY . .
RUN chmod 777 .
CMD ["bash"]