
ARG PYTHON_VERSION=3.11.7
FROM python:${PYTHON_VERSION}-slim as base
RUN useradd --create-home --shell /bin/bash biosim
WORKDIR /home/biosim
COPY requirements.txt ./
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
USER biosim
RUN chown -R biosim:biosim /home/biosim
RUN chmod 777 /home/biosim
USER biosim
CMD ["bash"]
