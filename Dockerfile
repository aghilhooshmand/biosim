
ARG PYTHON_VERSION=3.11.7
FROM python:${PYTHON_VERSION}-slim as base
RUN useradd -ms --create-home --shell /bin/bash biosim
COPY requirements.txt ./
COPY . .
WORKDIR /home/biosim
RUN pip install --no-cache-dir -r requirements.txt
RUN chown -R biosim:biosim /home/biosim
RUN chmod 777 /home/biosim
USER biosim
CMD ["bash"]
