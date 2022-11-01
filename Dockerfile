FROM nvcr.io/nvidia/pytorch:21.09-py3

RUN apt-get update && apt-get install -y libgl1-mesa-glx

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

ENV APP_PATH="/app"
RUN mkdir -p ${APP_PATH}
WORKDIR ${APP_PATH}

RUN pip uninstall -y pillow
RUN pip install pillow

ENV PYTHONPATH "${PYTHONPATH}:/app/"

# docker build -t $(whoami)/kapao .