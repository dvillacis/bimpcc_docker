FROM python:3.12

# Set the working directory
WORKDIR /usr/src/app

COPY requirements.txt ./

RUN apt-get update && apt-get install -y \
    coinor-libipopt-dev libblas-dev liblapack-dev libmetis-dev 

RUN pip install --no-cache-dir -r requirements.txt

