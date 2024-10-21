FROM python:3.12

# Set the working directory
WORKDIR /usr/src/app

COPY requirements.txt ./

RUN apt-get update && apt-get install -y \
    coinor-libipopt-dev libblas-dev liblapack-dev libmetis-dev 

RUN pip install --no-cache-dir -r requirements.txt

# Habilitar colores en el prompt de bash
RUN echo "force_color_prompt=yes" >> ~/.bashrc

# Crear alias y agregarlos a ~/.bashrc
RUN echo "alias ll='ls -la --color=auto'" >> ~/.bashrc && \
    echo "alias gs='git status'" >> ~/.bashrc && \
    echo "alias gd='git diff'" >> ~/.bashrc && \
    echo "alias ..='cd ..'" >> ~/.bashrc

# Cambia el shell predeterminado a bash
SHELL ["/bin/bash", "-c"]

