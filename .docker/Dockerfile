FROM python:3.9
LABEL maintainer="whata <waku.hatakeyama@gmail.com>"

ENV PYTHONBUFFERED=1

RUN apt-get update -y \
  && apt-get install -y --no-install-recommends \
    sudo \
    bc \
    bzip2 \
    ca-certificates \
    curl \
    git \
    libssl-dev \
    libffi-dev \
	  libncurses-dev \
    libgl1 \
    jq \
    nfs-common \
    parallel \
    unzip \
	  vim \
    wget \
    build-essential \
    sshfs \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*
SHELL ["/bin/bash", "-c"]

COPY requirements.txt /tmp
WORKDIR /tmp
RUN python3.9 -m pip install -r requirements.txt

WORKDIR /workspace

# # Add user
# ARG USER_ID
# ARG GROUP_ID
# RUN addgroup --gid $GROUP_ID user
# RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user
# RUN usermod -aG sudo user
# RUN passwd -d user
# USER user

# Create python alias
RUN mkdir -p $HOME/bin
RUN ln -s /usr/bin/python3.9 $HOME/bin/python
ENV PATH $PATH:$HOME/bin:$HOME/.local/bin
ENV PYTHONPATH /workspace

RUN ["/bin/bash"]
