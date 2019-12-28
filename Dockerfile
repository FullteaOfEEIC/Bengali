FROM nvidia/cuda:9.0-cudnn7-devel

LABEL maintainer="frt frt@hongo.wide.ad.jp"

ARG ACCOUNT="frt"
ARG PYTHON_VERSION="3.6.5"
ARG PYTHON_ROOT=/usr/local/bin/python


RUN apt update

RUN apt install -y sudo wget git curl build-essential vim htop
RUN apt install -y libreadline-dev libncursesw5-dev libssl-dev libsqlite3-dev libgdbm-dev libbz2-dev liblzma-dev zlib1g-dev uuid-dev libffi-dev libdb-dev 


#ADD USER
RUN groupadd -g 1000 developer && \
    useradd  -g      developer -G sudo -m -s /bin/bash ${ACCOUNT} && \
    echo ${ACCOUNT}:e3tree | chpasswd

#INSTALL PYTHON (use install support tool which is a part of pyenv)
RUN git clone https://github.com/pyenv/pyenv.git ~/.pyenv && cd ~/.pyenv/plugins/python-build && ./install.sh
RUN /usr/local/bin/python-build -v ${PYTHON_VERSION} ${PYTHON_ROOT}
RUN rm -rf ~/.pyenv
ENV PATH $PATH:$PYTHON_ROOT/bin

RUN pip install --upgrade setuptools pip
RUN pip install numpy tensorflow-gpu keras

RUN pip install seaborn matplotlib tqdm jupyter pandas xlrd xgboost sklearn
