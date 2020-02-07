FROM nvidia/cuda:10.0-cudnn7-devel

LABEL maintainer="frt frt@hongo.wide.ad.jp"

ARG ACCOUNT="frt"
ARG ACCOUNT_PW="e3tree"
#Note. You should change pw in your container(not overwrite above). Cuz everyone in this machine can read this Dockerfile.  
ARG PYTHON_VERSION="3.6.5"
ARG PYTHON_ROOT=/usr/local/bin/python
ARG JUPYTER_PW_HASH="sha1:6bc3f9f9b8c8:465b161136af835d4d26a64918457d2f6a2fdea4"


RUN apt update

RUN apt install -y sudo wget git curl build-essential vim htop
RUN apt install -y libreadline-dev libncursesw5-dev libssl-dev libsqlite3-dev libgdbm-dev libbz2-dev liblzma-dev zlib1g-dev uuid-dev libffi-dev libdb-dev libglib2.0-0 libsm6 libxrender1 libxext6


#ADD USER
RUN groupadd -g 1000 developer && \
    useradd  -g      developer -G sudo -m -s /bin/bash ${ACCOUNT} && \
    echo ${ACCOUNT}:${ACCOUNT_PW} | chpasswd

#INSTALL PYTHON (use install support tool, which is a part of pyenv)
RUN git clone https://github.com/pyenv/pyenv.git ~/.pyenv && cd ~/.pyenv/plugins/python-build && ./install.sh
RUN /usr/local/bin/python-build -v ${PYTHON_VERSION} ${PYTHON_ROOT}
RUN rm -rf ~/.pyenv
ENV PATH $PATH:$PYTHON_ROOT/bin

RUN pip install --upgrade setuptools pip
RUN pip install numpy tensorflow-gpu==2.0.0 tensorflow_hub
RUN pip install scikit-image opencv-python seaborn matplotlib tqdm jupyter pandas xgboost sklearn pyarrow

RUN mkdir /root/.jupyter && touch /root/.jupyter/jupyter_notebook_config.py
RUN echo "c.NotebookApp.ip = '0.0.0.0'" >> /root/.jupyter/jupyter_notebook_config.py
RUN echo c.NotebookApp.open_browser = False >> /root/.jupyter/jupyter_notebook_config.py
RUN echo c.NotebookApp.password = \'${JUPYTER_PW_HASH}\' >> /root/.jupyter/jupyter_notebook_config.py

CMD jupyter notebook --allow-root 
