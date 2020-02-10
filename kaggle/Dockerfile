FROM kaggle/python

ARG JUPYTER_PW_HASH="sha1:6bc3f9f9b8c8:465b161136af835d4d26a64918457d2f6a2fdea4"


RUN touch /root/.jupyter/jupyter_notebook_config.py
RUN echo "c.NotebookApp.ip = '0.0.0.0'" >> /root/.jupyter/jupyter_notebook_config.py
RUN echo c.NotebookApp.open_browser = False >> /root/.jupyter/jupyter_notebook_config.py
RUN echo c.NotebookApp.password = \'${JUPYTER_PW_HASH}\' >> /root/.jupyter/jupyter_notebook_config.py

CMD jupyter notebook --allow-root 
