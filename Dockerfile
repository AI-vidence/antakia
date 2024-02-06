FROM jupyter/scipy-notebook

RUN pip install -U git+https://github.com/AI-vidence/antakia.git

COPY examples/ /home/jovyan/examples
COPY data/ /home/jovyan/data

EXPOSE 8888

CMD jupyter-lab