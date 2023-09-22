FROM python:3.10

RUN apt-get update && apt-get install -y python3 python3-pip git

COPY . /demo/

WORKDIR /demo

RUN pip3 install -r requirements.txt
# RUN pip install git+https://github.com/scikit-learn-contrib/skope-rules.git
RUN pip3 install --editable .

EXPOSE 8888

CMD ["jupyter-lab","--port=8888 ", "--ip=0.0.0.0","--allow-root", "--no-browser", "--NotebookApp.iopub_data_rate_limit=1.0e10"]