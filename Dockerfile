FROM python:3.11

RUN apt-get update && apt-get install -y python3 python3-pip git

WORKDIR /jup

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
RUN pip install git+https://github.com/scikit-learn-contrib/skope-rules.git


COPY antakia.py antakia.py
COPY california_housing.ipynb california_housing.ipynb

RUN mkdir -p data && cd data && mkdir california_housing
COPY data/california_housing/* data/california_housing/

RUN mkdir assets
COPY assets/* assets/ 

EXPOSE 8888

CMD ["jupyter-lab","--port=8888 ", "--ip=0.0.0.0","--allow-root", "--no-browser"]