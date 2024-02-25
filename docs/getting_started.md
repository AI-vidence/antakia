# Installation

## Install

AntakIA is available on [PyPI](https://pypi.org/project/antakia/) :

```
pip install antakia
```

## Running example notebooks

Once you've installed `antakia`, you can download some of our notebook examples from our repo [here](https://github.com/AI-vidence/antakia/tree/main/examples).

Then, launch a Jupyter server from the notebook file (`.ipynb`) location :

```
jupyter notebook # or jupyter lab
```

At least you'll need `california_housing.ipynb`and `california_housing.csv` from our repo if you want to do our [tutorial](./examples/california1.md). 


!!! Important

    If you're using a virtual env, it's handy to be able to chose it from Jupyter's kernel list. Do the following :

```
python -m ipykernel install --user --name your_venv_name --display-name "My great virtual env"
```

## Online demo

You can give `antakia` a try online  : https://demo.antakia.ai 

Log as `demo`  with password `antakia`.
Note it runs on a simple server and may be busy. You may want to log with other accounts : `demo1` to `demo5` are accepted with the same password.


## Run with Docker

!!! Important

    Be sure to have a Docker engine running on your computer (ie. launch Docker Desktop)

```
docker build -t antakia .
docker run -p 8888:8888 antakia
```

In your Terminal, click on the `http://127.0.0.1:8888/lab?token=WHATEVER_YOUR_TOKEN_IS URL` link.
