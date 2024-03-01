# Welcome to AntakIA !

AntakIA is an open-source tool from AI-vidence to explain ML black box models. See [doc](https://doc.antakia.ai).

Here is a quick overview on AntakIA:
![AntakIA demo](/docs/img/antakia.gif)

See full video on [Youtube](https://youtu.be/wQFC_20OIOM).

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

You'll find a complete tutorial for our California housing example here : https://doc.antakia.ai

> [!IMPORTANT] 
If you're using a virtual env, it's handy to be able to chose it from Jupyter's kernel list. Do the following :

```
python -m ipykernel install --user --name your_venv_name --display-name "My great virtual env"
```

## Online demo

You can give `antakia` a try online  : https://demo.antakia.ai 

Log as `demo`  with password `antakia`.
Note it runs on a simple server and may be busy.


## Run with Docker

> [!IMPORTANT] 
Be sure to have a Docker engine running on your computer (ie. launch Docker Desktop)

```
docker build -t antakia .
docker run -p 8888:8888 antakia
```

In your Terminal, click on the `http://127.0.0.1:8888/lab?token=WHATEVER_YOUR_TOKEN_IS URL` link.

## Logs

`antakia` is our open-source project. We plan to learn a lot from it to build our SaaS product, and earn our living. By default, `antakia` sends anonymous usage logs to our server. With this data, we cannot discover who you are, what you do, or what data you work on. It only tells us how people use the software. However, if you don't want us to receive this data, just create a `.env` file at the root of your project folder with the following line `SEND_ANONYMOUS_LOGS=false`