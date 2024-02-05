# Welcome to AntakIA !

AntakIA is an open-source tool from AI-vidence to explain ML black box models.

Here is a quick tour of AntakIA:
![AntakIA demo](/docs/img/antakia.gif)

See full video on [Youtube](https://youtu.be/wQFC_20OIOM).

## Install and run from source code

Clone the repo and create a virtual environment

```
git clone https://github.com/AI-vidence/antakia.git
cd antakia
```

Then, with `pip` :

```
python3.10 -m venv .
source bin/activate
pip install -e .
```

Or,if you prefer `poetry` :

```
poetry shell
peotry install
```

Got to https://doc.antakia.ai to learn how to use AntakIA.

You may also need to install Git LFS to download our CSV files in /data:
```
brew install git-lfs # if you're a Mac / brew user
git lfs install
cd antakia
git lfs track "*.csv"
git add .gitattributes
git lfs fetch
git lfs checkout
```

Reference : [see Git-LFS website](https://git-lfs.com)

> [!IMPORTANT] 
Install your antakia env in IPython, in order to select it from Jupyter :

```
python -m ipykernel install --user --name antakia --display-name "Antakia"
```

Finally, launch Jupyter server from `antakia` folder :
```
jupyter notebook
```
and open a notebook .ipynb file in `example` folder.

### Troubleshooting 

If you get JS errors in your notebook / see broken link icons for each widget / see this kind of errors `404 GET /static/jupyterlab-plotly.js` or other missing JS library :
* try to remove your local `etc/` folder in `antakia` folder
* try `jupyter lab` instead of `jupyter notebook`
* probably useless but worth trying : `jupyter nbextension enable --py widgetsnbextension`
* try to `pip install --force-reinstall` ipywidgets widgetsnbextension
 

## Install and run with Docker

> [!IMPORTANT] 
Be sure to have a Docker engine running on your computer (ie. launch Docker Desktop)

```
docker build -t demo-antakia .
docker run -p 8888:8888 demo-antakia
```

In your Terminal, click on the `http://127.0.0.1:8888/lab?token=WHATEVER_YOUR_TOKEN_IS URL` link.
