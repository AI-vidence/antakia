
## Install and run from source code

Clone the repo and create a virtual environment!

```
git clone https://github.com/AI-vidence/antakia.git
cd antakia
python3.10 -m venv .
source bin/activate
pip install -r requirements.txt
pip install -e .
```

You'll also need to install Git LFS to download our CSV files in /data:
```
brew install git-lfs # if you're a Mac / brew user
git lfs install
cd antakia
git lfs trakc "*.csv"
git add .gitattributes
```

Reference : [see Git-LFS website](https://git-lfs.com)

Finally, launch Jupyter server from `antakia` folder :
```
jupyter notebook
```
and open the notebook .ipynb file in `example` folder.

## Install and run with Docker

> [!IMPORTANT] 
Be sure to have a Docker engine running on your computer (ie. launch Docker Desktop)

```
docker build -t antakia .
docker run -p 8888:8888 antakia
```

In your Terminal, click on the `http://127.0.0.1:8888/lab?token=WHATEVER_YOUR_TOKEN_IS URL` link.
