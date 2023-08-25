# :computer: Getting Started

<!-- Installation -->

### :gear: Installation with pip

Install using `pip`

```
pip install antakia
```

<!-- V-env -->
### :house: Local installation
   
Clone the repo and create a virtual environment!

```
git clone https://github.com/AI-vidence/antakia.git
cd antakia
python3.10 -m venv .
source bin/activate
pip install -e .
```

### :whale: Installation with docker

Be sure to have a Docker engine running on your computer.

```
docker build -t antakia .
docker run -p 8888:8888 antakia
```