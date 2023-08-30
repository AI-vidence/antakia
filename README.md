<div align="center">

  <img src="antakia/assets/logo_ai-vidence.png" alt="logo" width="200" height="auto" />
  </br>
  <img src="antakia/assets/logo_antakia.png" alt="logo" width="300" height="auto" />
  <p>
    The XAI solution for your model!
  </p>

  <a href="https://ai-vidence.github.io/antakia/"><strong>Explore the docs »</strong></a>

</div>

<br />

<!-- Table of Contents -->

# :notebook_with_decorative_cover: Table of Contents

- [About the Project](#star2-about-the-project)
  - [Our idea](#thought_balloon-our-idea)
  - [Screenshots](#camera-screenshots)
- [Getting Started](#computer-getting-started)
  - [Installation using pip](#gear-installation-with-pip)
  - [Installation with Docker](#whale-installation-with-docker)
- [Usage](#eyes-usage)
- [Type of data currently supported](#chart_with_upwards_trend-type-of-data-currently-supported)
- [Notes on some python librairies](#snake-comments-on-some-of-the-python-librairies-we-are-using)
- [Contributing](#wave-contributing)
  - [Code of Conduct](#scroll-code-of-conduct)
- [FAQ](#grey_question-faq)
- [License](#warning-license)
- [Contact](#handshake-contact)
- [Acknowledgements](#gem-acknowledgements)

<!-- About the Project -->

## :star2: About the Project

<!-- Notre idée -->

### :thought_balloon: Our idea

Our Python library combines all the steps of our regional approach for your model:

- Simultaneous "dyadic" exploration of the value and explanation spaces of your model to be explained,
- Automatic recommendation of dyadic segmentation (regions that are both homogeneous in the 2 spaces) or manual selection
- For each region, expression of the selection made by the input or explained variables (cf. "Skope rules")
- Possibility of refining the definition of regions for each attribute (see "flows")
- For each region, a surrogate model is proposed from a library (including PiML and iModels)
- Performance, continuity and completeness testing of the final model

<div align="center"> 
  <img src="antakia/assets/gif_antakia.gif" alt="AntaKIA idea" />
</div>

<!-- Screenshots -->

### :camera: Screenshots

<div align="center"> 
  <img src="antakia/assets/git_screen.png" alt="screenshot" />
</div>

<!-- Getting Started -->

## :computer: Getting Started

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

[! IMPORTANT] 
Be sure to have a Docker engine running on your computer (ie. launch Docker Desktop)

```
docker build -t antakia .
docker run -p 8888:8888 antakia
```

Then click on the http://127.0.0.1:8888/lab?token=WHATEVER_YOUR_TOKEN_IS URL

[!NOTE]
If you stop `jupyterlab` you can relaunch it with `docker run -p 8888:8888 antakia`

<!-- Usage -->

## :eyes: Usage

Example of usage (find more example in the <a href="https://code.ai-vidence.com/laurent/antakia/">example</a> folder)

In a notebook :

```python
import pandas as pd
df = pd.read_csv('data/california_housing.csv')
X = df.iloc[:,0:8]
Y = df.iloc[:,9]
```

```python
from sklearn.ensemble import GradientBoostingRegressor
model = GradientBoostingRegressor(random_state = 9)
model.fit(X, Y)
```

```python
import antakia
explain = antakia.Xplainer(X = X, Y = Y, model = model)
display(explain.interface(explanation = SHAP, default_projection = "PaCMAP"))
```

<!-- Roadmap -->

## :chart_with_upwards_trend: Type of data currently supported

- [x] Tabular data
- [ ] Time series
- [ ] Images

<!-- Note sur les librairies -->

## :snake: Comments on some of the python librairies we are using

### SHAP

- Does not work with numpy >= 1.24 (see [here](https://github.com/slundberg/shap/issues/2911))
- For the moment : version 1.23 in `requirements.txt`

### Numba

- Numba depreciated [here](https://github.com/slundberg/shap/issues/2909)
- Waiting for the merge to be accepted [here](https://github.com/dsgibbons/shap/pull/9)
- The exceptions are caught in the code !

### Skope rules

- [The repository](https://github.com/scikit-learn-contrib/skope-rules) is fine, but note the PyPi version.
- Install it using github (like in `requirements.txt`) : `pip install git+https://github.com/scikit-learn-contrib/skope-rules.git`

<!-- Contributing -->

## :wave: Contributing

Contributions are always welcome!

See `CONTRIBUTING.md` for ways to get started.

<!-- Code of Conduct -->

### :scroll: Code of Conduct

Please read the [Code of Conduct](https://github.com/Louis3797/awesome-readme-template/blob/master/CODE_OF_CONDUCT.md)

<!-- FAQ -->

## :grey_question: FAQ

- Is AntakIA open source ?

  - Yes ! And forever !

- Can I transfer my computations on a huge GPU ?

  - Yes, very soon on the [ai-vidence](http://www.ai-vidence.com) website

<!-- License -->

## :warning: License

Distributed under the no License. See LICENSE.txt for more information.

<!-- Contact -->

## :handshake: Contact

:computer: [www.ai-vidence.com](http://www.ai-vidence.com)

:inbox_tray: laurent@ai-vidence.com

:inbox_tray: david@ai-vidence.com

<!-- Acknowledgments -->

## :gem: Acknowledgements

We wrote this paper about AntakIA :
[Antakia](www.ai-vidence.com)
