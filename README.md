# SGHMC

[![Build Status](https://travis-ci.org/JiajunSong629/SGHMC.svg?branch=master)](https://travis-ci.org/JiajunSong629/SGHMC)

The repository for STA663 Final Project. Focus on the implementation and optimizaiton of SGHMC sampler.

## Project structure

Here is a brief overview of the repo.

```bash
.
├── README.md                       <-- This instructions file
├── reports
│   ├── report_2song.pdf            <-- our final report
|   └── 1402.4102.pdf               <-- Chen et.al 2014, original paper
├── sghmc                           <-- sghmc package source code
│   ├── __init__.py    
│   ├── preprocess.py               <-- data preprocess
|   └── sghmc.py                    <-- sghmc sampler
├── experiments    
│   ├── Figure_*.ipynb              <-- figures 1-5 in experiments
│   └── bayesnn                     <-- figure 6, bayes neural network
├── notebooks    
│   ├── eigen/                      <-- dependencies for optimization
│   ├── Optimization_demo.ipynb     <-- optimization work
│   └── Comparative_analysis.ipynb  <-- comparative analysis work
├── requirements.txt                <-- dependencies you need to run the example
└── setup.py             
```


## How to use

### To use the package

```bash
pip install -i https://test.pypi.org/simple/ sghmc-2song
```

### To review the project

- Clone the repository in sta663 server or a similar docker container

```bash
git clone https://github.com/JiajunSong629/SGHMC.git
cd SGHMC/
```

- Create the virtural environment and activate. Intall the dependencies and you should see the list of packages installed. sghmc-2song is the one this project implements. Others will be needed in experiment and optimization notebooks.

```bash
python3 -m venv .sghmc
source .sghmc/bin/activate
python -m pip install --upgrade pip wheel
pip install -i https://test.pypi.org/simple/ sghmc-2song
pip install -r requirements.txt
pip list
```

- Add the environment for ipython kernel and change kernel to `.sghmc` when launching the notebook.

```bash
python -m ipykernel install --user --name=.sghmc
```

- Add the dependencies to compile C++

```bash
cd notebooks/
git clone https://gitlab.com/libeigen/eigen.git
```