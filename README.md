# particula

Particula is a simple, fast, and powerful particle simulator, or at least two of those three, we hope. 
It is a simple particle system that is designed to be easy to use and easy to extend.
The goal is to have a robust aerosol (gas phase + particle phase) simulation system that can be used to answer simple questions that arise for experiments and research discussions. 

-Gorkowski, ...

## Goals & conduct

The main goal is to develop an aerosol particle model that is usable, efficient, and productive. 
In this process, we all will learn developing models in Python and associated packages. 
Let us all be friendly, respectful, and nice to each other. 
Any code added to this repository is automatically owned by all. 
Please speak up if something (even if trivial) bothers you. 
Talking through things always helps. 
This is an open source project, so feel free to contribute, however small or big your contribution may be.

## Notebooks & scratchbooks

The development of this model will be illustrated through Jupyter notebooks that will be put together in the form of a Jupyterbook on our website.

<!-- ## Run & viz -->

<!-- ### Local environment (recommendation: use VSCode) -->

### Windows

The quick way is to use anaconda (or miniconda) and install a new environment with the following command: 

```bash
conda env install -f requirements.txt
```



### Linux

With Docker, simply create the container defined in `.devcontainer`. Otherwise, use a Conda environment or something similar using `environment.yml`. (In VSCode, you could do these things automatically to an extent, if Docker is installed.) If Conda, [miniforge](https://github.com/conda-forge/miniforge) is recommended; once installed you need to do create and environment like,

```bash
conda env install -f requirements.txt
```

and then, make sure to add the associated kernel with this environment to your Jupyter.

```bash
python -m ipykernel install --user
```

### MacOS

Similar to Linux above.

