# *PARTICULA*

Particula is a simple, fast, and powerful particle simulator, or at least two of those three, we hope. It is a simple particle system that is designed to be easy to use and easy to extend. The goal is to have a robust aerosol (gas phase + particle phase) simulation system that can be used to answer simple questions that arise for experiments and research discussions.

## Goals & conduct

The main goal is to develop an aerosol particle model that is usable, efficient, and productive. In this process, we all will learn developing models in Python and associated packages. Let us all be friendly, respectful, and nice to each other. Any code added to this repository is automatically owned by all. Please speak up if something (even if trivial) bothers you. Talking through things always helps. This is an open source project, so feel free to contribute, however small or big your contribution may be.

We follow the Google Python style guide [here](https://google.github.io/styleguide/pyguide.html).

## Usage & development

The development of this model will be illustrated through Jupyter notebooks that will be put together in the form of a Jupyter book on our website. The model can be run in a container or a virtual environment. With Docker, simply create the container defined in `.devcontainer/`. If Conda, [miniforge](https://github.com/conda-forge/miniforge) is recommended; once installed you will need to create the environment like in the command below.

```bash
conda create -n particula --file requirements.txt
```
