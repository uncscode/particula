# particula

Particula is a simple, fast, and powerful particle system, or at least it aims to be two of those three. 
It is a simple particle system that is designed to be easy to use and easy to extend.
The goal is to have a robust aerosol (gas phase + particle phase) simulation system that can be used to answer simple questions that arise for experiments and research discussions. 

-Gorkowski, ...

## Goals & conduct

This is an open source project, so feel free to contribute. 

## Notebooks & scratchbooks

Each aerosol process will have its own notebook to discuss the dynamics and limitations of the implication. 
These notebooks will be joined in to a jypyter book. 


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

### Codespaces

The Codespaces feature is enabled for this organization. Click on "Code" (in green) on the right hand side and you can take it from there. Read more [here](https://github.com/features/codespaces). Basically a virtual machine that could be run in the browser or in VSCode.

### Browser

For light editing, try pressing the dot (`.`) on your keyboard, and this repository should turn from `github.com/uncscode/particula` to `github.dev/uncscode/particula` (changing from `.com` to `.dev`) --- now you could edit the files and commit! Read more [here](https://docs.github.com/en/codespaces/the-githubdev-web-based-editor). Related: [vscode.dev](https://vscode.dev).

<!-- ## Experimental features

Planned future activities

### Inverse methods + ML/DL

### Instrument inversions -->
