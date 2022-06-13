FROM mcr.microsoft.com/vscode/devcontainers/python:0.203.3-3.10

COPY . /workspaces/particula/
RUN pip3 --disable-pip-version-check --no-cache-dir install -e "/workspaces/particula/[dev]"
