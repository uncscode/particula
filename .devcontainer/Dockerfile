FROM mcr.microsoft.com/vscode/devcontainers/python

COPY . /workspaces/particula/
RUN pip3 --disable-pip-version-check --no-cache-dir install -e "/workspaces/particula/[dev]"
