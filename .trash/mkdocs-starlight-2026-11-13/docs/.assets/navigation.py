"""Adapt the shared navigation to the local preview and publication builds."""


def on_config(config):
    """Use generated API/notebook pages only when their plugins are enabled."""
    notebooks = "mkdocs-jupyter" in config.plugins
    generated_api = "gen-files" in config.plugins

    def adapt(items):
        result = []
        for item in items:
            for title, target in item.items():
                if isinstance(target, list):
                    children = adapt(target)
                    if children:
                        result.append({title: children})
                elif target.endswith(".ipynb") and not notebooks:
                    continue
                elif title == "API" and generated_api:
                    result.append({title: "API/index.md"})
                else:
                    result.append({title: target})
        return result

    config.nav = adapt(config.nav)
    return config
