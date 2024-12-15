from loguru import logger

import aphrodite.common.envs as envs


def load_general_plugins():
    """WARNING: plugins can be loaded for multiple times in different
    processes. They should be designed in a way that they can be loaded
    multiple times without causing issues.
    """
    import sys
    if sys.version_info < (3, 10):
        from importlib_metadata import entry_points
    else:
        from importlib.metadata import entry_points

    allowed_plugins = envs.APHRODITE_PLUGINS

    discovered_plugins = entry_points(group='aphrodite.general_plugins')
    for plugin in discovered_plugins:
        logger.info(f"Found general plugin: {plugin.name}")
        if allowed_plugins is None or plugin.name in allowed_plugins:
            try:
                func = plugin.load()
                func()
                logger.info(f"Loaded general plugin: {plugin.name}")
            except Exception:
                logger.exception("Failed to load general plugin: "
                                 f"{plugin.name}")
