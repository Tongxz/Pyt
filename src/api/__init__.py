# API module
# API模块

__all__ = [
    'app',
    'routes',
    'models',
    'middleware'
]

def __getattr__(name):
    if name == 'app':
        from .app import app
        return app
    elif name == 'routes':
        from . import routes
        return routes
    elif name == 'models':
        from . import models
        return models
    elif name == 'middleware':
        from . import middleware
        return middleware
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")