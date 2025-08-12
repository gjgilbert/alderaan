__all__ = ['PipelineContext',
           'capture_locals',
           'capture_context'
          ]

import os
import sys
import importlib.machinery
import importlib.util
from pathlib import Path
import types


class PipelineContext:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass


def capture_locals(func):
    def wrapper(*args, **kwargs):
        local_vars = {}
        def tracer(frame, event, arg):
            if event == 'return':
                local_vars.update(frame.f_locals)
            return tracer

        sys.setprofile(tracer)
        result = func(*args, **kwargs)
        sys.setprofile(None)

        # Filter out private variables and function args if needed
        filtered = {k: v for k, v in local_vars.items() if not k.startswith('_')}
        return filtered
    
    return wrapper


def invoke_subrecipe(context, subrecipe):
    path = Path(os.path.join(context.BASE_PATH, 'alderaan/recipes/subrecipes', subrecipe))
    name = path.stem

    loader = importlib.machinery.SourceFileLoader(name, str(path))
    module = types.ModuleType(name)
    loader.exec_module(module)

    if not hasattr(module, 'run'):
        raise AttributeError(f"{name} has no function 'run'")

    vars_dict = module.run(context)
    for k, v in vars_dict.items():
        setattr(context, k, v)
