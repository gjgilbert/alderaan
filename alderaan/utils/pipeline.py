__all__ = ['PipelineContext',
           'capture_locals',
           'invoke_subrecipe'
          ]

import os
import sys
import functools
import gc
import importlib.machinery
import matplotlib.pyplot as plt
from pathlib import Path
from timeit import default_timer as timer
import types


class PipelineContext:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass


def capture_locals(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        captured_locals = {}

        def tracer(frame, event, arg):
            if event == 'return' and frame.f_code == func.__code__:
                captured_locals.update(frame.f_locals)
            return tracer

        sys.setprofile(tracer)
        result = func(*args, **kwargs)
        sys.setprofile(None)

        # drop function args and private names
        exclude = ['context', 'config', 'parser', 'args', 'kwargs', 'catch']
        user_defined_locals = {}
        for k, v in captured_locals.items():
            if (k not in exclude  and not k.startswith('_') and len(k) >= 3):
                user_defined_locals[k] = v

        return user_defined_locals

    return wrapper


def invoke_subrecipe(context, subrecipe, cleanup=True, progress=True, **kwargs):
    path = Path(os.path.join(context.BASE_PATH, 'alderaan/recipes/subrecipes', subrecipe))
    name = path.stem

    loader = importlib.machinery.SourceFileLoader(name, str(path))
    module = types.ModuleType(name)
    loader.exec_module(module)

    if not hasattr(module, 'run'):
        raise AttributeError(f"{name} has no function 'run'")

    vars_dict = module.run(context, **kwargs)
    for k, v in vars_dict.items():
        setattr(context, k, v)

    if cleanup:
        sys.stdout.flush()
        sys.stderr.flush()
        plt.close('all')
        gc.collect()

    if progress:
        print(f"\ncumulative runtime = {((timer()-context.pipeline_start_time)/60):.1f} min")
