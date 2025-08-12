from pathlib import Path

from alderaan.recipes.context import PipelineContext, capture_context
from alderaan.recipes.subrecipes import startup, load_kepler_data

BASE_PATH = Path(__file__).resolve().parents[2]

def main():
    print("\n\n\nstarting main pipeline")

    with PipelineContext() as context:
        context.BASE_PATH = BASE_PATH

        capture_context(context, startup.run(context))
        capture_context(context, load_kepler_data.run(context))


    print("\n\n\nexiting main pipeline")


if __name__ == '__main__':
    main()
