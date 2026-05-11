from contextlib import contextmanager
from importlib import resources


@contextmanager
def data_dir():
    """Yield a filesystem path to the packaged test data directory."""
    if hasattr(resources, "files"):
        with resources.as_file(resources.files(__name__) / "data") as path:
            yield path
    else:
        with resources.path(__name__, "data") as path:
            yield path

# TODO this needs to be changed upon merging into main
@contextmanager
def github_data_dir():
    yield "https://github.com/luigibonati/mlcolvar/raw/refs/heads/release/2.0/mlcolvar/tests/data"