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
