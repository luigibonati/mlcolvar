import sys
import pytest

from mlcolvar.utils.export import test_export_gnn as _test_export_gnn


IS_LINUX = sys.platform.startswith("linux")


@pytest.mark.skipif(
    not IS_LINUX,
    reason="AOTInductor requires Linux",
)
def test_export_gnn():
    _test_export_gnn()


if __name__ == "__main__":
    if IS_LINUX:
        _test_export_gnn()
    else:
        print("Skipped: export tests require Linux (AOTInductor)")