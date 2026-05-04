import sys
import pytest

# Import test functions from mlcolvar
from mlcolvar.utils.export import (
    test_export_1,
    test_export_2,
    test_export_3,
)

# Check whether the current platform is Linux
# AOTInductor (used in export) is only fully supported on Linux
IS_LINUX = sys.platform.startswith("linux")

# ============================================================
# Pytest compatibility
# ============================================================

# When running under pytest (e.g., in CI), tests are collected
# automatically based on their names (test_*). The __main__ block
# is NOT executed in that case.
#
# Therefore, we dynamically mark the imported test functions as
# "skipped" on non-Linux platforms so that pytest will not try
# to execute them and fail.

if not IS_LINUX:
    test_export_1 = pytest.mark.skip(
        reason="AOTInductor requires Linux"
    )(test_export_1)

    test_export_2 = pytest.mark.skip(
        reason="AOTInductor requires Linux"
    )(test_export_2)

    test_export_3 = pytest.mark.skip(
        reason="AOTInductor requires Linux"
    )(test_export_3)


# ============================================================
# Script entry point (manual execution)
# ============================================================

# This block is only executed when running the file directly:
#     python test_utils_export.py
#
# It is NOT used by pytest.

if __name__ == "__main__":
    if IS_LINUX:
        # Run export tests normally on Linux
        test_export_1()
        test_export_2()
        test_export_3()
    else:
        # Gracefully skip when running manually on non-Linux systems
        print("Skipped: export tests require Linux (AOTInductor)")