import sys

from mlcolvar.utils.export import test_export_1, test_export_2, test_export_3

IS_LINUX = sys.platform.startswith("linux")

if __name__ == "__main__":
    if IS_LINUX:
        test_export_1()
        test_export_2()
        test_export_3()
    else:
        print("Skipped: export tests require Linux (AOTInductor)")