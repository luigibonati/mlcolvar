import pytest

from mlcvs.utils.data.dataloader import test_fasttensordataloader
from mlcvs.utils.data.datamodule import test_tensordatamodule

if __name__ == "__main__":
    test_fasttensordataloader()
    test_tensordatamodule()