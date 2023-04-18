import pytest
import urllib
from mlcolvar.utils.io import load_dataframe
from mlcolvar.utils.io import test_datasetFromFile

example_files = {'str': 'mlcolvar/tests/data/state_A.dat',
                 'list': ['mlcolvar/tests/data/state_A.dat','mlcolvar/tests/data/state_B.dat'], 
                 'url': 'https://raw.githubusercontent.com/luigibonati/mlcolvar/main/mlcolvar/tests/data/2d_model/COLVAR_stateA'}

@pytest.mark.parametrize("file_type", ['str','list','url'])
def test_loadDataframe(file_type):

    filename = example_files[file_type]
    if file_type == 'url':
        # disable test if connection is not available
        try:
            urllib.request.urlopen(filename)
        except urllib.error.URLError:
            pytest.skip('internet not available')

    df = load_dataframe(filename, start=0, stop=10, stride=1)

if __name__ == "__main__":
    #test_loadDataframe()
    test_datasetFromFile() 