import pytest
from numpy.testing import assert_array_almost_equal
import numpy as np

from seismolab.gaia import query_gaia

@pytest.fixture
def targets():
    return [1022096392749670016,3107142829863485952]

@pytest.fixture
def results_dr3():
    results = np.genfromtxt('result_table_dr3.txt')[1:]
    return results

def test_query_gaia_dr3(targets,results_dr3):
    targets = targets

    result_table = query_gaia(targets)
    result_table = result_table.to_pandas().values.astype(float)

    results_in = results_dr3

    assert_array_almost_equal(result_table,results_in)

