import pytest
from numpy.testing import assert_array_almost_equal
import numpy as np

from seismolab.gaia import query_gaia

@pytest.fixture
def targets():
    return [4267549637851481344,2033008274099620480]

@pytest.fixture
def results_dr3():
    results = np.genfromtxt('result_table_dr3.txt')[1:]
    return results

@pytest.fixture
def results_dr3_photodistances():
    results = np.genfromtxt('result_table_dr3_photodist.txt')[1:]
    return results

@pytest.fixture
def results_dr2():
    results = np.genfromtxt('result_table_dr2.txt')[1:]
    return results

@pytest.fixture
def results_dr2_offset():
    results = np.genfromtxt('result_table_dr2_Stassun.txt')[1:]
    return results

def test_query_gaia_dr3(targets,results_dr3):
    targets = targets

    result_table = query_gaia(targets,dustmodel='Marshall06')
    result_table = result_table.to_pandas().values.astype(float)

    results_in = results_dr3

    assert_array_almost_equal(result_table,results_in)

def test_query_gaia_dr3_photodistances(targets,results_dr3_photodistances):
    targets = targets

    result_table = query_gaia(targets,use_photodist=True,dustmodel='Marshall06')
    result_table = result_table.to_pandas().values.astype(float)

    results_in = results_dr3_photodistances

    assert_array_almost_equal(result_table,results_in)

def test_query_gaia_dr2(targets,results_dr2):
    targets = targets

    result_table = query_gaia(targets,gaiaDR=2,dustmodel='Marshall06')
    result_table = result_table.to_pandas().values.astype(float)

    results_in = results_dr2

    assert_array_almost_equal(result_table,results_in)

def test_query_gaia_dr2_offset(targets,results_dr2_offset):
    targets = targets

    result_table = query_gaia(targets,gaiaDR=2,plx_offset='Stassun',dustmodel='Marshall06')
    result_table = result_table.to_pandas().values.astype(float)

    results_in = results_dr2_offset

    assert_array_almost_equal(result_table,results_in)

