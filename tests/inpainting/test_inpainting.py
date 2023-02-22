import pytest
from numpy.testing import assert_array_almost_equal
import numpy as np

from seismolab.inpainting import kinpainting, insert_gaps

@pytest.fixture
def light_curve():
    np.random.seed(12345)

    time = np.linspace(0,7,300)

    time = time[ np.random.choice( np.arange(time.shape[0]), size=int(time.shape[0]*.7), replace=False) ]
    time = np.sort(time) + 1400.1

    mag = 0.1 * np.sin(2*np.pi*time/0.14)

    mag += np.random.normal(0,0.005,time.shape[0])

    return time,mag

@pytest.fixture
def load_inpainted_irreg():
    inpainted_irreg = np.loadtxt('inpainted_irreg.txt',unpack=True)
    return inpainted_irreg

@pytest.fixture
def load_time_mag_inpainted():
    time_inpainted,mag_inpainted = np.loadtxt('time_mag_inpainted.txt',unpack=True)
    return time_inpainted,mag_inpainted

def test_inpainting(light_curve,load_inpainted_irreg,load_time_mag_inpainted):
    time,mag = light_curve

    _, inpainted_irreg = kinpainting(time,mag)

    inpainted_irreg_in = load_inpainted_irreg.T

    assert_array_almost_equal(inpainted_irreg,inpainted_irreg_in)


    time_inpainted = inpainted_irreg[:,0]
    mag_inpainted  = inpainted_irreg[:,1]

    time_inpainted, mag_inpainted = insert_gaps(time,
                                                time_inpainted,
                                                mag_inpainted,
                                                max_gap_size=0.1)

    time_inpainted_in,mag_inpainted_in = load_time_mag_inpainted

    assert_array_almost_equal(time_inpainted,time_inpainted_in)
    assert_array_almost_equal(mag_inpainted,mag_inpainted_in)



