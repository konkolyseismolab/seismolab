import pytest
from numpy.testing import assert_array_almost_equal
import numpy as np

from seismolab.tfa import windowed_lomb_scargle
from seismolab.tfa import gabor, wavelet, choi_williams

@pytest.fixture
def light_curve():
    np.random.seed(12345)

    time = np.linspace(0,1.2,50)
    mag = 0.1 * np.sin(2*np.pi*time/0.12 )

    # Add time and brightness nullpoint
    time = time + 1400.
    mag  = mag + 10.

    # Randomly generated and add noise
    mag += np.random.normal(0,0.005,time.shape[0])

    return time,mag

@pytest.fixture
def load_powers_wls():
    powers_wls = np.loadtxt('powers_wls.txt')
    return powers_wls

def test_windowed_lomb_scargle(light_curve,load_powers_wls):
    time,mag = light_curve

    _, _, powers_wls = windowed_lomb_scargle(time,mag)

    powers_wls_in = load_powers_wls

    assert_array_almost_equal(powers_wls,powers_wls_in)

@pytest.fixture
def load_powers_gbr():
    powers_gabor = np.loadtxt('powers_gabor.txt')
    return powers_gabor

def test_gabor(light_curve,load_powers_gbr):
    time,mag = light_curve

    _, _, powers_gbr = gabor(time,mag)

    powers_gbr_in = load_powers_gbr

    assert_array_almost_equal(powers_gbr,powers_gbr_in)

@pytest.fixture
def load_powers_wavelet():
    powers_wavelet = np.loadtxt('powers_wavelet.txt')
    return powers_wavelet

def test_wavelet(light_curve,load_powers_wavelet):
    time,mag = light_curve

    _, _, powers_wavelet = wavelet(time,mag)

    powers_wavelet_in = load_powers_wavelet

    assert_array_almost_equal(powers_wavelet,powers_wavelet_in)

@pytest.fixture
def load_powers_choi_williams():
    powers_choi_williams= np.loadtxt('powers_choi_williams.txt')
    return powers_choi_williams

def test_wavelet(light_curve,load_powers_choi_williams):
    time,mag = light_curve

    _, _, powers_choi_williams = choi_williams(time,mag,M=128/2)

    powers_choi_williams_in = load_powers_choi_williams

    assert_array_almost_equal(powers_choi_williams,powers_choi_williams_in)





