import pytest
from numpy.testing import assert_array_almost_equal
import numpy as np

from seismolab.OC import OCFitter

@pytest.fixture
def light_curve():
    time,brightness,brightness_error = np.loadtxt('t_men_light_curve.txt',unpack=True)
    return time,brightness,brightness_error

@pytest.fixture
def get_period():
    return 0.409837

@pytest.fixture
def load_mintimes_model():
    mintimes = np.loadtxt('mintimes_model.txt',unpack=True)
    return mintimes

@pytest.fixture
def load_mintimes_poly():
    mintimes = np.loadtxt('mintimes_poly.txt',unpack=True)
    return mintimes

@pytest.fixture
def load_mintimes_nonparametric():
    mintimes = np.loadtxt('mintimes_nonparametric.txt',unpack=True)
    return mintimes

@pytest.fixture
def load_OC_model():
    midtimes, OC = np.loadtxt('midtimes_OC_model.txt',unpack=True)
    return midtimes, OC

@pytest.fixture
def load_OC_poly():
    midtimes, OC = np.loadtxt('midtimes_OC_poly.txt',unpack=True)
    return midtimes, OC

@pytest.fixture
def load_OC_nonparametric():
    midtimes, OC = np.loadtxt('midtimes_OC_nonparametric.txt',unpack=True)
    return midtimes, OC

def test_OCFitter_model(light_curve,load_mintimes_model,load_OC_model,get_period):
    time,brightness,brightness_error = light_curve

    period = get_period
    fitter = OCFitter(time, brightness, brightness_error, period)
    mintimes,_ = fitter.fit_minima()

    midtimes, OC, _ = fitter.calculate_OC()

    mintimes_in = load_mintimes_model
    midtimes_in, OC_in = load_OC_model

    assert_array_almost_equal(mintimes,mintimes_in)

    assert_array_almost_equal(midtimes,midtimes_in)
    assert_array_almost_equal(OC,OC_in)

def test_OCFitter_poly(light_curve,load_mintimes_poly,load_OC_poly,get_period):
    time,brightness,brightness_error = light_curve

    period = get_period
    fitter = OCFitter(time, brightness, brightness_error, period)
    mintimes,_ = fitter.fit_minima(fittype='poly')

    midtimes, OC, _ = fitter.calculate_OC()

    mintimes_in = load_mintimes_poly
    midtimes_in, OC_in = load_OC_poly

    assert_array_almost_equal(mintimes,mintimes_in)

    assert_array_almost_equal(midtimes,midtimes_in)
    assert_array_almost_equal(OC,OC_in)

def test_OCFitter_nonparametric(light_curve,
                                load_mintimes_nonparametric,
                                load_OC_nonparametric,
                                get_period):
    time,brightness,brightness_error = light_curve

    period = get_period
    fitter = OCFitter(time, brightness, brightness_error, period)
    mintimes,_ = fitter.fit_minima(fittype='nonparametric')

    midtimes, OC, _ = fitter.calculate_OC()

    mintimes_in = load_mintimes_nonparametric
    midtimes_in, OC_in = load_OC_nonparametric

    assert_array_almost_equal(mintimes,mintimes_in)

    assert_array_almost_equal(midtimes,midtimes_in)
    assert_array_almost_equal(OC,OC_in)
