import pytest
from numpy.testing import assert_array_almost_equal
import numpy as np

from seismolab.template import TemplateFitter

@pytest.fixture
def light_curve():
    time,brightness,brightness_error = np.loadtxt('t_men_light_curve.txt',unpack=True)
    return time,brightness,brightness_error

@pytest.fixture
def template_OC():
    times, amp, amperr, phase, phaseerr, zp, zperr = \
    np.loadtxt('st_pic_template_OC.txt',unpack=True)
    return times, amp, amperr, phase, phaseerr, zp, zperr

@pytest.fixture
def template_lc():
    template = np.loadtxt('st_pic_template_lc.txt',unpack=True)
    return template

@pytest.fixture
def template_lc_interp():
    template = np.loadtxt('st_pic_template_lc_interp.txt',unpack=True)
    return template

def test_TemplateFitter(light_curve,template_OC,template_lc,template_lc_interp):
    time,brightness,brightness_error = light_curve

    fitter = TemplateFitter(time, brightness, brightness_error)
    times, amp, amperr, phase, phaseerr, zp, zperr = fitter.fit(maxharmonics=5)
    template = fitter.get_lc_model()
    template_interp = fitter.get_lc_model_interp()

    times_in, amp_in, amperr_in, phase_in, phaseerr_in, zp_in, zperr_in = template_OC
    template_in = template_lc
    template_interp_in = template_lc_interp

    assert_array_almost_equal(times,times_in)
    assert_array_almost_equal(amp,amp_in)
    assert_array_almost_equal(amperr,amperr_in)
    assert_array_almost_equal(phase,phase_in)
    assert_array_almost_equal(phaseerr,phaseerr_in)
    assert_array_almost_equal(zp,zp_in)
    assert_array_almost_equal(zperr,zperr_in)

    assert_array_almost_equal(template,template_in)
    assert_array_almost_equal(template_interp,template_interp_in)