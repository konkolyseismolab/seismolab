import numpy as np

__all__ = ['insert_gaps']

def insert_gaps(timeorig,time,brightness,max_gap_size = 0.1):
    """
    Insert gaps into a time series by getting gaps
    from another time series.

    Parameters
    ----------
    timeorig : array-like
        Time values of the original light curve.
    time : array-like
        Time values of the continuous light curve.
    brightness : array-like
        Brightness values of the continuous light curve.
    max_sz_gap : float, default: None
        Maximal size of gaps to be filled into the
        continuous light curve.

    Returns
    -------
    time_gapped : array-like
        Time values of the gap inserted light curve.
    brightness_gapped : array-like
        Brightness values of the gap inserted light curve.
    """

    maggapped = brightness.copy()
    timegapped = time.copy()

    gapat = np.where(np.diff(timeorig) > max_gap_size)[0]
    gapat = np.repeat(gapat,2)
    gapat[1::2] += 1

    for gapstart,gapend in zip(timeorig[gapat][::2],timeorig[gapat][1::2]):
        missingdata = (timegapped > gapstart) & (timegapped < gapend)
        maggapped[missingdata] = np.nan

    um = np.isfinite(maggapped)

    return timegapped[um],maggapped[um]