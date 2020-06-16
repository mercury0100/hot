# hot
[![Licence](http://img.shields.io/badge/license-GPLv3-blue.svg?style=flat)](http://www.gnu.org/licenses/gpl-3.0.html)

Applying iterative sine fitting, Oxford CBV correction and BLS search to detect transits and eclipses of hot pulsating stars in the TESS mission. 

We follow the approach of [Sowicka et al., 2017](http://adsabs.harvard.edu/abs/2017MNRAS.467.4663S), who found transits around KIC 5613330 and KIC 8197761.

## Installation Instructions

First run (with `--user` if necessary)

	`pip install astropy fitsio lightkurve`

and 

	`pip install https://github.com/dfm/acor/archive/master.zip`

Not all of the dependencies of hot can be installed via pip.

TESS cotrending basis vectors can be obtained in shell with `bash run_all.sh`.

You then need to separately clone and install [PyBLS](https://github.com/benjaminpope/PyBLS) (NB - my fork from Hannu's), [k2ps](https://github.com/hpparvi/k2ps), [PyTransit](https://github.com/hpparvi/PyTransit), [PyExoTk](https://github.com/hpparvi/PyExoTK)
