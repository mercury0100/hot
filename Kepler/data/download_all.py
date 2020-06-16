from astropy.table import Table, join, Column
import lightkurve
from lightkurve import KeplerLightCurveFile, KeplerLightCurve

fname_targets = 'simon_binaries.tsv'
bins = Table.read(fname_targets,format='ascii',delimiter=';')

for targ in bins:
    kic = targ['KIC']
    print('Doing KIC %s' % str(kic))
    lcs = lightkurve.search_lightcurvefile(kic)
    lcs.download_all()

print('Done!')