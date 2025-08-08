import os
import sys

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if base_path not in sys.path:
    sys.path.insert(0, base_path)

from alderaan.schema.litecurve import LiteCurve
from alderaan.utils.io import parse_koi_catalog, parse_holczer16_catalog


def execute():
    print('\n\nI/O BLOCK\n')

    # load KOI catalog
    catalog = parse_koi_catalog(catalog_csv, target)

    assert np.all(np.diff(catalog.period) > 0), "Planets should be ordered by ascending period"

    NPL = int(catalog.npl[0])
    koi_id = catalog.koi_id[0]
    kic_id = int(catalog.kic_id[0])

    # load lightcurves
    #litecurve_master = LiteCurve(data_dir, kic_id, 'long cadence', data_source='Kepler PDCSAP')
    litecurve_master = LiteCurve().from_kplr_pdcsap(data_dir, kic_id, 'long cadence', visits=2)

    t_min = litecurve_master.time.min()
    t_max = litecurve_master.time.max()
    if t_min < 0:
        raise ValueError("Lightcurve has negative timestamps...this will cause problems")

    # split litecurves by quarter
    litecurves = litecurve_master.split_visits()

    for j, litecurve in enumerate(litecurves):
        assert len(np.unique(litecurve.visit)) == 1, "expected one quarter per litecurve"
        assert len(np.unique(litecurve.obsmode)) == 1, "expected one obsmode per litecurve"

    print(f"{len(litecurves)} litecurves loaded for {target}")

    # initialize planets (catch no ephemeris warning)
    with warnings.catch_warnings(record=True) as catch:
        warnings.simplefilter('always', category=UserWarning)
        planets = [None]*NPL
        for n in range(NPL):
            planets[n] = Planet(catalog, target, n)

    print(f"\n{NPL} planets loaded for {target}")
    print([np.round(p.period,6) for p in planets])

    # update planet ephemerides
    for n, p in enumerate(planets):
        if p.ephemeris is None:
            _ephemeris = Ephemeris(period=p.period, epoch=p.epoch, t_min=t_min, t_max=t_max)
            planets[n] = p.update_ephemeris(_ephemeris)

    # load Holczer+2016 catalog
    filepath = os.path.join(catalog_dir, 'holczer_2016_kepler_ttvs.txt')
    holczer_ephemerides = parse_holczer16_catalog(filepath, koi_id, NPL)

    print(f"\n{len(holczer_ephemerides)} ephemerides found in Holczer+2016")

    # match Holczer ephemerides to Planets
    count = 0

    for n, p in enumerate(planets):
        for ephem in holczer_ephemerides:
            match = np.isclose(ephem.period, p.period, rtol=0.01, atol=p.duration)

            if match:
                print(f"  Planet {n} : {p.period:.6f} --> {ephem.period:.6f}")
                planets[n] = p.update_ephemeris(ephem)
                count += 1

    print(f"{count} matching ephemerides found ({len(holczer_ephemerides)} expected)")

    # quicklook litecurve
    filepath = os.path.join(quicklook_dir, f"{target}_litecurve_raw.png")
    _ = plot_litecurve(litecurve_master, target, planets, filepath)

    # end-of-block cleanup
    cleanup()

    print(f"\ncumulative runtime = {((timer()-global_start_time)/60):.1f} min")



if __name__ == '__execute__':
    execute()