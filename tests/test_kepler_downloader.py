
from kepler_downloader import KeplerDownloader

# Set paths (modify these to match your system)
catalog_path = "/Users/admin/Documents/alderaan/tests/catalogs/kepler_dr25_gaia_dr2_crossmatch.csv"
script_path = "/Users/admin/Documents/alderaan/bin/get_kepler_data.py"
output_dir = "/Users/admin/kepler_fits"

# List of KOI_IDs you want to process
koi_ids = ["K01234", "K04567"]

# Call the main function — this does everything
KeplerDownloader.run_all(
    koi_ids=koi_ids,
    catalog_path=catalog_path,
    script_path=script_path,
    output_dir=output_dir
)