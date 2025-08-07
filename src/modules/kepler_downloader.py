
import os

import logging
import pandas as pd
import re
import subprocess
import tempfile
import urllib.request


class KeplerCatalog:
    def __init__(self, project_root, catalog_dir): 
        self.project_root = project_root
        self.catalog_dir = catalog_dir
        self.catalog = pd.read_csv(self.catalog_dir, index_col=0)

    def get_kic_ids(self, koi_ids):
        if isinstance(koi_ids, str):
            koi_ids = [koi_ids]
        kic_ids = []
        for koi in koi_ids:
            match = self.catalog[self.catalog.koi_id == koi]
            if not match.empty:
                kic_ids.append(str(match['kic_id'].values[0]))
        return kic_ids


class KeplerDataFetcher:
    def __init__(self, kic_id, get_kepler_data_py_path, output_dir, log_file='kepler_fetcher.log'):
        self.kic_id = kic_id
        self.script_path = get_kepler_data_py_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        logging.basicConfig(
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ],
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

        logging.info("=" * 60)
        logging.info(f"Starting processing for KIC {self.kic_id}")

    def fetch_fits_urls(self):
        fits_urls = []
        with tempfile.TemporaryDirectory() as tmpdir:
            command = [
                "python", self.script_path,
                str(self.kic_id),
                "-c", "long",
                "--cmdtype", "wget"
            ]
            subprocess.run(command, cwd=tmpdir, check=True)
            logging.info("Ran get_kepler_data.py successfully")

            sh_path = os.path.join(tmpdir, "get_kepler_data.sh")
            with open(sh_path, "r") as f:
                for line in f:
                    match = re.search(r"(http.*kplr\d+-\d+_llc)", line)
                    if match:
                        fits_urls.append(match.group(1))

        logging.info(f"🔎 Parsed {len(fits_urls)} expected .fits URLs")
        return fits_urls

    def check_and_download(self, base_urls):
        missing = []
        for base_url in base_urls:
            fits_url = base_url + ".fits"
            local_filename = os.path.join(self.output_dir, os.path.basename(fits_url))
            if not os.path.exists(local_filename):
                logging.info(f"Downloading missing file: {fits_url}")
                try:
                    urllib.request.urlretrieve(fits_url, local_filename)
                except Exception as e:
                    logging.error(f"Download failed for {fits_url}: {e}")
                    missing.append(fits_url)
        return missing

    def run(self, max_retries=2):
        base_urls = self.fetch_fits_urls()
        for attempt in range(max_retries):
            missing = self.check_and_download(base_urls)
            if not missing:
                logging.info(f"All files for KIC {self.kic_id} downloaded successfully.")
                return True
            else:
                logging.warning(f"Retrying {len(missing)} files... (Attempt {attempt + 2})")
                base_urls = [url[:-5] for url in missing]
        logging.error(f"Final missing files for KIC {self.kic_id}: {missing}")
        return missing


class KeplerDownloader:
    @staticmethod
    def run_all(koi_ids, catalog_path, script_path, output_dir, project_root=".", log_file="kepler_fetcher.log"):
        catalog = KeplerCatalog(project_root=project_root, catalog_dir=catalog_path)
        kic_ids = catalog.get_kic_ids(koi_ids)

        if not kic_ids:
            print("No matching KIC_IDs found for the given KOI_IDs.")
            logging.warning("No KIC_IDs found for input KOI_IDs.")
            return

        for koi_id, kic_id in zip(koi_ids, kic_ids):
            print(f"Processing KOI_ID: {koi_id} → KIC_ID: {kic_id}")
            fetcher = KeplerDataFetcher(kic_id=kic_id,
                                        get_kepler_data_py_path=script_path,
                                        output_dir=output_dir,
                                        log_file=log_file)
            result = fetcher.run(max_retries=2)

            if result is True:
                print(f"All files for {koi_id} downloaded successfully.\n")
            else:
                print(f"Some files for {koi_id} could not be downloaded. See log for details.\n")
