
import os

import requests
from bs4 import BeautifulSoup
import logging
import pandas as pd
import re
import subprocess
import tempfile
import urllib.request
import datetime


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

    def get_parent_url(self):
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
                    match = re.search(r"(http.*/)(kplr\d+-\d+_llc\.fits)", line)
                    if match:
                        return match.group(1)  # parent directory URL
        raise RuntimeError("Could not find a valid parent directory URL.")

    def download_llc_fits_from_directory(self, base_url):

        response = requests.get(base_url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        fits_links = [
            a['href'] for a in soup.find_all('a', href=True)
            if a['href'].endswith("_llc.fits") or a['href'].endswith("_slc.fits")
        ]

        missing = []
        for link in fits_links:
            fits_url = base_url + link
            local_path = os.path.join(self.output_dir, link)
            if os.path.exists(local_path):
                logging.info(f"Already exists: {link}")
                continue

            logging.info(f"Downloading: {fits_url}")
            try:
                r = requests.get(fits_url)
                r.raise_for_status()
                with open(local_path, "wb") as f:
                    f.write(r.content)
            except Exception as e:
                logging.error(f"Failed to download {fits_url}: {e}")
                missing.append(fits_url)
        return missing

    def run(self, max_retries=2):
        base_url = self.get_parent_url()
        for attempt in range(max_retries):
            missing = self.download_llc_fits_from_directory(base_url)
            if not missing:
                logging.info(f"All _llc.fits and _slc.fits files downloaded for KIC {self.kic_id}.")
                return True
            else:
                logging.warning(f"Retrying {len(missing)} files... (Attempt {attempt + 2})")
        logging.error(f"Final missing files for KIC {self.kic_id}: {missing}")
        return missing

class KeplerDownloader:
    @staticmethod
    def run_all(koi_ids, catalog_path, script_path, output_dir, project_root="."):
        catalog = KeplerCatalog(project_root=project_root, catalog_dir=catalog_path)
        kic_ids = catalog.get_kic_ids(koi_ids)

        if not kic_ids:
            print("No matching KIC_IDs found for the given KOI_IDs.")
            return

        for koi_id, kic_id in zip(koi_ids, kic_ids):
            print(f"Processing KOI_ID: {koi_id} → KIC_ID: {kic_id}")

            # Generate a unique timestamp for the log filename
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_koi_id = koi_id.replace(" ", "_").replace("/", "_")
            log_filename = f"kepler_fetcher_koi_id_{safe_koi_id}_{timestamp}.log"
            log_file_path = os.path.join(output_dir, log_filename)

            fetcher = KeplerDataFetcher(
                kic_id=kic_id,
                get_kepler_data_py_path=script_path,
                output_dir=output_dir,
                log_file=log_file_path
            )

            result = fetcher.run(max_retries=2)

            if result is True:
                print(f"All files for {koi_id} downloaded successfully.\n")
            else:
                print(f"Some files for {koi_id} could not be downloaded. See log for details.\n")
