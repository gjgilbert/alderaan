"""Tests for alderaan/utils/plotting.py: PNG output via save_and_close"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

from alderaan.utils.plotting import DPI, save_and_close  # noqa: E402


class TestSaveAndClose:
    def test_writes_palette_png_and_closes(self, tmp_path):
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.plot([0, 1], [0, 1])
        fig_path = tmp_path / "fig.png"

        save_and_close(fig, str(fig_path))

        with Image.open(fig_path) as im:
            assert im.format == "PNG"
            assert im.mode == "P"
            assert im.size == (4 * DPI, 3 * DPI)
        assert not plt.fignum_exists(fig.number)

    def test_none_path_keeps_figure_open(self):
        fig = plt.figure()
        save_and_close(fig, None)
        assert plt.fignum_exists(fig.number)
        plt.close(fig)
