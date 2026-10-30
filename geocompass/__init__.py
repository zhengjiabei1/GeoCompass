"""GeoCompass: calibrated skill selection and tiered memory for spatiotemporal tasks.

The package is deliberately dependency-free (Python standard library only) so
that the reported experiments can be reproduced on a CPU-only machine.
"""

__version__ = "0.2.0"

from .io import GeoRaster, GeoTIFFError, read_geotiff, read_geotiff_scene  # noqa: F401
from .types import BANDS, CLASS_NAMES, FAMILIES, Skill, TaskSpec  # noqa: F401
