# Running GeoCompass on real imagery

The benchmark generator exists so that the reuse decision can be studied with
known ground truth.  The same interfaces accept real chips, and this note
describes the contract.

## Contract

A scene is one or two dates plus, optionally, a label raster:

| Field | Meaning |
|---|---|
| `pre` | band name to flat row-major reflectance, one entry per `types.BANDS` |
| `post` | same for the second date; bi-temporal families need it |
| `truth` | integer label per pixel, matching the task's `truth_kind` |
| `valid` | 1 for observed pixels, 0 for nodata |
| `samples`, `sample_labels` | labelled reference pixels for supervised skills |

`geocompass.io.read_geotiff` decodes the files; `geocompass.realdata.load_scene`
builds the `Scene`.  Missing bands raise `MissingBandError` instead of being
silently zero-filled.

```python
from geocompass.realdata import SceneSpec, load_scene
from geocompass.types import BANDS

scene = load_scene(
    "s2_B02.tif",              # or a list of single-band files
    band_names=BANDS,          # blue, green, red, nir, swir1, swir2
    labels_path="labels.tif",
    size=32,                   # nearest-neighbour resample
    spec=SceneSpec(family="water", scale=1e-4),
)
```

Sentinel-2 Level-2A stores integer reflectance at 1e-4, so `scale=1e-4` is the
usual setting; Landsat Collection 2 uses 2.75e-5 with an offset.  Both are
caller-supplied because they depend on the product.

## Supported files

- classic TIFF and BigTIFF, little- or big-endian;
- stripped and tiled layouts;
- uncompressed and Deflate (zlib) segments, including the horizontal
  differencing predictor;
- 8/16/32/64-bit integer and 16/32/64-bit float samples;
- chunky and planar sample storage;
- `ModelPixelScale`, `ModelTiepoint`, `ModelTransformation`, `GeoKeyDirectory`,
  and `GDAL_NODATA`.

Not supported, and rejected with a clear error: LZW, JPEG, WebP, ZSTD, LERC,
per-sample bit depths that differ inside one file, and palettes expanded to RGB.
Only the first image file directory is read.  If a collection ships Cloud
Optimized GeoTIFFs with JPEG compression, re-encode the chip to Deflate first
(`gdal_translate -co COMPRESS=DEFLATE`) or extend `io.py`.

## Building a directory of chips

Any collection with per-pixel labels works.  Three are convenient because the
label semantics match families the benchmark already defines:

| Collection | Family | Notes |
|---|---|---|
| Sen1Floods11 | `water`, `flood` | Sentinel-1/2 chips with hand-labelled water masks |
| EuroSAT | `landcover`, `crop` | 64x64 patches with class labels; expand to a dense mask |
| LoveDA | `landcover`, `urban` | 1024x1024, urban and rural scenes |
| LEVIR-CD | `change` | bi-temporal pairs with building-change masks |

The adapter needs six optical bands; collections that ship RGB only (EuroSAT)
have to be paired with the corresponding Sentinel-2 scene, or restricted to the
skills that only use visible bands, which the current library does not offer.

## What to expect

The synthetic generator produces clean, spatially contiguous scenes with
class-specific signatures.  Real chips bring cloud edges, sub-pixel mixing,
co-registration error, and label noise, all of which lower absolute scores.
The comparison that remains meaningful is *relative*: the same selector, the
same gate, and the same skills on the same chips.
