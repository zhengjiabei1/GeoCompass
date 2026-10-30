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

## Manifest and command line

A collection is described by a `manifest.json`; no code changes are needed to
run on it.

```json
{
  "name": "sen1floods11-sample",
  "band_names": ["blue", "green", "red", "nir", "swir1", "swir2"],
  "scale": 0.0001,
  "size": 32,
  "entries": [
    {
      "id": "Bolivia_103757",
      "family": "flood",
      "split": "target_test",
      "pre": ["Bolivia_103757_S2Hand.tif"],
      "labels": "Bolivia_103757_LabelHand.tif",
      "region": "bolivia",
      "season": "wet"
    }
  ]
}
```

- `pre` / `post` are a list of per-band paths in `band_names` order, the same
  list keyed by band name, or a single multi-band file.
- `scale` and `offset` are applied to every band; they depend on the product.
- `size` resamples each chip to a square; omit it to keep the source size.
- `split` is one of `source_train`, `source_val`, `target_dev`, `target_test`,
  `stream`. A collection that only defines families may omit `source_train`; one
  representative per family is then used to induce the skill library.

```bash
python -m geocompass experiment --dataset data/sen1floods11 --seeds 0,1,2 --output experiments/real
python -m geocompass report --results experiments/real/results.json
```

The runner reports the same metrics as on synthetic data, so the comparison that
matters is between methods within the collection rather than against the
synthetic numbers.

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

Sen1Floods11 stores 13 Sentinel-2 bands in acquisition order, so the six bands
used here are indices 1, 2, 3, 7, 11, 12 (`B02`, `B03`, `B04`, `B08`, `B11`,
`B12`). Reflectance is stored as integers at 1e-4, hence `scale: 0.0001`. The
hand-labelled mask uses -1 for no data, 0 for land, and 1 for water; remap -1 to
the nodata value before writing the label raster, otherwise unobserved pixels
count as land.

## Converting a Sen1Floods11 download

`scripts/prepare_sen1floods11.py` does the mechanical part: it selects the six
bands, rescales reflectance, remaps the label encoding, and writes the manifest.

```bash
python scripts/prepare_sen1floods11.py \
    --images data/raw/S2Hand --labels data/raw/LabelHand \
    --out data/sen1floods11 --family flood --size 32 \
    --split-map splits.json
```

`splits.json` maps chip ids to splits; without it every chip goes to
`--default-split`. `--dry-run` lists what would be converted, and `--limit`
caps the run, which is useful for a first pass. The script never downloads and
never writes outside `--out`.

## What to expect

The synthetic generator produces clean, spatially contiguous scenes with
class-specific signatures.  Real chips bring cloud edges, sub-pixel mixing,
co-registration error, and label noise, all of which lower absolute scores.
The comparison that remains meaningful is *relative*: the same selector, the
same gate, and the same skills on the same chips.
