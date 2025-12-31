Annotation UI
===============

A simple annotation UI is provided to label ambiguous marker crops and collect training examples.

Usage
-----

- Start the UI: `python scripts/annotation_ui.py --host 0.0.0.0 --port 5000`
- Open the server in a browser and click an image name to annotate; draw rectangles and select a category, then click **Save**.
- Export crops for training from the same page (Export button); this saves crops under `data/labels/<category>/`.
- Ingest annotations to append to `data/annotation_labels.csv`: `python scripts/ingest_annotations.py`
- Optionally recompute canonical colors from labeled crops: `python scripts/recompute_canonical_from_labels.py`

Notes
-----

- The extractor will use `data/label_canonicals.json` if present to override builtin canonical color centers.
- Use `recompute_canonical_from_labels.py` after you have enough labeled crops to generate suggested canonical colors.
