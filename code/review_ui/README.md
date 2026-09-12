# AMR manual review UI — Short Moment Probe 30

This is a dependency-free local inspection tool for the fixed 30-case short-moment probe. It does not run a model and does not infer scientific labels automatically.

## Run

```bash
cd /private/research-artifact
python serve.py
```

Then open:

```text
http://127.0.0.1:8000
```

If the server is remote, use SSH port forwarding from your local machine:

```bash
ssh -L 8000:127.0.0.1:8000 -p 18477 <private-server>
```

The UI expects manually populated audio files at:

```text
/private/research-artifact{vid}.wav
```

Missing audio is shown as `AUDIO NOT FOUND`; the case metadata and review form remain usable.

## Persistence and export

Review saves are sent to `data/reviews.json` by `serve.py` and keyed by qid. The page exports the current manifest plus saved labels as `manual_reviews.csv` or `manual_reviews.json`.

The 30-case selection is fixed with seed `20260820` and must not be changed after review begins. Existing v0A outputs and the 100-case review sample are separate and are not overwritten.

## Keyboard shortcuts

- Left/Right Arrow: previous/next case
- Space: play/pause
- `S`: save
- `G`: jump to first GT
- `P`: jump to Top-1 prediction
