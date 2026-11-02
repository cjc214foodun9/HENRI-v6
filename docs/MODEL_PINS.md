# Model pins

An encoder loaded by name is not reproducible. `sentence-transformers/all-MiniLM-L6-v2`
resolves to whatever upstream published last, so a new upstream commit changes
every embedding, every slot digest, and any test that pins a byte.

## The pin

| Item | Value |
|---|---|
| Model | `sentence-transformers/all-MiniLM-L6-v2` |
| Revision | `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` |
| Embedding dimension | 384 |
| Loaded by | `henri.semantic_backbone.MODEL_REVISION`, `henri.axiom_seed.MODEL_REVISION` |
| Recorded environment | sentence-transformers 5.5.1, transformers 5.9.0, torch 2.11.0+cu128, Python 3.14.0 |
| Digest pins | `tests/pins/embedding_pins.json` |

Both loaders pass `revision=` explicitly. `StemInfo` reports it, so a receipt can
say which encoder produced a measurement.

## The digest pins

`tests/test_phase10_structural_morphisms.py` compares two slot regions byte for
byte. The digests live in `tests/pins/embedding_pins.json` together with the
revision and the library versions that produced them.

A pin is only meaningful inside one environment. So the test:

1. skips with a named reason when the running revision differs,
2. skips with a named reason when a library version differs,
3. asserts the digest only when both match.

That is deliberate. A bare digest mismatch is an environment change reported as
a code failure, which is how this suite went red while nothing in the code had
moved. A skip says what changed.

## Regenerating a pin

Only when the encoder change is intended:

```powershell
# from the v6 checkout
$env:PYTHONPATH = $PWD
python scripts/record_embedding_pins.py --out tests/pins/embedding_pins.json
```

Then review the diff of the two digests, the revision and the versions as one
change. The script refuses to pin when the encoder fell back to the character
n-gram stem, because a pin from the fallback would describe a different model.

## What still moves

The encoder is fixed. The tokenizer stack is fixed by the recorded versions. A
Python or torch upgrade can still change floating-point results, which is why the
versions are recorded and why a mismatch skips instead of failing.
