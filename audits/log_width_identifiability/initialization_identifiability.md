# Initialization identifiability

## Baseline initialization

For a query slot with normalized initial width `w_0=sigmoid(q_w)`, the baseline physical initial width is

```text
w_0_sec(D) = D w_0
```

This is static in normalized coordinates but scales linearly with audio duration. A representative `w_0=0.5` gives:

| `D` seconds | baseline `w_0` normalized | baseline `w_0_sec` | `u_0=log(w_0_sec/1 s)` |
|---:|---:|---:|---:|
| 60 | 0.500000 | 30 | 3.401197 |
| 120 | 0.500000 | 60 | 4.094345 |
| 240 | 0.500000 | 120 | 4.787492 |
| 300 | 0.500000 | 150 | 5.010635 |

The same table applies slot-wise after replacing `0.5` with each learned slot width. At `D=240`, the actual baseline median normalized width was approximately `0.5121`, or about `122.9 s`; the M1 median was approximately `0.5013`, or about `120.3 s`. The exact independently trained tensors are not identical, but both mechanisms use static normalized query widths.

## Can a log-seconds variable preserve the exact baseline initialization?

Yes, but only with duration-dependent initialization:

```text
u_0(D) = log(D * sigmoid(q_w) / tau)
```

Then `tau exp(u_0(D))/D = sigmoid(q_w)` exactly, so the normalized initialization geometry is preserved. This means the supposedly absolute log-seconds parameter is initialized as a function of `D`; it is not a static audio-length-invariant initialization.

If instead `u_0` is a single static learned value independent of `D`, then

```text
w_0(D) = tau exp(u_0)/D
```

and the normalized initialization changes with audio duration. It cannot equal the baseline static `sigmoid(q_w)` for multiple values of `D` unless `u_0` is allowed to depend on `D`.

Conversely, choosing a static physical initial width can make `w_0_sec` invariant across audios, but changes the baseline initialization geometry for all durations except the one used to set that constant. There is no free representation change that simultaneously preserves baseline normalized initialization, preserves a static absolute-seconds initialization, and removes the `D` conversion from normalized downstream paths.

## M1-specific point

M1 overwrites the center reference with a selected local center but leaves `w_0=sigmoid(q_w)` in `src/qd_detr.py:175-176`. A width intervention must therefore be audited separately from M1 center selection. It must not attribute a center change to a width initialization effect.

## Identifiability conclusion

Initialization is not cleanly separable from an absolute log-seconds representation:

1. exact baseline initialization requires `u_0(D)` and therefore preserves duration dependence in the initialization map;
2. static absolute initialization changes baseline geometry;
3. retaining original downstream normalized paths reintroduces `D` after the representation conversion.

This is a partial coupling, not a proof that a log-width intervention is invalid. It means any causal pilot must state which initialization geometry is being held fixed.
