# Loss identifiability

## Definitions

For centered intervals, use a common center and compare widths. The baseline normalized width L1 is

```text
L_norm_width = |p-g|/D
```

The implementation's `F.l1_loss` averages center and width, so for a zero center error its span-loss contribution is `|p-g|/(2D)`. The absolute log-width alternative is

```text
L_log_width = |log(p/g)|
```

For centered nested intervals, `GIoU=IoU=min(p,g)/max(p,g)` and `L_giou=1-GIoU`. These values are independent of `D` for the same physical ratio.

## Scalar comparison

| GT / predicted seconds | `D` | baseline normalized width L1 | implemented mean span L1 | absolute log-width error | centered GIoU | GIoU loss |
|---|---:|---:|---:|---:|---:|---:|
| 1 / 5 | 60 | 0.066667 | 0.033333 | 1.609438 | 0.200000 | 0.800000 |
| 1 / 5 | 240 | 0.016667 | 0.008333 | 1.609438 | 0.200000 | 0.800000 |
| 2 / 5 | 60 | 0.050000 | 0.025000 | 0.916291 | 0.400000 | 0.600000 |
| 2 / 5 | 240 | 0.012500 | 0.006250 | 0.916291 | 0.400000 | 0.600000 |
| 20 / 24 | 60 | 0.066667 | 0.033333 | 0.182322 | 0.833333 | 0.166667 |
| 20 / 24 | 240 | 0.016667 | 0.008333 | 0.182322 | 0.833333 | 0.166667 |

The log error and centered GIoU do not change when only `D` changes, while the normalized L1 changes by `240/60=4`. The weighted baseline span contribution consequently has the same duration factor; the weighted GIoU contribution does not.

## Gradients with respect to internal width

For the baseline normalized coordinate `w=p/D`, away from equality,

```text
d L_norm_width / d w = sign(p-g)
```

but with respect to physical seconds at fixed `D`, `dL/dp=sign(p-g)/D`. With Variant A's `u=log(p/tau)`,

```text
d L_norm_width / d u = sign(p-g) * p/D
```

For Variant B,

```text
d L_log_width / d u = sign(p-g)
```

The Variant B width gradient therefore removes both the explicit duration factor and the current-width magnitude factor. This is precisely why B cannot be interpreted as a representation-only result.

For Variant C, the final loss remains normalized, so its `v=log(w)` gradient is `sign(w-g/D)*w`; the loss geometry is unchanged and retains duration dependence through `g/D`.

## Which mechanism is removed?

- A: none of the normalized-loss duration dependence is removed; only the internal chain rule changes.
- B: explicit `D` dependence is removed from the width log-loss, but the loss itself is changed and GIoU/final geometry remain coupled through the conversion.
- C: no loss mechanism is removed; only refinement geometry changes.

## Loss conclusion

Loss is separately isolatable only by retaining the baseline final coordinate, initialization, refinement, attention, and matcher, then changing one declared width-loss term. A log-seconds representation with a log loss is not that intervention. The most auditable loss-only pilot would be a predeclared width-loss replacement in the existing normalized output, but it should be treated as a loss-geometry experiment rather than evidence about representation.
