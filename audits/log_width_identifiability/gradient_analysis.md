# Gradient analysis

## Derivation convention

Let `p=w_p_sec` and `g=w_g_sec`, and let `tau=1 s`. The tables below report the width-coordinate contribution. The implemented `F.l1_loss` averages center and width, so its width-only gradient is one half of the listed normalized-coordinate derivative when center and width are both present.

## Variant A: original normalized L1 through `u`

The reconstructed normalized widths are

```text
w_p = tau exp(u_p)/D = p/D
w_g = g/D
```

The width coordinate of the original normalized L1 is

```text
L_A(u_p;D,g) = |tau exp(u_p)/D - g/D|
```

Away from equality,

```text
dL_A/du_p = sign(p-g) * tau exp(u_p)/D
            = sign(p-g) * p/D
```

For the actual mean over `(center,width)`, multiply this by `1/2`; for the configured span-loss coefficient, multiply by `10` as well. The magnitude depends on `D` and on the current predicted physical width `p`; the target enters the sign and the normalized target comparison. Thus Variant A does not remove duration dependence. A fixed physical error is represented by a smaller normalized error in longer audio, and the log-coordinate gradient is still divided by `D`.

## Variant B: absolute log-width L1

The width loss is

```text
L_B(u_p;u_g) = |u_p-u_g| = |log(p/g)|
```

and, away from equality,

```text
dL_B/du_p = sign(u_p-u_g) = sign(p-g)
```

The width-log gradient has no explicit `D` dependence and no predicted-width magnitude factor. This is a loss-geometry change, not evidence that the log coordinate alone caused the behavior. If a smooth Huber variant were selected, its transition scale would be an additional tunable loss mechanism and would need to be declared before any result inspection; it is not introduced here.

## Variant C: log-space refinement only

The refinement variable is `v=log(w)` but the final normalized width and original losses are retained:

```text
w = exp(v)
L_C(v;D,g) = |exp(v)-g/D|
dL_C/dv = sign(exp(v)-g/D) * exp(v)
```

This is the same chain-rule form as Variant A with `tau exp(u)=D exp(v)`, so the final normalized L1 remains duration-dependent. The distinction is that C changes the layer update while preserving the final loss and initialization coordinate.

## Numerical short-event examples

For a 5-second prediction, the raw Variant A width gradient magnitude is `5/D`; for a 24-second prediction it is `24/D`. The implemented mean span L1 uses half of those values.

| predicted/GT seconds | `D` | normalized width L1 | implemented mean span L1 for a 4 s error | Variant A raw `|dL/du|` at prediction |
|---|---:|---:|---:|---:|
| 5 / 1 | 60 | 0.066667 | 0.033333 | 0.083333 |
| 5 / 1 | 240 | 0.016667 | 0.008333 | 0.020833 |
| 5 / 2 | 60 | 0.050000 | 0.025000 | 0.083333 |
| 5 / 2 | 240 | 0.012500 | 0.006250 | 0.020833 |
| 24 / 20 | 60 | 0.066667 | 0.033333 | 0.400000 |
| 24 / 20 | 240 | 0.016667 | 0.008333 | 0.100000 |

The `4 s` rows make clear that the normalized loss changes by exactly a factor of four when `D` changes from 60 to 240, while the corresponding absolute log error does not.

## What is and is not removed

- Variant A changes the gradient with respect to the internal variable but retains `1/D` in both the target mapping and the derivative.
- Variant B removes explicit `D` from the width log-loss for a fixed physical ratio, while changing the optimization objective.
- Variant C changes refinement geometry only if the initial/final normalized coordinate and losses are kept fixed; it does not remove duration dependence from final normalized regression or attention.
- GIoU is dimensionless and invariant to a common seconds-to-normalized scaling for a fixed physical prediction/target pair, but its derivative path changes when the internal coordinate changes. The full objective is therefore not identified by inspecting only its scalar GIoU value.
