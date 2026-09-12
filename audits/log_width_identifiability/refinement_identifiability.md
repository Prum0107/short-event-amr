# Refinement identifiability

## Current update

The baseline decoder maintains normalized references and applies additive residuals in logit coordinates:

```text
z_l = ell(w_l)
z_{l+1} = z_l + Delta_{w,l}
w_{l+1} = sigmoid(z_{l+1})
```

The same update applies to center and width. Width also enters the temporal sine embedding and the modulation factor `sigmoid(ref_anchor_head(output_l))/w_l`, so a width trajectory affects attention before the final span loss.

## Proposed additive log update

An absolute physical log-width update would be

```text
u_l = log(w_l_sec/tau)
u_{l+1} = u_l + delta_{u,l}
w_{l+1,sec} = w_{l,sec} exp(delta_{u,l})
```

For a target transition, the required log update is independent of `D`:

| physical transition | linear seconds delta | additive normalized delta at `D=60` | additive normalized delta at `D=240` | multiplicative/log delta `log(target/current)` |
|---|---:|---:|---:|---:|
| 40 -> 1 s | -39 s | -0.650000 | -0.162500 | -3.688879 |
| 10 -> 1 s | -9 s | -0.150000 | -0.037500 | -2.302585 |
| 5 -> 1 s | -4 s | -0.066667 | -0.016667 | -1.609438 |
| 20 -> 10 s | -10 s | -0.166667 | -0.041667 | -0.693147 |

These are coordinate distances, not claims about the learned network's actual residuals. They show why a fixed additive normalized update expresses the same physical correction differently at different `D`, whereas a log ratio expresses the relative correction identically.

## Current sigmoid/logit update magnitudes

If the current normalized sigmoid/logit parameter is required to move directly from `p/D` to `g/D`, the exact logit displacement is

```text
Delta_z = ell(g/D) - ell(p/D)
```

For the same transitions:

| physical transition | `Delta_z` at `D=60` | `Delta_z` at `D=240` |
|---|---:|---:|
| 40 -> 1 s | -4.770685 | -3.867026 |
| 10 -> 1 s | -2.468100 | -2.340969 |
| 5 -> 1 s | -1.679642 | -1.626316 |
| 20 -> 10 s | -0.916291 | -0.737599 |

The exact values depend on the current state and the sigmoid parameterization. The important point is that they are not equal to the additive log-seconds values and still depend on `D` through the normalized endpoints.

## Can the representation change while refinement stays functionally equivalent?

Only with an explicit state-dependent reparameterization. If `u_l=log(D sigmoid(z_l)/tau)`, exact replay of the baseline physical trajectory requires

```text
u_{l+1} = log(D * sigmoid(z_l + Delta_{w,l}) / tau)
delta_{u,l} = log(sigmoid(z_l + Delta_{w,l}) / sigmoid(z_l))
```

This is not the generic additive rule `u_{l+1}=u_l+delta_{u,l}` with the old residual copied unchanged. Conversely, if the new update is chosen as `delta_u=log(w_{l+1}/w_l)` after computing the old baseline update, the trajectory is equivalent by construction, but no refinement mechanism has been intervened on; it is only a coordinate rewrite.

Therefore a simple log-width update is a refinement intervention. A deliberately exact induced update is a representation rewrite and cannot test whether multiplicative refinement helps.

## Refinement identifiability conclusion

Log width does not logically force multiplicative refinement; multiplicative refinement is a choice of update law in log coordinates. It is identifiable as a mechanism only when initialization, final coordinate, attention conversion, and loss are held fixed. The cleanest such pilot is Variant C in the normalized coordinate, where the update law changes but the baseline initialization and final geometry do not.
