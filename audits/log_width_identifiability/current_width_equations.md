# Current QD-DETR width equations and implementation map

## Scope and evidence

This is a static identifiability audit of the matched official baseline/M1 implementation. It does not train, run inference, modify QD-DETR, or propose a performance method. The source of truth is the remote worktree `/private/research-artifact` at the time of this audit.

Relevant files and exact code paths are:

- `src/qd_detr.py:102`: `query_embed = nn.Embedding(num_queries, 2)`; each query stores two unconstrained values.
- `src/qd_detr.py:166-180`: M1 may replace only the center reference; the width component remains the learned query width.
- `src/qd_detr.py:181-187`: the returned reference and `span_embed(hs)` are added in inverse-sigmoid coordinates and mapped to normalized `(center,width)` by a sigmoid.
- `src/qd_detr_transformer.py:26-30`: inverse sigmoid is a clamped logit with `eps=1e-3`.
- `src/qd_detr_transformer.py:222-243`: decoder reference-point head, zero-initialized final box-regression layer, and temporal attention anchor head.
- `src/qd_detr_transformer.py:260-314`: reference initialization, temporal sine embedding, modulated attention, and iterative reference update.
- `src/qd_detr_transformer.py:328-338`: with two decoder layers, returned intermediate references are the initial reference and each non-final update; the final QD-DETR span head then supplies the output used by the criterion.
- `src/span_utils.py:61-77, 113-153`: `(center,width)` to `(start,end)` conversion and generalized temporal IoU.
- `src/matcher.py:62-100`: Hungarian matching uses normalized L1 span cost, negative generalized IoU, and class cost.
- `src/qd_detr.py:268-290`: matched span loss uses elementwise normalized L1 and generalized IoU.
- `src/qd_detr.py:535-565` and `config.yml`: the baseline coefficients are `set_cost_span=10`, `set_cost_giou=1`, `span_loss_coef=10`, `giou_loss_coef=1`, `span_loss_type=l1`, `dec_layers=2`, `num_queries=10`.

The implementation comments state that spans are normalized `(center_x,w)` coordinates relative to each audio. The code contains no absolute-seconds width variable.

## Variables

Let `D` be the valid audio duration in seconds, `c` the normalized center, and `w` the normalized width. The physical interval is

```text
c_sec = D c
w_sec = D w
[start_sec,end_sec] = [D(c - w/2), D(c + w/2)]
```

The learned query is `q=(q_c,q_w)`. Define the clamped implementation logit

```text
ell(x) = log(clamp(x,eps,1) / clamp(1-x,eps,1)),  eps=1e-3
sigmoid(z) = 1/(1+exp(-z))
```

## Initialization

For ordinary baseline decoding, `query_embed.weight` is repeated across the batch as `refpoint_embed`. The decoder applies

```text
r_0 = sigmoid(q) = (c_0,w_0)
```

where `w_0` is a static normalized width for a query slot. In M1, selected local center locations replace only the center logit:

```text
c_0 = (selected_index + 0.5) / valid_local_length
w_0 = sigmoid(q_w)
```

Thus M1 does not make width initialization audio-dependent; its width remains the learned normalized query width.

The checkpoint audit found the following learned normalized widths. They are listed to distinguish the invariant mechanism from exact tensor equality between independently trained checkpoints.

| slot | baseline `sigmoid(q_w)` | M1 `sigmoid(q_w)` |
|---:|---:|---:|
| 0 | 0.179809 | 0.192969 |
| 1 | 0.310817 | 0.336583 |
| 2 | 0.592502 | 0.574509 |
| 3 | 0.444840 | 0.470063 |
| 4 | 0.085511 | 0.087444 |
| 5 | 0.477918 | 0.476928 |
| 6 | 0.830323 | 0.849326 |
| 7 | 0.843327 | 0.838832 |
| 8 | 0.546285 | 0.525588 |
| 9 | 0.675818 | 0.652648 |

For example, at `D=240 s`, the baseline slots correspond to approximately 43.154, 74.596, 142.200, 106.762, 20.523, 114.701, 199.278, 202.398, 131.108, and 162.196 seconds. A short physical event therefore begins far below most static physical initial widths while its normalized target is additionally divided by `D`.

## Decoder refinement and temporal attention

At decoder layer `l`, the current normalized reference is `r_l=(c_l,w_l)`. The code creates a sine embedding of both components:

```text
phi(c_l,w_l) = concat(sine_embedding(2*pi*c_l),
                      sine_embedding(2*pi*w_l))
```

This width component is used by the reference-point head and by the temporal attention modulation

```text
query_sine_embed <- query_sine_embed *
                    (sigmoid(ref_anchor_head(output_l)) / w_l)
```

The width is therefore not only an output coordinate; it is also an attention-conditioning value.

The iterative box update is performed in logit coordinates:

```text
z_l = ell(r_l)
Delta_l = bbox_embed(output_l)
z_{l+1} = z_l + Delta_l
r_{l+1} = sigmoid(z_{l+1})
```

The last linear layer of `bbox_embed` is zero-initialized, so the initial update is zero at initialization of a newly constructed model. The trained model can of course produce nonzero updates.

The default configuration has two decoder layers. With intermediate references enabled, the decoder returns the initial reference and the non-final updated reference. QD-DETR then computes, per returned decoder state,

```text
y_l = sigmoid(span_embed(hs_l) + ell(r_l))
```

and uses the last `y_l` as `pred_spans`. This is a second residual span head on top of the decoder reference, not a direct physical-seconds prediction.

## Final geometry, loss, and matching

For a predicted normalized span `p=(c_p,w_p)` and target `g=(c_g,w_g)`, the criterion receives normalized coordinates. The L1 term is

```text
L_span = mean(|c_p-c_g|, |w_p-w_g|)
```

and the GIoU term is

```text
L_giou = 1 - GIoU([c_p-w_p/2,c_p+w_p/2],
                   [c_g-w_g/2,c_g+w_g/2])
```

The weighted span objective contains `10 L_span + 1 L_giou`; the full criterion also contains class, saliency, and any auxiliary decoder losses. Hungarian matching uses

```text
C = 10 * ||p-g||_1 - GIoU(p,g) + 4 * class_cost
```

for the configured L1 span, GIoU, and class coefficients.

If centers are equal and one interval contains the other, temporal GIoU equals IoU. Therefore width GIoU is ratio-sensitive in this special case, while normalized L1 width sensitivity is absolute-in-normalized-coordinate and scales as `1/D` for a fixed seconds error.

## Seconds conversion and identifiability consequence

The code path has one explicit width coordinate: normalized `w`. Seconds appear only when a normalized span is interpreted as an interval relative to `D`. A proposed `u=log(w_sec/1 s)` can be inserted in at least three non-equivalent ways:

1. store `u` but immediately map `w=exp(u)/D` before the existing attention, final geometry, matching, and loss;
2. store `u`, map to seconds, and also optimize an absolute log-width loss;
3. keep the baseline normalized final variable and change only the layer-to-layer update law.

These choices change different paths. Equality of the output dimension or equality of final seconds does not make them the same intervention. The next documents make the coupling explicit.
