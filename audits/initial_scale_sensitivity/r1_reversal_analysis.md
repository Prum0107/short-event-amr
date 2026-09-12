# R1 layer2-to-final reversal analysis

The existing saved R1 and baseline well-centered trajectories were compared on common `(qid, slot)` keys. A reversal means R1 has lower absolute log-width residual than baseline at decoder layer 1 and layer 2, but a higher residual at final span.

| GT bin | common proposals | reversal proposals | reversal fraction | median R1 layer2 width/GT | median R1 final width/GT | median center movement (s) | baseline probe final width/GT range | reversal final inside range |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| 0-2s | 11 | 5 | 0.45454545454545453 | 4.9035186767578125 | 5.4274749755859375 | 7.440681457519531 | [1.6598325818777084, 254.79633808135986] | 0.8 |
| 2-5s | 89 | 33 | 0.3707865168539326 | 1.9238128662109375 | 2.9941635131835938 | 9.467453002929688 | [0.23033127716432014, 125.21670162677765] | 1.0 |
| 5-10s | 82 | 21 | 0.25609756097560976 | 1.037841796875 | 1.3455187479654949 | 8.83053207397461 | [0.09820220537949353, 56.90606585741043] | 1.0 |
| 10-20s | 78 | 6 | 0.07692307692307693 | 0.7426588058471679 | 1.3960760934012275 | 12.137096405029297 | [0.04392266416778931, 26.51735706762834] | 1.0 |
| 20s+ | 53 | 3 | 0.05660377358490566 | 0.47672835640285327 | 0.6831558890964674 | 30.707595825195312 | [0.017221856396645308, 11.043904066085815] | 1.0 |

The saved trajectory source is used without retraining R1. Overlap with the baseline probe range is evidence of consistency with a shared final-scale regime, not proof of an attractor mechanism.
