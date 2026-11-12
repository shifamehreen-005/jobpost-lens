# Retrieval eval

Date range 2023-12-05T21:08:53+00:00 to 2024-04-20T00:26:43+00:00, no work_type filter, k=5.
Hit@k is the share of questions whose gold_id is in the top k.
MRR is the mean of 1/rank, or 0 when the gold_id is outside the top 5.

| questions | column | Hit@1 | Hit@5 | MRR |
| --- | --- | ---: | ---: | ---: |
| copied wording | emb_title | 0.650 | 0.850 | 0.717 |
| copied wording | emb_full | 1.000 | 1.000 | 1.000 |
| realistic | emb_title | 0.850 | 1.000 | 0.925 |
| realistic | emb_full | 1.000 | 1.000 | 1.000 |

## Missed by copied wording, emb_title

Gold id was not in the top 5.

- **3887477849** — die-cast factory role in Liberty writing control plans, running PPAP, and holding the plant to IATF rules
  - top ids: 3903830878, 3904953796, 3905334013, 3902948757, 3903496772
- **3900964417** — Sunnyvale opening designing GPU clusters that train and serve large language models
  - top ids: 3905868233, 3901939547, 3900942707, 3889262390, 3902948757
- **3904935358** — Maumelle factory job measuring newly molded medical parts and documenting the method before production
  - top ids: 3887477849, 3901939547, 3905868233, 3902790263, 3887873046

## Missed by copied wording, emb_full

Gold id was not in the top 5.

None.

## Missed by realistic, emb_title

Gold id was not in the top 5.

None.

## Missed by realistic, emb_full

Gold id was not in the top 5.

None.
