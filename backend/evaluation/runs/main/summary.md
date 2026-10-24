# Evaluation summary

## Overall

| Mode | n | Accuracy | Answerable acc. | Wrong but confident | Abstain correct | Evidence hit | Graph non-empty | p50 s | p95 s | LLM calls/q | Errors |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| vanilla | 54 | 65% | 64% | 12% | 75% | 74% | - | 8.7 | 15.2 | 2.0 | 0 |
| graph | 54 | 72% | 70% | 8% | 100% | 0% | 39/54 | 9.3 | 20.3 | 3.0 | 0 |
| hybrid | 54 | 81% | 80% | 14% | 100% | 72% | 39/54 | 9.5 | 15.8 | 3.2 | 0 |
| router | 54 | 80% | 78% | 14% | 100% | 38% | 27/36 | 7.5 | 16.0 | 2.9 | 0 |

## Accuracy by category

| Category | n | vanilla | graph | hybrid | router |
| --- | --- | --- | --- | --- | --- |
| aggregation | 3 | 67% | 33% | 67% | 33% |
| disambiguation | 4 | 100% | 100% | 100% | 100% |
| list | 2 | 100% | 100% | 100% | 100% |
| multi_hop | 11 | 0% | 45% | 45% | 45% |
| needle | 1 | 0% | 0% | 0% | 0% |
| relation_lookup | 13 | 92% | 77% | 100% | 100% |
| single_fact | 7 | 86% | 57% | 86% | 86% |
| temporal | 9 | 67% | 100% | 89% | 89% |
| unanswerable | 4 | 75% | 100% | 100% | 100% |

## Accuracy by reasoning depth (hops)

| Hops | n | vanilla | graph | hybrid | router |
| --- | --- | --- | --- | --- | --- |
| 1 | 38 | 87% | 76% | 89% | 87% |
| 2 | 5 | 40% | 100% | 100% | 100% |
| 3 | 2 | 0% | 100% | 100% | 100% |
| 4 | 7 | 0% | 43% | 43% | 43% |
| 5 | 2 | 0% | 0% | 0% | 0% |

## Router decisions

graph: 25, hybrid: 2, vanilla: 27
