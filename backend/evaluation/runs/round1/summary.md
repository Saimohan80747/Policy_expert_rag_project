# Evaluation summary

## Overall

| Mode | n | Accuracy | Answerable acc. | Wrong but confident | Abstain correct | Evidence hit | Graph non-empty | p50 s | p95 s | LLM calls/q | Errors |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| vanilla | 31 | 65% | 65% | 13% | - | 77% | - | 15.3 | 22.2 | 2.0 | 0 |
| graph | 31 | 81% | 81% | 3% | - | 0% | 25/31 | 19.4 | 33.3 | 3.1 | 0 |
| hybrid | 30 | 87% | 87% | 10% | - | 73% | 24/30 | 16.6 | 25.0 | 3.3 | 0 |
| router | 30 | 90% | 90% | 7% | - | 37% | 16/18 | 12.6 | 25.9 | 2.7 | 0 |

## Accuracy by category

| Category | n | vanilla | graph | hybrid | router |
| --- | --- | --- | --- | --- | --- |
| list | 2 | 100% | 100% | 100% | 100% |
| multi_hop | 8 | 0% | 88% | 86% | 86% |
| needle | 1 | 0% | 0% | 0% | 0% |
| relation_lookup | 13 | 85% | 77% | 92% | 92% |
| single_fact | 7 | 100% | 86% | 86% | 100% |

## Accuracy by reasoning depth (hops)

| Hops | n | vanilla | graph | hybrid | router |
| --- | --- | --- | --- | --- | --- |
| 1 | 23 | 87% | 78% | 87% | 91% |
| 3 | 1 | 0% | 100% | 100% | 100% |
| 4 | 5 | 0% | 100% | 100% | 100% |
| 5 | 2 | 0% | 50% | 50% | 50% |

## Router decisions

graph: 16, vanilla: 14
