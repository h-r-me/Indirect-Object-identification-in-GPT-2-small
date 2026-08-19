# Causal Analysis of Attention Heads in GPT-2 Small

Mechanistic interpretability study of **Indirect Object Identification (IOI)** in GPT-2 Small using **activation patching** and **attention-head ablation**.

The project asks:

> **Which internal components of GPT-2 Small causally contribute to IOI behavior?**

---

## Approach

The analysis follows a coarse-to-fine causal analysis:

```text
IOI behavior
     ↓
Clean / corrupted prompts
     ↓
Residual-stream activation patching
     ↓
Layer localization
     ↓
Attention-head patching
     ↓
Head ablation
````

All experiments use **GPT-2 Small** and **1,000 synthetic IOI examples**, with no model training or fine-tuning.

---

## Key Results

### Late-layer localization

Residual-stream activation patching shows that the causal effect is concentrated in the final layers:

|  Layer | Mean Patch Effect |
| -----: | ----------------: |
|      7 |             0.003 |
|      8 |             0.109 |
|      9 |             0.265 |
|     10 |             0.528 |
| **11** |         **0.838** |

This motivated a focused attention-head analysis of Layers 10 and 11.

### Candidate attention heads

The strongest candidates by mean patch effect include:

| Head      | Mean Patch Effect | Positive Fraction |
| --------- | ----------------: | ----------------: |
| **L10H0** |         **0.101** |             98.5% |
| **L10H6** |         **0.068** |             99.5% |
| **L10H1** |         **0.056** |             92.0% |
| L10H10    |             0.050 |             90.9% |
| L11H3     |             0.032 |             95.6% |
| L11H11    |             0.032 |              100% |

### Ablation

Ablation produced a different ranking, with the largest average effects observed for:

| Head      | Mean Ablation Effect |
| --------- | -------------------: |
| **L11H0** |            **0.544** |
| **L10H0** |            **0.338** |
| **L10H6** |            **0.318** |

The difference between patching and ablation highlights that these interventions probe different aspects of the model's computation.

---

## Model & Dataset

**Model:** GPT-2 Small
**Framework:** TransformerLens
**Dataset:** 1,000 synthetic IOI examples

Example:

```text
Clean:
When Carol and Frank arrived at school, Frank showed the picture to
→ Carol

Corrupted:
When Frank and Carol arrived at school, Carol showed the picture to
→ Frank
```

The clean and corrupted prompts reverse the roles of the two names, providing a controlled causal test.

---



## Main Takeaway

The experiments provide evidence that IOI behavior in GPT-2 Small is strongly concentrated in **late-layer representations and a relatively small subset of attention heads**.

The current results identify candidate components, but do not yet reconstruct the complete IOI circuit.

