# Causal Analysis of Attention Heads in GPT-2 Small

Mechanistic interpretability study of **Indirect Object Identification (IOI)** in GPT-2 Small using **activation patching** and **attention-head ablation**.

The project investigates where IOI behavior is encoded inside the model and which late-layer attention heads have measurable causal effects on the model's predictions.

---

## Overview

The main question is:

> **Which internal components of GPT-2 Small causally contribute to Indirect Object Identification?**

The analysis follows a coarse-to-fine approach:

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
Candidate head identification
     ↓
Attention-head ablation
````

The experiments use **1,000 synthetic IOI examples** and require no model training or fine-tuning.

---

## Main Results

### 1. Strong late-layer localization

Residual-stream activation patching showed very small effects in early layers and a sharp increase in the final layers.

| Layer | Mean Patch Effect |
| ----: | ----------------: |
|     7 |             0.003 |
|     8 |             0.109 |
|     9 |             0.265 |
|    10 |             0.528 |
|    11 |         **0.838** |

This motivated focusing the attention-head analysis on Layers 10 and 11.

### 2. Candidate attention heads

The strongest heads by mean activation-patching effect were:

| Head      |      Mean | Positive Fraction |
| --------- | --------: | ----------------: |
| **L10H0** | **0.101** |             98.5% |
| **L10H6** | **0.068** |             99.5% |
| **L10H1** | **0.056** |             92.0% |
| L10H10    |     0.050 |             90.9% |
| L11H1     |     0.036 |             86.2% |
| L11H3     |     0.032 |             95.6% |
| L11H11    |     0.032 |          **100%** |

### 3. Head ablation

Ablating candidate heads produced a different ranking:

| Head      | Mean Ablation Effect |
| --------- | -------------------: |
| **L11H0** |            **0.544** |
| **L10H0** |            **0.338** |
| **L10H6** |            **0.318** |
| L11H1     |                0.091 |
| L10H10    |                0.087 |

This difference is expected because activation patching and ablation answer different causal questions.

---

## Experimental Setup

### Model

**GPT-2 Small**

* 12 transformer layers
* 12 attention heads per layer
* No training or fine-tuning

### Dataset

A synthetic IOI dataset containing 1,000 controlled examples.

Example:

```text
Clean:
When Carol and Frank arrived at school, Frank showed the picture to

Clean answer:
Carol
```

The corrupted version reverses the roles:

```text
Corrupted:
When Frank and Carol arrived at school, Carol showed the picture to

Corrupted answer:
Frank
```

For every example:

```text
clean_correct       = A
clean_incorrect     = B

corrupted_correct   = B
corrupted_incorrect = A
```

---

## Behavioral Metric

All experiments use the same clean-vs-corrupted logit axis:

[
LD ={logit(A)}-{logit(B)}
]

Therefore:

```text
LD > 0    → preference for clean answer
LD < 0    → preference for corrupted answer
```

This fixed axis is important because the correct answer changes between the clean and corrupted prompts.

---

## Activation Patching

For residual-stream patching, the clean activation is inserted into the corrupted computation:

```text
Clean prompt
     │
     └── activation ───────┐
                           ↓
Corrupted prompt → Transformer → Prediction
                           ↑
                        patch
```

The normalized patching score is:

[
P =
\frac{
LD_{patched}-LD_{corrupt}
}{
LD_{clean}-LD_{corrupt}
}
]

Interpretation:

```text
P ≈ 0    → little recovery
P ≈ 0.5  → partial recovery
P ≈ 1    → strong recovery
P < 0    → movement away from clean behavior
```

Residual-stream interventions use:

```python
blocks.L.hook_resid_pre
```

and head-level interventions use:

```python
blocks.L.attn.hook_z
```

---

## Attention-Head Ablation

For head ablation, the selected attention head is set to zero at every token position:

```python
activation[:, :, head, :] = 0
```

The ablation effect is:

[
A =
LD_{ablated} - LD_{corrupt}
]

Because the metric is measured on the clean-vs-corrupted axis:

```text
A > 0 → ablation moves toward the clean answer
A < 0 → ablation moves toward the corrupted answer
A ≈ 0 → little change
```

Ablation should therefore not be interpreted as a simple "importance score"; its sign describes the direction in which behavior changes after removing the head.

---

## Repository Structure

---


## Interpretation

The main conclusion is:

> **IOI behavior in GPT-2 Small is strongly localized to late-layer representations and a relatively small subset of attention heads.**

The strongest patching candidates include:

```text
L10H0
L10H6
L10H1
L10H10
L11H1
L11H3
L11H11
```

However, these should be considered **candidate components**, not a complete reconstructed IOI circuit.

A high patching effect means that the clean activation contains information that helps restore the corrupted computation. It does not, by itself, establish what computation the head performs.

---

## Limitations

This project deliberately uses a small and controlled experimental setup.

* The dataset is synthetic.
* The corruption mechanism is simpler than the full IOI benchmark.
* Head-level analysis focuses on Layers 10 and 11.
* Initial head patching uses a fixed prediction position.
* The experiments identify candidate components but do not fully reconstruct information flow between heads.

---

## Next Steps

The natural continuation of this project is:

```text
Candidate heads
      ↓
Head × token-position patching
      ↓
Attention-pattern analysis
      ↓
Clean-prompt ablation
      ↓
Q / K / V analysis
      ↓
Pairwise head interventions
      ↓
Circuit hypothesis
```

The next goal is to determine **what the strongest heads are actually doing**, rather than only showing that they have a causal effect.

---

## References

* Wang et al. — *Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 Small*
* TransformerLens — mechanistic interpretability toolkit
* Elhage et al. — *A Mathematical Framework for Transformer Circuits*

```bibtex
@article{wang2022ioi,
  title={Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 Small},
  author={Wang, Kevin and Variengien, Alexandre and Conmy, Arthur and Shlegeris, Buck and Steinhardt, Jacob},
  year={2022},
  url={https://arxiv.org/abs/2211.00593}
}
```

