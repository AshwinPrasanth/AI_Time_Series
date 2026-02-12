# MiniRocket Reproducibility Plan (R1--R4 Complete Protocol)

This document provides a fully detailed, reproducibility-grade plan for
reproducing and auditing the MiniRocket (KDD 2021) paper.

------------------------------------------------------------------------

# I. Core Claims to Verify

From the paper:

## A. Accuracy Claims

* Marginally more accurate than Rocket (Fig 4, p6)
* Not significantly worse than SOTA (Fig 1, p1)
* Deterministic variant ≈ same accuracy (Fig 5, p7)

## B. Speed Claims

* 30× faster on average (p2)
* 75× faster on large datasets (p7–8)
* 8 minutes total vs 2h 2m (single core) (p2)

## C. Design Claims

* 84 kernels (length 9, 3 β weights) (p3–4)
* α = −1, β = 2
* Weights must sum to zero (p4)
* Bias from convolution output is critical (Fig 8, p8)
* PPV alone is sufficient (Fig 9, p8)
* 10k features saturate accuracy (Fig 10, p8)
* Max dilations per kernel = 32 default (Fig 11, p8)

We must explicitly test each of these.

---

# II. REPRODUCTION PLAN 

We structure this along reproducibility science barriers.

---

# 1️⃣ R1 – DESCRIPTION REPRODUCTION

Extract every parameter from the paper.

### Kernel

* Length = 9
* 3 weights = β
* 6 weights = α
* α = −1
* β = 2
* Must satisfy β = −2α (p4)
* 84 kernels (subset of 9 choose 3)

Verify:

* Number of combinations = C(9,3) = 84
* Sum of weights = 0
* Invariance to constant shift

Unit test:

* For random X and constant c:

  * Convolution(X, W)
  * Convolution(X+c, W)
  * Difference must be zero

---

### Bias

Default:

* Quantiles from convolution output of ONE randomly selected training example (p4)
* Low-discrepancy sequence for assigning quantiles
* Deterministic variant uses entire training set (p4)

Reproduce both.

Verify:

* Sampling bias from U(-1,1) reduces accuracy (Fig 8)

---

### Dilation

From p4:

$$
D = \{ \lfloor 2^0 \rfloor, \ldots, \lfloor 2^{\max} \rfloor \}
$$



Where:

$$
\text{max} = \log_2\left(\frac{l_{\text{input}} - 1}{l_{\text{kernel}} - 1}\right)
$$

* Uniform spacing in exponent
* Exponentially more features for small dilations
* Cap max dilations per kernel = 32

We must:

* Reproduce exact dilation schedule
* Validate count of features per dilation

---

### Padding

* Alternate padding for each kernel/dilation
* Standard zero padding
* Half padded, half not (p4)

Verify alternation logic.

---

### Features

* PPV only
* No max pooling
* Total features ≈ 9,996 (nearest multiple of 84 < 10,000)

---

### Classifier

From Appendix A (p10):

For logistic regression:

* Validation size = 2048
* Minibatch = 256
* LR = 1e-4
* Halve LR if no improvement after 50 updates
* Stop after 100 no-improvement updates
* Adam optimizer

Reproduce exactly.

---

# 2️⃣ R2 – CODE REPRODUCTION

DO NOT just run GitHub.

We must:

* Freeze commit hash
* Record:

  * Python version
  * NumPy version
  * Numba version
  * BLAS backend
  * CPU model
  * Single-thread enforcement

Disable:

* OpenMP multi-threading
* MKL multi-threading

Set:

```
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
```

---

### Unit Tests to Write

1. Kernel enumeration test
2. Weight sum = 0 test
3. Convolution equivalence:

   * naive convolution
   * optimized convolution
   * outputs identical
4. PPV(W) and PPV(-W) complementarity (p5)
5. Bias quantile correctness
6. Dilation coverage test
7. Feature count correctness

If any mismatch → document as reproducibility finding.

---

# 3️⃣ R3 – DATA REPRODUCTION

They use:

* 109 UCR datasets
* 30 resamples (same splits as Bagnall et al.)

Must:

* Obtain exact resample splits
* Hash dataset files
* Verify class distribution matches paper
* Confirm z-normalization not required (they say normalization unnecessary due to zero-sum kernels, p4)

Log:

* Dataset size
* Length
* Class imbalance
* Hash of raw file

---

# 4️⃣ R4 – EXPERIMENT REPRODUCTION (IMPORTANT)
---

## A. Accuracy Reproduction

Run:

* 30 resamples
* 109 datasets
* Same statistical protocol

Statistical testing:

* Friedman test
* Wilcoxon signed-rank
* Holm correction

Reproduce Fig 1 mean ranks.

Compare:

* MiniRocket vs Rocket
* MiniRocket vs deterministic variant

---

## B. Speed Reproduction

Measure:

* Transform time only
* Classifier time
* Total training time
* Single CPU core

Reproduce:

* Large datasets:

  * FruitFlies
  * InsectSound
  * MosquitoSound
    (p7–8)

Expect:

* 43× to 75× speedup

If not:

* Document hardware sensitivity
* Report cache effects
* Analyze BLAS differences

---

## C. Complexity Validation

They claim:

$$
O(k \cdot n \cdot l_{\text{input}})
$$

Test:

* Fix n, vary l_input
* Fix l_input, vary n
* Fix n, l_input, vary k

Plot:

* Log-log scaling
* Confirm linear slope

---

## D. Memory Validation

From p6:

* Stores 13 additional vectors
* ~52MB for length 1M

Test:

* Measure peak RAM
* Validate proportionality

---

# 5️⃣ SENSITIVITY REPRODUCTION

Reproduce Section 4.3 exactly:

### Kernel length:

7, 9, 11
Subsets 9{1}, 9{2}, 9{3}, 9{4}

### Bias:

Conv output vs Uniform

### Features:

84 → 99,960

### Dilations:

8, 16, 32, 64, 119

Plot mean ranks.

Match Fig 7–11 behavior.

---

# 6️⃣ DETERMINISM STUDY

They claim:

* Only stochastic element = bias example selection (p4)
* Deterministic variant negligible difference (Fig 5)

We test:

* 20 seeds
* Compute variance
* Report std deviation

If variance larger than reported → strong reproducibility finding.

---

# 7️⃣ GENERALISATION (FOR ECIR-LEVEL WORK)

Now go beyond UCR.

Test:

* Noise injection
* Missing values
* Multivariate extension
* Class imbalance
* Reduced training size
* Time series > UCR length

Check:

* Does speed advantage hold?
* Does accuracy collapse?

This transforms replication into a reproducibility paper.

---

# 8️⃣ ASSUMPTION AUDIT

Extract and test assumptions:

| Assumption                        | Test                         |
| --------------------------------- | ---------------------------- |
| Weight scale irrelevant           | Multiply weights by constant |
| Zero-sum ensures shift invariance | Add constant to input        |
| 10k features sufficient           | Increase to 50k              |
| PPV sufficient                    | Add max pooling              |
| Bias-from-conv critical           | Replace with uniform         |
| Dilation cap irrelevant           | Increase to 119              |
| Deterministic ≈ default           | Multi-seed test              |

Explicitly confirm or refute each.

---

# 9️⃣ ARTIFACT PACKAGE

To reach full R4 reproducibility:

Provide:

* Dockerfile
* Requirements.txt
* Dataset hashes
* Exact command scripts
* Seeds list
* Statistical test scripts
* Raw CSV results
* README with step-by-step reproduction
* Hardware description
* Energy consumption (optional but strong)

---

# 10️⃣ WHAT WOULD MAKE THIS STRONG SCIENTIFICALLY

If our results show:

* Confirmed accuracy + speed claims
* Low variance across seeds
* Speed sensitive to CPU architecture
* Accuracy drops under distribution shift
* Bias sampling more fragile on small datasets
* Feature saturation depends on series length

Then our paper becomes:

> A robustness and generalisation audit of MiniRocket

That is a real reproducibility contribution.

---

# Final Summary

To properly reproduce MiniRocket, we must:

1. Re-derive kernel math
2. Validate optimized convolution
3. Reproduce 109×30 evaluation
4. Reproduce statistical tests
5. Reproduce scaling experiments
6. Reproduce sensitivity analysis
7. Validate determinism
8. Audit assumptions
9. Control environment
10. Release full artifacts

Anything less is replication.

---

