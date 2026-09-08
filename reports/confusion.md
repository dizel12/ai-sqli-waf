# Confusion Matrices

_Held-out test split, decision threshold 0.5._

## baseline

| | predicted malicious | predicted benign |
|---|---|---|
| **actual malicious** | 33 (TP) | 1 (FN) |
| **actual benign** | 1 (FP) | 33 (TN) |

## cnn

| | predicted malicious | predicted benign |
|---|---|---|
| **actual malicious** | 34 (TP) | 0 (FN) |
| **actual benign** | 0 (FP) | 34 (TN) |

## distilbert

| | predicted malicious | predicted benign |
|---|---|---|
| **actual malicious** | 34 (TP) | 0 (FN) |
| **actual benign** | 0 (FP) | 34 (TN) |
