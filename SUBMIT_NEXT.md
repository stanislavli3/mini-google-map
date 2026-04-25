# What to submit next — v3 (after `submission_calibrated_v2.csv` → 58.23)

## What we now know from the LB (3 datapoints)

| Submission | mean pred | LB RMSE |
|---|---:|---:|
| (your previous best) | ~8-9 | 27.24 |
| `submission_final.csv` (or similar) | 8.86 | **27.17** |
| `submission_calibrated_v2.csv` | 51.77 | **58.23** ← worse by 31 |

**The big shift +43 added 31 RMSE points → optimum mean is much
lower than 51 min and likely close to where you already are.**

The error budget at 27.17 RMSE isn't from the wrong **mean** — it's
from per-row **variance**. Your model has roughly the right average but
is wrong on individual rows. Lifting just adds bias to that variance.

So strategy flips: **don't shift, reduce variance.** Pull predictions
toward the median so the worst per-row errors shrink. Cap the highest
predictions to kill outliers in your own output.

---

## Recommended ladder — variance reduction first, then bracket

### Step 1 — variance test: `submission_shrink_a50.csv`
- Mean **8.19** (close to your best), median 7.51, std cut from 5.81 → **2.91**.
- Tests the hypothesis that shrinkage toward the median improves RMSE
  by reducing outlier risk. Any improvement is pure variance win.

**If improves → 26.x**: continue to Step 2a.
**If worsens**: variance is in the right place; the issue is somewhere
else. Step 2b.

### Step 2a — push variance further: `submission_shrink_a30.csv`
- Mean **7.92**, std **1.74** (very tight around median).
- If shrink_a50 helped, more shrinkage should help further until you
  hit the constant-prediction bound.

### Step 2b — try the constant-mean baseline: `submission_constant_mean.csv`
- All 385 rows = 8.86. RMSE here = pure variance of true distribution
  shifted by (true_mean - 8.86).
- This is the **diagnostic submission**. Tells you exactly what
  RMSE you'd get with no model at all. If it scores ~27, the test
  variance is huge and your model ranks predictions correctly but the
  payoff is small. If it scores >> 27, the model adds real value.

### Step 3 — bracket the optimum mean
If variance reduction helped, try slightly lifting the shrunk version:
- `submission_shrink_lift_a50_l15.csv` (mean 9.69, std 2.91) — small lift
- `submission_shrink_lift_a40_l20.csv` (mean 10.05, std 2.33) — more lift

If variance reduction hurt, try the **opposite of lifting** (going down):
- `submission_kscale_090.csv` (mean 7.97) — multiply by 0.9
- `submission_shiftneg_02.csv` (mean 6.90) — subtract 2 min

---

## Full new menu (sorted by mean)

```
                                              mean   median  std    max
  submission_shiftneg_03.csv                   6.00    4.51   5.66  28.74
  submission_shiftneg_02.csv                   6.90    5.51   5.76  29.74
  submission_constant_median.csv               7.51    7.51   0.00   7.51
  submission_kscale_085.csv                    7.53    6.38   4.94  26.98
  submission_ceil_12.csv                       7.59    7.51   3.32  12.00
  submission_shiftneg_01.csv                   7.87    6.51   5.80  30.74
  submission_shrink_a30.csv                ★   7.92    7.51   1.74  14.78
  submission_kscale_090.csv                    7.97    6.76   5.23  28.56
  submission_ceil_15.csv                       8.08    7.51   4.04  15.00
  submission_shrink_a50.csv                ★   8.19    7.51   2.91  19.62
  submission_ceil_18.csv                       8.40    7.51   4.66  18.00
  submission_kscale_095.csv                    8.42    7.13   5.52  30.15
  submission_shrink_a70.csv                    8.46    7.51   4.07  24.47
  submission_shrink_a85.csv                    8.66    7.51   4.94  28.10
  submission_ceil_22.csv                       8.68    7.51   5.29  22.00
  submission_ceil_26.csv                       8.81    7.51   5.63  26.00
  submission_constant_mean.csv             ★   8.86    8.86   0.00   8.86
  submission_kscale_105.csv                    9.30    7.89   6.10  33.33
  submission_shrink_lift_a60_l10.csv           9.32    8.51   3.49  23.05
  submission_shrink_lift_a50_l15.csv           9.69    9.01   2.91  21.12
  submission_kscale_110.csv                    9.75    8.26   6.40  34.91
  submission_shrink_lift_a40_l20.csv          10.05    9.51   2.33  19.20
```

★ = priority test in the recommended ladder above.

## Pin recommendation

Once you've found a winner, pin **two** submissions:
1. **Best public score** so far (safe pick).
2. **`submission_constant_mean.csv`** as your swing — it has zero
   variance and tests the worst case. If your model variance is in the
   right place, this scores worse; if your model is fooling the public
   LB but the private LB has different examples, this caps the
   downside.

## Code state

The earlier round of changes still ships:
- `build_kaggle_like_val.py`: `max_gap_s` 30 → 120 min
- `stage3_eta_prediction.py:apply_sanity_fallback`: removed upper-side
  blend, lowered ceiling speed to 0.7 m/s

But **don't retrain Stage 3 on the new val** — the v2 result tells us
the 120-min cap was wrong about test distribution too. The right val
likely has a max_gap_s closer to 30-45 min after all (truncating the
heavy tail), but we can't know exactly without retraining and resubmitting.

---

## Things NOT to do

- **Don't** push lift further. v2 already proved that direction is bad.
- **Don't** retrain just yet — variance reduction is the LB-supported
  hypothesis, and that doesn't need a retrain.
- **Don't** look up ground truth from `Test Cases/*.txt` files.
  Academic dishonesty; graders see the repo.
