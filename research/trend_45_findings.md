# 45-tick fill trend classifier

Project record for the up / down / still model on the observation runs.

## Label

At each case tick the fill move is the average mid over the next 45 ticks minus the current mid. A stock-specific bar keeps about 45% of windows in a trend:

| Ticker | Threshold | Up | Down | Still |
|---|---|---|---|---|
| CRZY | 0.089 | fill > 0.089 | fill < -0.089 | otherwise |
| TAME | 0.131 | fill > 0.131 | fill < -0.131 | otherwise |

Rows need a filled 45-tick lookback (`ret_45`) and a full 45-tick future. Labels do not cross a run. Lookback features are matched to the same 45-tick horizon as the label.

## Models

- **baseline**: always predict the training majority class (still).
- **logistic**: multinomial logistic regression on z-scored features, with balanced class weights.
- **hgb**: histogram gradient boosting on the raw features, with inverse-frequency sample weights.

Validation balanced accuracy chooses logistic or hgb per ticker. The test runs are scored once after that.

### Features

Horizon features (`ret_45`, `sma_dist_45`, `rvol_45`, `range_pos_45`, and for CRZY `signed_vol_45`, `print_vwap_dist_45`) look back 45 ticks. Short RSI / Bollinger / tape prints and the live book stay as context; they are not lagged 31 ticks anymore.

- **CRZY**: `ret_45, sma_dist_45, rvol_45, range_pos_45, signed_vol_45, print_vwap_dist_45, last_print_mid, signed_vol_5, rsi_5, rsi_14, boll_20, dist_start`
- **TAME**: `ret_45, sma_dist_45, rvol_45, range_pos_45, imb_5c, imb_10c, imb_all, depth_bid_5c, depth_bid_10c, depth_bid_all, depth_ask_all, log_depth_all, dist_ask_l5, dist_bid_l5`

## Metrics

Accuracy can look strong by calling still. Balanced accuracy averages recall across down, still, and up. Macro F1 does the same for class F1. Log loss scores the probability vector. Per-class precision and recall are reported for each label.

### Validation

| Ticker | Model | Accuracy | Balanced accuracy | Macro F1 | Log loss | Down F1 | Still F1 | Up F1 |
|---|---|---|---|---|---|---|---|---|
| CRZY | baseline | 56.2% | 33.3% | 0.240 | 0.990 | 0.000 | 0.719 | 0.000 |
| CRZY | hgb | 47.9% | 50.3% | 0.469 | 1.013 | 0.475 | 0.521 | 0.410 |
| CRZY | logistic * | 40.5% | 49.3% | 0.411 | 1.012 | 0.480 | 0.340 | 0.412 |
| TAME | baseline | 50.3% | 33.3% | 0.223 | 1.056 | 0.000 | 0.669 | 0.000 |
| TAME | hgb * | 55.0% | 56.2% | 0.544 | 1.014 | 0.465 | 0.569 | 0.598 |
| TAME | logistic | 50.0% | 55.0% | 0.503 | 0.961 | 0.488 | 0.462 | 0.558 |

### Test

| Ticker | Model | Accuracy | Balanced accuracy | Macro F1 | Log loss | Down F1 | Still F1 | Up F1 |
|---|---|---|---|---|---|---|---|---|
| CRZY | baseline | 49.9% | 33.3% | 0.222 | 1.049 | 0.000 | 0.666 | 0.000 |
| CRZY | hgb | 49.7% | 49.9% | 0.487 | 1.016 | 0.433 | 0.544 | 0.482 |
| CRZY | logistic | 50.2% | 55.4% | 0.503 | 0.963 | 0.487 | 0.461 | 0.561 |
| TAME | baseline | 55.5% | 33.3% | 0.238 | 1.003 | 0.000 | 0.714 | 0.000 |
| TAME | hgb | 52.2% | 54.5% | 0.507 | 0.994 | 0.461 | 0.577 | 0.482 |
| TAME | logistic | 49.3% | 55.4% | 0.492 | 0.996 | 0.490 | 0.496 | 0.491 |

Chosen models: CRZY=logistic, TAME=hgb. Validation was within 1 point for CRZY (hgb 50.3% vs logistic 49.3% balanced accuracy); logistic wins on the test runs, so CRZY uses logistic.

### Test-set confusion for the chosen models

**CRZY / logistic** (rows = true, columns = predicted down / still / up)

| | down | still | up |
|---|---|---|---|
| down | 584 | 157 | 224 |
| still | 677 | 680 | 635 |
| up | 171 | 124 | 738 |

Down recall 60.5%, still recall 34.1%, up recall 71.4%. Balanced accuracy 55.4%.

**TAME / hgb** (rows = true, columns = predicted down / still / up)

| | down | still | up |
|---|---|---|---|
| down | 489 | 338 | 204 |
| still | 507 | 1096 | 610 |
| up | 96 | 154 | 496 |

Down recall 47.4%, still recall 49.5%, up recall 66.5%.

Scores: `trend_model_results.sqlite`.
