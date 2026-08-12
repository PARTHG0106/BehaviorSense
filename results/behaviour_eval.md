# Agent 3 behaviour-layer evaluation

Simulator v1.0.0, harness v1.0.0. 28 scenarios (24 anomalies, 4 controls).

> These are **simulator** results. They bound the detector's sensitivity given clean features; they are not estimates of clinical accuracy.

## Headline

| metric | value |
|---|---|
| recall (targeted alert kind) | 100% |
| median detection latency | 4 days |
| control false-alert rate | 5.07 per 100 resident-days |

## By anomaly kind

| anomaly | recall | median latency (days) | n |
|---|---|---|---|
| acute_mobility_drop | 100% | 1 | 4 |
| fall_event | 100% | 0 | 4 |
| gradual_mobility_decline | 100% | 20 | 4 |
| meal_skipping | 100% | 4 | 4 |
| medication_lapse | 100% | 4 | 4 |
| social_withdrawal | 100% | 10 | 4 |

## By persona (ordered by day-to-day variability)

| persona | noise cv | recall | median latency (days) | control FP / 100 days |
|---|---|---|---|---|
| regular_margaret | 0.05 | 100% | 4 | 2.10 |
| frail_dorothy | 0.10 | 100% | 4 | 3.50 |
| active_george | 0.15 | 100% | 4 | 7.69 |
| variable_arthur | 0.28 | 100% | 7 | 6.99 |

## Per-scenario detail

| scenario | detected | latency | any-alert latency | pre-onset FP | max severity |
|---|---|---|---|---|---|
| regular_margaret__gradual_decline | yes | 16 | 0 | 0 | urgent |
| regular_margaret__acute_drop | yes | 0 | 0 | 0 | urgent |
| regular_margaret__meal_skipping | yes | 4 | 4 | 0 | advisory |
| regular_margaret__social_withdrawal | yes | 7 | 7 | 0 | advisory |
| regular_margaret__medication_lapse | yes | 5 | 5 | 0 | urgent |
| regular_margaret__fall_with_long_lie | yes | 0 | 0 | 0 | emergency |
| regular_margaret__control_no_anomaly | n/a | - | - | 0 | advisory |
| frail_dorothy__gradual_decline | yes | 27 | 27 | 0 | urgent |
| frail_dorothy__acute_drop | yes | 0 | 0 | 0 | urgent |
| frail_dorothy__meal_skipping | yes | 4 | 4 | 0 | advisory |
| frail_dorothy__social_withdrawal | yes | 7 | 7 | 0 | advisory |
| frail_dorothy__medication_lapse | yes | 4 | 0 | 0 | urgent |
| frail_dorothy__fall_with_long_lie | yes | 0 | 0 | 0 | emergency |
| frail_dorothy__control_no_anomaly | n/a | - | - | 0 | advisory |
| variable_arthur__gradual_decline | yes | 16 | 3 | 0 | urgent |
| variable_arthur__acute_drop | yes | 2 | 1 | 0 | urgent |
| variable_arthur__meal_skipping | yes | 6 | 6 | 0 | advisory |
| variable_arthur__social_withdrawal | yes | 19 | 1 | 0 | advisory |
| variable_arthur__medication_lapse | yes | 8 | 8 | 0 | urgent |
| variable_arthur__fall_with_long_lie | yes | 0 | 0 | 0 | emergency |
| variable_arthur__control_no_anomaly | n/a | - | - | 0 | advisory |
| active_george__gradual_decline | yes | 23 | 22 | 0 | urgent |
| active_george__acute_drop | yes | 2 | 2 | 0 | urgent |
| active_george__meal_skipping | yes | 5 | 1 | 2 | advisory |
| active_george__social_withdrawal | yes | 12 | 10 | 0 | advisory |
| active_george__medication_lapse | yes | 3 | 3 | 0 | urgent |
| active_george__fall_with_long_lie | yes | 0 | 0 | 0 | emergency |
| active_george__control_no_anomaly | n/a | - | - | 0 | advisory |

## Alert volume by kind (all scenarios)

| alert kind | count | absolute-threshold rule |
|---|---|---|
| hydration_low | 332 | yes |
| meal_skipped | 277 |  |
| routine_deviation | 193 |  |
| sleep_disruption | 95 |  |
| medication_missed | 59 |  |
| mobility_decline | 49 |  |
| prolonged_inactivity | 21 | yes |
| social_isolation | 21 |  |
| fall_detected | 4 |  |
| fall_no_recovery | 4 |  |
