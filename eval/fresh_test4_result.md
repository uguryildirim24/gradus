# fresh_test4: the cycle-3 pick, scored once

Pick B+H''_without_work_mode|decider (cut-offs t_lo 0.2731045004103801, t_hi 0.5890379172624224); base comparator base+H''_without_work_mode|decider (t_lo 0.22252236411819226, t_hi 0.4949646267453026).
Two-way bootstrap (postings, then rows within them), 2000 draws, seed 20261021, one draw set per slice; 95% percentile intervals. Keep recall is shown with found / gold Keeps (amendment 4c).

## Pass bar (amendment 2, on the eligibility-pass slice)

PASS (n=513, postings=262, gold {'Skip': 388, 'Keep': 36, 'Explore': 89})

- B+H''_without_work_mode|decider minus rules_full: macro-F1 0.192 [0.079, 0.286], lower bound 0.078682 -> beaten (accuracy diff, reported only: 0.019 [-0.032, 0.070])
- B+H''_without_work_mode|decider minus base+H''_without_work_mode|decider: macro-F1 0.217 [0.124, 0.300], lower bound 0.123601 -> beaten (accuracy diff, reported only: 0.226 [0.163, 0.289])

## Agreement subset (labelers A and B agree; reported, not for pass)

- agreed_eligibility_pass (n=455, gold {'Skip': 352, 'Keep': 29, 'Explore': 74}): B+H''_without_work_mode|decider minus rules_full: 0.242 [0.124, 0.348], lower bound 0.124487 -> beaten
- agreed_eligibility_pass (n=455, gold {'Skip': 352, 'Keep': 29, 'Explore': 74}): B+H''_without_work_mode|decider minus base+H''_without_work_mode|decider: 0.252 [0.154, 0.340], lower bound 0.154212 -> beaten
- agreed_only (n=1051, gold {'Skip': 919, 'Keep': 34, 'Explore': 98}): B+H''_without_work_mode|decider minus rules_full: 0.250 [0.132, 0.354], lower bound 0.131859 -> beaten
- agreed_only (n=1051, gold {'Skip': 919, 'Keep': 34, 'Explore': 98}): B+H''_without_work_mode|decider minus base+H''_without_work_mode|decider: 0.216 [0.130, 0.298], lower bound 0.130018 -> beaten

## all (n=1139, postings=290, gold {'Skip': 968, 'Explore': 128, 'Keep': 43})

| system | macro-F1 | accuracy | Keep recall (found/gold) | Explore recall | Skip recall | Keep precision | unsafe Keeps |
|---|---|---|---|---|---|---|---|
| B+H''_without_work_mode|decider | 0.679 [0.599, 0.747] | 0.877 [0.849, 0.904] | 0.651 (28/43) | 0.4922 | 0.938 | 0.549 (28/51) | 23 |
| h+H' | 0.582 [0.505, 0.653] | 0.856 [0.826, 0.884] | 0.349 (15/43) | 0.4141 | 0.937 | 0.4167 (15/36) | 21 |
| rules_full | 0.456 [0.390, 0.526] | 0.860 [0.829, 0.890] | 0.070 (3/43) | 0.2422 | 0.9773 | 0.3 (3/10) | 7 |
| base+H''_without_work_mode|decider | 0.490 [0.435, 0.549] | 0.734 [0.698, 0.772] | 0.558 (24/43) | 0.4062 | 0.7851 | 0.2222 (24/108) | 84 |
| labeler ceiling (A vs B) | | 0.923 (kappa 0.7304) | | | | | |

## eligibility_pass (n=513, postings=262, gold {'Skip': 388, 'Keep': 36, 'Explore': 89})

| system | macro-F1 | accuracy | Keep recall (found/gold) | Explore recall | Skip recall | Keep precision | unsafe Keeps |
|---|---|---|---|---|---|---|---|
| B+H''_without_work_mode|decider | 0.676 [0.592, 0.745] | 0.813 [0.767, 0.855] | 0.667 (24/36) | 0.5281 | 0.8918 | 0.5455 (24/44) | 20 |
| h+H' | 0.606 [0.523, 0.682] | 0.774 [0.725, 0.822] | 0.417 (15/36) | 0.5955 | 0.8479 | 0.4167 (15/36) | 21 |
| rules_full | 0.484 [0.409, 0.564] | 0.793 [0.747, 0.841] | 0.083 (3/36) | 0.3483 | 0.9613 | 0.3 (3/10) | 7 |
| base+H''_without_work_mode|decider | 0.460 [0.395, 0.523] | 0.587 [0.529, 0.640] | 0.556 (20/36) | 0.4157 | 0.6289 | 0.2469 (20/81) | 61 |
| labeler ceiling (A vs B) | | 0.887 (kappa 0.7369) | | | | | |

## agreed_only (n=1051, postings=290, gold {'Skip': 919, 'Keep': 34, 'Explore': 98})

| system | macro-F1 | accuracy | Keep recall (found/gold) | Explore recall | Skip recall | Keep precision | unsafe Keeps |
|---|---|---|---|---|---|---|---|
| B+H''_without_work_mode|decider | 0.727 [0.641, 0.796] | 0.911 [0.887, 0.934] | 0.706 (24/34) | 0.5612 | 0.9565 | 0.6 (24/40) | 16 |
| h+H' | 0.615 [0.524, 0.696] | 0.881 [0.854, 0.907] | 0.412 (14/34) | 0.4694 | 0.9423 | 0.4667 (14/30) | 16 |
| rules_full | 0.477 [0.402, 0.559] | 0.883 [0.854, 0.910] | 0.088 (3/34) | 0.2755 | 0.9771 | 0.3 (3/10) | 7 |
| base+H''_without_work_mode|decider | 0.511 [0.446, 0.572] | 0.770 [0.735, 0.805] | 0.618 (21/34) | 0.4184 | 0.8128 | 0.2625 (21/80) | 59 |
| labeler ceiling (A vs B) | | 1.000 (kappa 1.0) | | | | | |

## agreed_eligibility_pass (n=455, postings=245, gold {'Skip': 352, 'Keep': 29, 'Explore': 74})

| system | macro-F1 | accuracy | Keep recall (found/gold) | Explore recall | Skip recall | Keep precision | unsafe Keeps |
|---|---|---|---|---|---|---|---|
| B+H''_without_work_mode|decider | 0.741 [0.655, 0.811] | 0.862 [0.818, 0.900] | 0.759 (22/29) | 0.5811 | 0.929 | 0.6111 (22/36) | 14 |
| h+H' | 0.633 [0.544, 0.707] | 0.793 [0.744, 0.839] | 0.483 (14/29) | 0.6216 | 0.8551 | 0.4667 (14/30) | 16 |
| rules_full | 0.499 [0.415, 0.586] | 0.809 [0.761, 0.855] | 0.103 (3/29) | 0.3649 | 0.9602 | 0.3 (3/10) | 7 |
| base+H''_without_work_mode|decider | 0.489 [0.416, 0.554] | 0.626 [0.569, 0.682] | 0.621 (18/29) | 0.4189 | 0.6705 | 0.2903 (18/62) | 44 |
| labeler ceiling (A vs B) | | 1.000 (kappa 1.0) | | | | | |

## New-employer slice

employer_new_vs_fresh_tests_1_3: n=20, postings=5, employers ['Akebia Therapeutics'], eligibility-pass rows 9, gold {'Skip': 20}. Too small to carry the bar: reported, not for pass (fresh test 4 is new at the posting level, not the employer level; eval/fresh_test4_manifest.json employers_vs_fresh_tests_1_3).

- B+H''_without_work_mode|decider: macro-F1 0.325 [0.296, 0.333]
- h+H': macro-F1 0.316 [0.275, 0.333]
- rules_full: macro-F1 0.333 [0.333, 0.333]
- base+H''_without_work_mode|decider: macro-F1 0.263 [0.191, 0.306]

Sealed-design eligibility-pass slice (sensitivity, not for pass; n=511, gold {'Skip': 388, 'Keep': 36, 'Explore': 87}):

- B+H''_without_work_mode|decider minus rules_full: 0.194 [0.084, 0.289], lower bound 0.083946 -> would be beaten
- B+H''_without_work_mode|decider minus base+H''_without_work_mode|decider: 0.211 [0.121, 0.296], lower bound 0.121368 -> would be beaten

Eligibility pass recomputed: 513 rows; sealed design: 511; moved out 0 ({}), moved in 2.
Rows with no extracted card: 4 (gold {'Skip': 4}); eval/fresh_test4_eval_inputs.json no_card_rows_policy: no rule verdict (no H'/H'' override), rules_full Explore, outside the eligibility-pass slice.

## Slices (reported, not for pass)

| slice | n | gold Keeps | B+H''_without_work_mode|decider | rules_full | base+H''_without_work_mode|decider | h+H' | pick - rules_full | pick - base |
|---|---|---|---|---|---|---|---|---|
| all | 1139 | 43 | 0.679 [0.599, 0.747] | 0.456 [0.390, 0.526] | 0.490 [0.435, 0.549] | 0.582 [0.505, 0.653] | 0.222 [0.114, 0.313] | 0.188 [0.104, 0.262] |
| eligibility_pass | 513 | 36 | 0.676 [0.592, 0.745] | 0.484 [0.409, 0.564] | 0.460 [0.395, 0.523] | 0.606 [0.523, 0.682] | 0.192 [0.079, 0.286] | 0.217 [0.124, 0.300] |
| teaser | 24 | 0 | 0.576 [0.420, 0.667] | 0.472 [0.263, 0.627] | 0.310 [0.095, 0.527] | 0.507 [0.347, 0.625] | 0.104 [-0.077, 0.297] | 0.265 [0.030, 0.492] |
| empty_body | 24 | 0 | 0.576 [0.420, 0.667] | 0.472 [0.263, 0.627] | 0.310 [0.095, 0.527] | 0.507 [0.347, 0.625] | 0.104 [-0.077, 0.297] | 0.265 [0.030, 0.492] |
| teaser_or_empty | 24 | 0 | 0.576 [0.420, 0.667] | 0.472 [0.263, 0.627] | 0.310 [0.095, 0.527] | 0.507 [0.347, 0.625] | 0.104 [-0.077, 0.297] | 0.265 [0.030, 0.492] |
| blank_profile_field | 677 | 29 | 0.694 [0.604, 0.772] | 0.464 [0.393, 0.545] | 0.499 [0.431, 0.565] | 0.593 [0.507, 0.670] | 0.231 [0.112, 0.335] | 0.195 [0.109, 0.279] |
| lives_in_not_given | 25 | 0 | 0.369 [0.260, 0.531] | 0.296 [0.260, 0.327] | 0.313 [0.209, 0.444] | 0.468 [0.288, 0.611] | 0.073 [-0.038, 0.228] | 0.056 [-0.152, 0.269] |
| agreed_only | 1051 | 34 | 0.727 [0.641, 0.796] | 0.477 [0.402, 0.559] | 0.511 [0.446, 0.572] | 0.615 [0.524, 0.696] | 0.250 [0.132, 0.354] | 0.216 [0.130, 0.298] |
| no_exact_title_in_resume | 1139 | 43 | 0.679 [0.599, 0.747] | 0.456 [0.390, 0.526] | 0.490 [0.435, 0.549] | 0.582 [0.505, 0.653] | 0.222 [0.114, 0.313] | 0.188 [0.104, 0.262] |
| exact_title_in_resume | 0 | | | | | | | |
| kind:data | 133 | 7 | 0.572 [0.348, 0.764] | 0.323 [0.273, 0.411] | 0.433 [0.287, 0.561] | 0.474 [0.303, 0.659] | 0.249 [0.005, 0.455] | 0.139 [-0.114, 0.403] |
| kind:health_nursing | 182 | 16 | 0.661 [0.489, 0.789] | 0.351 [0.294, 0.439] | 0.469 [0.354, 0.573] | 0.584 [0.419, 0.721] | 0.310 [0.084, 0.473] | 0.192 [0.007, 0.347] |
| kind:internship | 58 | 6 | 0.556 [0.327, 0.765] | 0.607 [0.311, 0.806] | 0.412 [0.249, 0.574] | 0.480 [0.299, 0.725] | -0.051 [-0.386, 0.308] | 0.145 [-0.127, 0.420] |
| kind:lab_bio | 213 | 4 | 0.783 [0.460, 0.904] | 0.398 [0.305, 0.505] | 0.531 [0.367, 0.662] | 0.686 [0.407, 0.829] | 0.385 [0.040, 0.552] | 0.252 [0.035, 0.398] |
| kind:other | 51 | 0 | 0.608 [0.470, 0.667] | 0.511 [0.342, 0.611] | 0.531 [0.356, 0.639] | 0.562 [0.459, 0.637] | 0.097 [0.000, 0.219] | 0.077 [-0.104, 0.263] |
| kind:sales_ops | 395 | 8 | 0.632 [0.463, 0.773] | 0.527 [0.380, 0.703] | 0.476 [0.360, 0.588] | 0.531 [0.399, 0.674] | 0.105 [-0.081, 0.295] | 0.156 [0.020, 0.284] |
| kind:software | 107 | 2 | 0.634 [0.315, 0.856] | 0.310 [0.290, 0.325] | 0.475 [0.315, 0.631] | 0.440 [0.304, 0.575] | 0.324 [-0.002, 0.545] | 0.159 [-0.127, 0.384] |
| all_with_card | 1135 | 43 | 0.678 [0.600, 0.747] | 0.459 [0.393, 0.531] | 0.490 [0.432, 0.545] | 0.582 [0.501, 0.655] | 0.219 [0.116, 0.312] | 0.188 [0.108, 0.264] |
| agreed_eligibility_pass | 455 | 29 | 0.741 [0.655, 0.811] | 0.499 [0.415, 0.586] | 0.489 [0.416, 0.554] | 0.633 [0.544, 0.707] | 0.242 [0.124, 0.348] | 0.252 [0.154, 0.340] |
| eligibility_pass_design | 511 | 36 | 0.673 [0.592, 0.741] | 0.479 [0.407, 0.557] | 0.462 [0.400, 0.524] | 0.603 [0.517, 0.680] | 0.194 [0.084, 0.289] | 0.211 [0.121, 0.296] |
| employer_new_vs_fresh_tests_1_3 | 20 | 0 | 0.325 [0.296, 0.333] | 0.333 [0.333, 0.333] | 0.263 [0.191, 0.306] | 0.316 [0.275, 0.333] | -0.009 [-0.037, 0.000] | 0.062 [0.018, 0.126] |
| employer_shared_with_fresh_tests_1_3 | 1119 | 43 | 0.680 [0.604, 0.747] | 0.456 [0.391, 0.527] | 0.494 [0.434, 0.550] | 0.584 [0.502, 0.656] | 0.225 [0.125, 0.314] | 0.186 [0.106, 0.264] |
| extension_4b | 199 | 3 | 0.734 [0.383, 0.857] | 0.515 [0.308, 0.719] | 0.492 [0.315, 0.648] | 0.609 [0.371, 0.797] | 0.218 [-0.090, 0.518] | 0.241 [-0.008, 0.429] |
| first_940 | 940 | 40 | 0.674 [0.588, 0.744] | 0.454 [0.388, 0.522] | 0.490 [0.430, 0.547] | 0.579 [0.490, 0.653] | 0.221 [0.118, 0.315] | 0.185 [0.097, 0.264] |
| field:data | 218 | 11 | 0.617 [0.444, 0.754] | 0.460 [0.319, 0.620] | 0.440 [0.331, 0.555] | 0.592 [0.414, 0.741] | 0.157 [-0.015, 0.336] | 0.176 [-0.027, 0.364] |
| field:health_nursing | 162 | 14 | 0.632 [0.469, 0.765] | 0.367 [0.298, 0.466] | 0.454 [0.340, 0.562] | 0.578 [0.393, 0.724] | 0.265 [0.036, 0.442] | 0.178 [-0.011, 0.343] |
| field:internship | 58 | 6 | 0.556 [0.327, 0.765] | 0.607 [0.311, 0.806] | 0.412 [0.249, 0.574] | 0.480 [0.299, 0.725] | -0.051 [-0.386, 0.308] | 0.145 [-0.127, 0.420] |
| field:lab_bio | 203 | 4 | 0.771 [0.407, 0.887] | 0.380 [0.308, 0.501] | 0.491 [0.323, 0.626] | 0.621 [0.336, 0.767] | 0.391 [0.017, 0.528] | 0.279 [0.051, 0.415] |
| field:other | 215 | 1 | 0.754 [0.516, 0.930] | 0.461 [0.353, 0.546] | 0.548 [0.412, 0.689] | 0.546 [0.470, 0.602] | 0.293 [0.053, 0.483] | 0.207 [-0.001, 0.357] |
| field:sales_ops | 212 | 6 | 0.658 [0.436, 0.840] | 0.441 [0.315, 0.556] | 0.476 [0.331, 0.608] | 0.466 [0.334, 0.635] | 0.218 [-0.038, 0.435] | 0.182 [0.012, 0.361] |
| field:software | 71 | 1 | 0.554 [0.300, 0.802] | 0.308 [0.281, 0.326] | 0.483 [0.300, 0.672] | 0.424 [0.296, 0.578] | 0.246 [-0.015, 0.506] | 0.070 [-0.115, 0.225] |
| length:1300-3000 | 79 | 5 | 0.738 [0.516, 0.909] | 0.442 [0.294, 0.570] | 0.577 [0.384, 0.737] | 0.605 [0.371, 0.812] | 0.296 [0.013, 0.529] | 0.161 [-0.056, 0.351] |
| length:3000+ | 1024 | 38 | 0.618 [0.535, 0.694] | 0.427 [0.358, 0.512] | 0.472 [0.410, 0.532] | 0.553 [0.466, 0.637] | 0.191 [0.072, 0.295] | 0.146 [0.059, 0.231] |
| length:700-1300 | 12 | 0 | 0.649 [0.303, 0.667] | 0.381 [0.121, 0.583] | 0.497 [0.244, 0.667] | 0.370 [0.163, 0.552] | 0.268 [0.017, 0.500] | 0.152 [0.000, 0.347] |
| length:empty | 24 | 0 | 0.576 [0.420, 0.667] | 0.472 [0.263, 0.627] | 0.310 [0.095, 0.527] | 0.507 [0.347, 0.625] | 0.104 [-0.077, 0.297] | 0.265 [0.030, 0.492] |

Colab sessions: matcher-c3b-test4 (B, L4, 22:09-22:27Z), matcher-h-test4 try 1 (h, L4, 22:27-22:45Z: exec timed out during setup, stopped before any job ran, no output), matcher-h-test4 try 2 (h, L4, 22:58-23:19Z), matcher-base-test4 (base, L4, 22:45-22:58Z).
