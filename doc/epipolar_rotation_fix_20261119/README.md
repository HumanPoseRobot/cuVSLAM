# Epipolar rotation consistency fix

Based on `release/runtime_reduced_v1.35`, commit `7ac651f613fc9074bc161887530eb04778131ae1`. The published v1.35 package is unchanged. The evidence below was collected from the independent experiment before this commit; the final committed build is verified separately in the shared experiment directory.

## Cause and correction

Stock Eigen LM adds absolute log coordinates. `SurveyedTracking` supplied a local left-perturbation projection Jacobian without the SE(3)/SO(3) left-Jacobian chain rule. A deterministic central-difference check measured a maximum absolute discrepancy of 0.341897517 before correction and 0.000086 after correction.

Rotation-only `VectorTwist<3>` dropped translation. Refinement now fixes the original world optical center C and evaluates R(P-C), including angular prealignment and final pose recovery. The regression previously moved the center by 0.748331487 m; the corrected result passes the 0.00001 m preservation bound.

The Huber-transformed residual needs its own derivative weight. Refinement now also checks finite residual/pose and the existing residual threshold before writing back, leaving the input pose unchanged after rejection. Existing acceptance thresholds and iteration budgets are preserved.

## Files and reasons

| File | Change and reason |
|---|---|
| `libs/epipolar/resectioning_utils_internal.h` | Match the analytic Jacobian to absolute LM coordinates, differentiate the Huber transformation, preserve the optical center in rotation-only projection. |
| `libs/epipolar/resectioning_utils.cpp` | Pass the original center, reconstruct translation consistently, and publish only an accepted finite candidate. |
| `libs/epipolar/test/resectioning_utils_test.cpp` | Add six regressions: ordinary/robust Jacobians for 3/6 DOF, nonzero center preservation, and rejection without mutation. |
| `doc/epipolar_rotation_fix_20261119/README.md` | Explain the root cause, production scope, test method, and full replay results. |
| `doc/epipolar_rotation_fix_20261119/comparison.csv` | Store per-sequence before/after height, peak speed, ATE and continuity results. |
| `doc/epipolar_rotation_fix_20261119/solver_checks.json` | Store numerical derivative discrepancies, synthetic angular errors and test/seed counts. |
| `doc/epipolar_rotation_fix_20261119/strict_bitwise_checks.json` | Record byte-for-byte pose, gravity and timestamp equality for the four full replays. |
| `doc/epipolar_rotation_fix_20261119/experiment_provenance.json` | Identify the precommit experimental core, v1.35 baseline and patch hashes; these are not hashes of the later committed rebuild. |

## Test evidence

- CTest: 22/22 targets passed, with `GTEST_FILTER=-*SpeedUp*:*Speedup*`.
- Epipolar: 42 enabled tests passed; the existing six disabled tests remain disabled.
- The original three rotation cases passed for 51 fixed seeds, including 71125; no failures.
- Central-difference maximum absolute error: ordinary 6 DOF 0.000086, robust 6 DOF 0.000095, ordinary/robust 3 DOF 0.000089.
- Synthetic noiseless rotation errors decreased from 8.76224/9.02947/8.76133 degrees to 0.00001097/0.00000895/0.00001559 degrees.
- Repository pre-commit checks for the three changed C++ files and `git diff --check` passed.

## Full replay comparison with v1.35

Native resolution and matching inputs/settings; PGO off. The following numbers are identical in the v1.35 baseline and experimental fix.

| Sequence | Valid poses | Height change (m) | Peak speed (m/s) | ATE RMSE (m) | Tracking losses / gravity gap frames |
|---|---:|---:|---:|---:|---:|
| 195610 | 1175/1175 | -0.07519879928114157 | 1.1833827129403391 | No GT | 0 / 0 |
| data_test2 | 1793/1793 | -0.05514879765972239 | 0.8828953059252487 | No GT | 0 / 0 |
| room2 | 2882/2882 | -0.022489326400311364 | 2.245452156578879 | 0.08788202414602098 | 0 / 0 |
| room6 | 2636/2636 | 0.0783022867950856 | 1.093405385106842 | 0.05268457550409151 | 0 / 0 |

8486 valid poses total. Raw bytes of positions, orientations, gravity and timestamps are identical; input audit hashes match. Height uses gravity up and first/last one-second medians. ATE uses the saved replay evaluator GT-sample association and SE(3) alignment without scale; do not mix it with historical camera-sampled summaries. Gravity gaps are counted after initialization.

## Production scope

Whole-repository call search finds the unconstrained `<3/6>` helper only in tests. Current stereo/multicamera tracking uses `pnp_.solve()` in `track_online_multi.cpp`. Mono uses the separate `OptimizeCameraExtrinsicsExpMapConstrained<false>` with numerical differentiation; its residual and update code are unchanged. Other LM call sites use numerical differentiation consistent with additive coordinates.

The legacy helper uses CPU float column-pivoted QR, not GPU Schur. The global LM bridge, GPU PnP, cuNLS and navigation window are unchanged. This fixes real legacy defects but does not improve or explain existing stereo/four-camera VIO height errors in these replays.

## Independent artifacts

Shared experiment root: `/nas/shared/baidx/cuvslam_runs/epipolar_rotation_fix_20261119/`. It contains source, test logs, fixed-seed logs, candidate runtime, four replay outputs and binary provenance. From-commit reconstruction and verification use a separate `committed_rebuild/` beneath that root so precommit evidence remains intact.
