# TreadmillCodeBOA

General code base for extracting dependent variables from our motion capture treadmill set up
All inputs are .txt files from The Motion Monitor using a Bertec Treadmill and Vicon cameras
This code is provided without warranty.

## Treadmill performance scripts (`Python/`)

| Script | Use | Force plates |
|---|---|---|
| `Performance_Run_ExtractKinematicKineticVars.py` | Level running | Both feet land on the left plate; left/right is inferred |
| `Performance_Walk_ExtractKinematicKineticVars.py` | Uphill/downhill walking | One foot per plate |

Set `fPath`, the belt `speed` for the study, `save_on` and `debug` at the top of
each script. Run with `debug = 1` on the first pass to accept or reject each
trial; rejected trials are written to `BadFiles.csv`.

### Energy efficiency outputs

COM work is computed with the individual limbs method over each stride (foot
contact to the subsequent foot contact of the same foot).

| Column | Units | Description |
|---|---|---|
| `COMWork_pos`, `COMWork_neg` | J | Positive/negative COM work per stride |
| `StrideTime`, `StrideDist` | s, m | Stride time; stride distance = stride time × belt speed |
| `COMWork_pos_dis`, `COMWork_neg_dis` | J/m | COM work per meter (cost of transport): stride work / stride distance |
| `Mass` | kg | Body mass from the force plate (mean vertical GRF / g) |

**Report COM work per meter (J/m).** On a treadmill at a set speed, a
configuration that lengthens the stride does more work per stride but also
covers more distance, so J per stride partly reflects stride length. Across
seven 2026 tests, the J/m effect was smaller than the J effect in 20 of 21
comparisons, and the confidence tier changed in 14 of 30 comparisons.

For J/kg/m, divide by `Mass` **averaged per subject across all configurations**
— each trial's force-plate mass includes the shoe, so a per-trial mass would
introduce a configuration effect.

### Left/right in running

Both feet land on one plate, so sides are assigned by strict alternation and a
majority vote of the good steps, using the best signal in the export:

1. Left knee sagittal angle at mid-stance (`LKneeAngle_Sagittal`)
2. Left foot position (`LFootPosDetect`)
3. Left toe height (`LeftToeZPosition`)
4. Medial/lateral GRF impulse (no motion capture)

The signal used is output as `LR_method`, and the vote margin is printed per
trial (near 50% = uncertain). The M/L impulse agreed with motion capture in only
~70% of trials, so treat `Side` from `LR_method == 'ML'` as unreliable. This does
not affect the work values: every step is analyzed once, as the lead limb of
its stride.

## Updates

- **2026-10:** Added COM work per meter (`COMWork_pos_dis`, `COMWork_neg_dis`)
  and `Mass` to the running and walking scripts. Running left/right now uses
  the best available signal (`LR_method`). Walking good strides must contain
  exactly one opposite foot contact. Per-stride metrics are appended directly
  where they are computed.
- **2026-09:** Added stride time/distance and left/right segmentation; COM work
  computed per stride (foot contact to foot contact of the same foot); first
  20 s of each trial disregarded.
