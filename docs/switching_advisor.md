# Switching advisor usage

The switching advisor recommends carrier-phase offsets using the existing
Hybrid POA-PFO optimizer. It does **not** predict arbitrary switch states. The
inverter simulator generates the binary gate array for the selected phases,
and a verification step rejects invalid cell states, invalid gate dimensions,
non-binary gates, or shoot-through.

Example:

```python
from assistant.switching_advisor import recommend_switching

recommendation = recommend_switching(
    modulation_index=0.75,
    reference_phase_degrees=0.0,
    dc_voltage_v=100.0,
    resistance_ohm=30.0,
    inductance_h=0.03,
    objective="voltage",
    seed=42,
)
print(recommendation.carrier_phase_offsets_degrees)
print(recommendation.simulator_verified_thd_percent)
print(recommendation.gate_states.shape)  # (3, 4, 400)
```

`objective="voltage"` minimizes event-based Parseval total voltage THD.
`objective="current"` minimizes event-based current harmonics 2–100, derived
from event voltage harmonics through the fixed RL impedance. Both metrics are
recomputed using the simulator conditions; sampled voltage/current THD
harmonics 2–100 are also returned separately. `predicted_thd_percent` is
`None`: no prediction model is used.

If optimization fails, or the independently verified selected objective is
worse than the baseline, the returned recommendation uses the baseline phases
`(0°, 60°, 120°)` and explicitly says no improvement was found. Modulation
outside inclusive `[0, 1]` is rejected and never clamped. Zero modulation is
also rejected by the advisor because THD is undefined with zero fundamental.

This software is an offline analysis/recommendation aid. It is **not** a
hardware controller, real-time control system, or protection system. It does
not validate physical hardware, switching dead time, or fault response.
