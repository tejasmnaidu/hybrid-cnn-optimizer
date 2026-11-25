# Optimizer Validation

## Units and search space

All phase optimizer positions and carrier offsets use degrees.
`candidate_to_carrier_offsets` keeps cell 1 at `0°` and maps the two optimized
coordinates to cell 2 and cell 3. The event and ideal simulators receive those
degree offsets directly. The reference-phase operating point is radians only
inside `evaluate_event_based_thd`/`simulate_inverter`; the live operating
phase UI converts degrees to radians before calling those APIs.

Search space:

- cell 1: fixed at 0°
- cell 2: 0–360° circular
- cell 3: 0–360° circular
- modulation fixed for each run
- output objective: event-based `total_thd_percent`

Cell-swap canonicalization sorts the cell 2/cell 3 phase pair, so equivalent
permutations represent the same search candidate.

## Diagnosis

The earlier identical-baseline result was not assumed to be an implementation
defect. The real phase landscape was checked on a 36×36 grid. It is non-flat
and has several unique feasible values. The conventional `(60°,120°)` point
is also the grid minimum at the validated operating point used here.

Unit mixing was not found: degrees remain degrees in optimizer state and carrier
offset construction, while radians are used only for the separate reference
phase argument. Cache keys use canonical degree pairs and do not mix units.
The objective minimizes `mean_total_thd_percent`; a smaller THD is a better
feasible rank.

## Toy and real validation

Toy sphere/Rastrigin tests validate optimizer mechanics but are not evidence
of inverter optimization. Real validation requires POA, PFO and hybrid to
approach the 36×36 actual-simulator grid optimum across multiple seeds.

No objective, baseline, fitness sign or feasibility rule was changed.

## Current real-simulator result

The 36×36 grid (1,296 combinations) found `(60°,120°)` at the
`24.7234791510%` minimum for the default operating point. Across seeds
20261124, 20261125 and 20261126:

- PSO reached zero THD gap and zero phase gap with 63–64 unique evaluations.
- POA reached zero THD and phase gap with 71 unique evaluations.
- PFO reached zero THD and phase gap with 67 unique evaluations.
- Hybrid POA-PFO reached zero THD and phase gap with 70 unique evaluations.

This is real-simulator validation for this operating point. It is not a claim
of improvement over baseline: the baseline is already the measured grid
optimum.
