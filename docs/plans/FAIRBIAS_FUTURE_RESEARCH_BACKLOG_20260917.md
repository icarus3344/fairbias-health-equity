# Future research backlog — keep outside the first-paper critical path

These are hypotheses/questions, not completed findings or promised method improvements.

1. **Budget-aware Joint search:** compare deferred AE, utility-first pruning, and feasible-incumbent policies under matched elapsed and model-evaluation budgets. Establish when delayed AE misses a beneficial early path. Current pilots are too narrow to decide.
2. **Numerical geometry stability:** quantify seed, initialization, iteration-cap and embedding-dimension sensitivity; separate stress convergence from downstream transformation stability. A recovered MDS cap does not prove a globally optimal embedding.
3. **Repeated model fitting:** optimized GBDT pilot spent roughly as much time evaluating utility as geometry. Investigate mathematically valid reuse or candidate screening; do not assume warm starts preserve original outcomes.
4. **GPU or compiled geometry:** consider only after profiling and a reproducible numerical contract; these are small feature-node matrices, not sample-pair embeddings. Accelerator overhead and changed arithmetic may outweigh benefits.
5. **Geometry and population fairness:** investigate why representation-level d_phi relates differently to survey-weighted DP/EO and access-to-care prediction across groups; avoid causal claims from geometric axes.
6. **External transportability and larger predictors:** additional independent health datasets, future annual cohorts, fuller-capacity TabM/TabICL comparisons. Register these as a subsequent study rather than selecting additions after viewing the present test rankings.

Add new items with evidence pointers and a concrete research question. Do not expand the overnight scope merely because an idea is interesting.
