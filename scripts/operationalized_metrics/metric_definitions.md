# Metric Definitions

This document describes the metric and diagnostic fields written by `scripts/metrics.py` to the metrics JSON output. Unless noted otherwise, higher scores indicate better quality. Values may be `NaN` when a metric cannot be computed for the provided dataset. Truong et al. (2026) only defines coverage, marginal diversity, content realism, and stylistic realism (the higher level summary metrics). All others defined in this file serve as diagnostics that have strong correlation with the higher level metrics (validating them) or help quantify the contribution of individual exampmles to each metric. 

## Summary Metrics

The following fields appear under `summary`. These are all the high level metrics for evaluation dataset quality. Only this section is covered in Truong et al. (2026).

 - `coverage` - The fraction of required expected constituent combinations observed in the dataset. A value of `1.0` means every required combination was observed.

 - `coverage_vary` **(Coverage)** - Coverage score at the requested `--coverage-k`, derived from the smooth per-combination coverage penalty. It rewards combinations represented at least`k` times and partially rewards combinations represented fewer than `k` times.

 - `coverage_constrained` - Coverage after removing combinations declared unrealistic. This prevents known-invalid combinations from lowering the coverage score.

 - `coverage_vary_constrained` - The smooth `coverage_vary` score after unrealistic combinations are removed.

 - `coverage_uncertain` - The required-constituent coverage-vary score used for comparisons where optional fields should not affect the coverage estimate.

 - `dist_diversity` - The diversity score for the full dataset produced by DC-Score.

 - `marginal_diversity` **(Diversity)** - The mean distance-based diversity score across sufficiently large individual constituent groups. Small groups are omitted from this aggregate.

 - `full_stylistic_realism` **(Stylistic realism)** - A style-realism score based on the generated examples' average distance from the seed-example style distribution.

 - `subset_stylistic_realism` - The same style-realism score computed against the selected seed subset. For ablation purposes only.

 - `full_distributional_realism` **(Content realism)** - Calibrated distributional alignment between generated examples and all seed examples, based on Sinkhorn distance. The implementation calibrates this value so that larger values indicate better alignment.

 - `subset_distributional_realism` - The calibrated distributional alignment score computed against the selected seed subset. For ablation purposes only.

 - `cross_distribution` - For each required constituent column, the percentage of rows assigned to each observed value.

 - `marginal_distribution` - For each constituent column, the percentage of rows containing each schema-defined atomic value after category normalization. It is useful for checking whether expected values are under- or over-represented.

---

## JSON File Structure

 - `summary` - Compact scores and distributions intended for quick comparison across datasets. These are the only scores reported in Truong et al. (2026).

 - `inputs` (metadata) - Records the schema path, dataset path, detected text column, constituent
columns, invalid-combination count, seed count, selected seed subset, and
configured realistic exclusions. These are metadata.

 - `coverage`

 - `coverage_vary` - Coverage results across increasing values of `k`, where `k` is the desired number of examples per expected constituent combination. Note that `metrics.py` also produces a plot showing how varying `k` impacts coverage.

 - `coverage_constrained` - Coverage summary after excluding combinations listed by `--not-realistic`.
This is `null` when no exclusion list is supplied.

 - `coverage_vary_constrained` - The `coverage_vary` analysis after excluding unrealistic combinations. This is `null` when no exclusion list is supplied.

 - `coverage_vary_all` - `coverage_vary` computed using both required and optional constituent columns. The ordinary `coverage_vary` result uses required constituent columns.

 - `coverage_uncertain` - A coverage-vary summary over required constituent columns. It is retained as a separate output for uncertainty-oriented comparisons.

 - `diversity` - Detailed embedding-based diversity and redundancy results.

 - `realism_full` - Content realism results computed against all non-empty seed examples in the schema.

 - `realism_subset` - Realism results computed against the seed examples selected with `--seed-subset-indices`. This is `null` when no subset is requested. This is for ablation purposes only.

 - `row_level` - Per-row distances and outlier summaries used to inspect or regenerate examples.

 - `regeneration_subset` - **Not implemented!** Indices of rows marked as outliers. The `ranking` field is currently a placeholder and is not implemented.

 - `schema` - Selected schema metadata such as capability, deployment population, and contextual constraints. These are descriptive fields, not metrics.

### Coverage

The fields below appear in `coverage`, `coverage_vary`,
`coverage_vary_constrained`, `coverage_vary_all`, and `coverage_uncertain` when applicable.

 - `expected_combinations` - The constituent-value combinations expected from the schema.

 - `observed_combinations` - Expected combinations that occur in at least one dataset row.

 - `missing_combinations`

 - `expected_combinations_count`, `observed_combinations_count`, and
`missing_combinations_count`

 - `completeness_ratio` - `observed_combinations_count / expected_combinations_count`. It measures required combination coverage before realistic exclusions.

 - `holistic_completeness_score` - The mean fraction of rows in which optional constituent columns contain a non-placeholder value. It is `1.0` when no optional columns are present.

 - `category_stats` - Per-category embedding and concentration diagnostics. Each category includes:
    - `count`: number of examples assigned to the category.
    - `concentration_kappa`: concentration statistic for the category's embeddings.
    - `mean_pairwise_similarity`: average pairwise semantic similarity within the category.
    - `min_pairwise_similarity`: minimum pairwise semantic similarity within the category.
    - `n_outliers`: number of examples below the category's similarity threshold.
    - `outlier_indices`: row indices flagged as category outliers.

 - `hierarchical_results` - For multi-value constituent labels, these records measure whether an example is coherent with both its fine-grained label and its component labels. Each record includes `example_idx`, `fine_grained_label`, `fine_grained_similarity`, `coarse_similarities`, and `flags`.

 - `overall_diversity_score`

 - `redundant_pairs` and `redundant_pairs_sample` - Pairs of row indices whose semantic similarity exceeds the configured redundancy threshold. The serializable coverage output includes a sample. This calculation is based on cosine similarity and other diversity measures.

 - `default_k`, `default_score`, `max_k`, `k_values`, and `scores` - These fields describe the coverage-vary curve for reproducibility/replotting:
    - `default_k`: the requested `--coverage-k` value.
    - `default_score`: the average miscoverage penalty at `default_k`; lower is better.
    - `max_k`: the largest observed count for any expected combination.
    - `k_values`: evaluated values of `k` from `1` through `max_k`.
    - `scores`: average miscoverage penalty for each corresponding `k` value; lower is better.

 - `group_count`, `excluded_group_count`, `group_count_mean`, and
`group_count_median`

 - `penalty_definition`

 - `coverage_constrained` fields - The constrained coverage object additionally reports:
    - `coverage_constrained`: ratio of observed non-excluded combinations to expected non-excluded combinations.
    - `expected_combinations_count`: expected combinations after exclusions.
    - `observed_combinations_count`: observed combinations counted in the constrained denominator.
    - `excluded_combinations_count`: combinations removed by the exclusion list.
    - `non_penalized_missing_count`: missing combinations that were excluded and therefore not penalized.

### Diversity

 - `overall_dc_score` - classification based

 - `overall_vendi_score`- eigenspectrum-based diversity measure computed from pairwise semantic similarities.

 - `overall_mean_pairwise_similarity` - Mean semantic similarity across generated-example pairs based on cosine similarity.

 - `subgroup_stats` - Diversity metrics for each constituent subgroup. Each subgroup reports `count`, `dc_score`, `vendi_score`, `mean_pairwise_similarity`, and `surface_diversity_score`, plus its `redundant_pairs`.

 - `all_redundant_pairs_count` and `all_redundant_pairs_sample` - Count and sample of highly similar row pairs across the full dataset. This part is also reported for the detailed coverage report.

 - `lowest_diversity_groups` - The lowest-scoring subgroup labels and their distance-based diversity scores, used to identify potential diversity collapse.

### Realism

The same fields are used under `realism_full` and, when requested, `realism_subset` (for ablations). These 

 - `instance_stats` - Per-example realism diagnostics to determine contribution to the overal metric. Each record includes:
    - `idx`: dataset row index.
    - `label`: constituent label for the row.
    - `novelty_min`: distance to the closest seed example.
    - `novelty_avg`: average distance to the seed examples.
    - `log_likelihood`: model-based or embedding-based likelihood diagnostic used by the realism validator.
    - `is_in_scope`: whether the example is judged within the schema's supported scope.

 - `category_stats` - Per-constituent-category diagnostics:
    - `n_generated`: generated examples in the category.
    - `n_seeds`: seed examples in the category.
    - `seed_validity_score`: validity of the seed examples for the category.
    - `novelty_min`: minimum generated-to-seed distance for the category.
    - `novelty_avg`: average generated-to-seed distance for the category.
    - `containment_rate`: fraction of generated examples contained within the seed distribution.
    - `extrapolation_quality`: quality of generated examples that extend beyond the seed distribution.

 - `raw_sinkhorn_distance` - Uncalibrated Sinkhorn distance between generated and seed embedding distributions. Lower raw distance means closer distributions.

 - `calibrated_sinkhorn_distance` - Calibrated version of the Sinkhorn alignment score used in the compact summary. The calibration makes larger values correspond to better distributional realism.

 - `invalid_combo_coverage` - Counts of generated examples that fall into schema combinations marked invalid.

 - `mean_dataset_novelty_min` and `mean_dataset_novelty_avg` - Means of the per-example minimum and average distances to seed examples. Higher novelty indicates greater distance from the seeds; interpretation should be balanced against the realism and scope scores.

 - `mean_style_distance_avg` - Mean stylistic distance between generated examples and the seed style reference.

 - `style_realism_score` - A transformed style-distance score where higher values indicate closer stylistic alignment with the seed examples.

### Row-Level Fields

The `row_level` object reports dataset-wide summaries of row-level diagnostics:

- `n_rows`: number of rows evaluated.
- `outlier_counts`: counts for outlier codes `0` through `3`.
- `centroid_dist_mean`: mean distance from each row to its constituent/category centroid.
- `centroid_dist_median`: median distance from each row to its centroid.
- `dataset_center_dist_mean`: mean distance from each row to the overall dataset centroid.
- `min_seed_dist_mean`: mean closest-seed distance across rows.
- `coverage_outlier_indices`: rows flagged by coverage-category outlier detection.
- `invalid_group_indices`: rows whose constituent combination is invalid under the schema. Outlier codes are: `0` normal, `1` coverage outlier, `2` invalid constituent group, and `3` both coverage outlier and invalid constituent group.
