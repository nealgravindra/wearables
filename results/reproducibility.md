# Results

| Figure/table            | Code                                                                              |
| :---------------------: | :-------------------------------------------------------------------------------: |
| model comparison, Fig 1 | notebooks/modelcmp_collect_csvs_v52.ipynb                                         |
| cohort summary, Tab 1   | modelcomp_nonDL_baselines_dev.ipynb-->modelcomp_nonDL_v71.ipynb                   |
| data agg, Supp Fig 1    | notebooks/data_agg_scores_dev.ipynb AND notebooks/modelcmp_collect_csvs_v52.ipynb |
| Fig 2                   | notebooks/eval_error_analysis_v71.ipynb                         |
| Fig S2                  | notebooks/eval_error_analysis_v71.ipynb                                               |
| Fig 3                   | notebooks/interp_featattr.ipynb, interp_featattr_pvals.ipynb --> interp_featattr_v71.ipynb                     |
| Fig S3                  | notebooks/interp_featattr.ipynb, interp_featattr_pvals.ipynb                      |
| Fig 4                   | notebooks/eval_model_embeddings.ipynb                                             |
| Fig S4                  | notebooks/eval_model_embeddings.ipynb                                             |
| Tab S4                  | notebooks/eval_model_embeddings.ipynb                                             |

## Quick storage

Pre-processed metadata and output of best model (v71, as of 12 Sept 2022) are stored in embeds per split or `md_220912.csv` in the results folder, which has the output and splits plus calculated activities.


# Data

| Task                         | Code                                          |
| :--------------------------: | :-------------------------------------------: |
| select cohort and load data  | notebooks/*.ipynb    |
| re-train model               | notebooks/*.ipynb    |
| model embeddings to csv      | notebooks/eval_merge_model_out_to_md.ipynb    |
| merge model and metadata     | notebooks/eval_merge_model_out_to_md.ipynb    |
| freq cnt and writing.        | notebooks/meth_results_writing.ipynb          |

## Numbers

- `n_v71.ipynb`


## Labels

As of September 2022, there was some confusion as to the accuracy of the GAs from the filenames. For the most part, they do correlate, and there is perhaps one outlier, which is a limitation of the study and can be re-evaluated. For this analysis, see `chk_filenamelabelVedc.ipynb`


# Experiments

| Task                       | Code                                                          |
| :------------------------: | :-------------------------------------------------------:     |
| optimize ts augmentations  | results/train_v44.csv and notebooks/eval_aug_exps.ipynb       |
| compare model to nonDL     | run python scripts/modelcmp_noDL.py update files              |
| compare model to other DL  | run scripts/exps_v71.py with GRU/CNN model spec, update files |





