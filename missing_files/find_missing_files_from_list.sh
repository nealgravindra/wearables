#!/bin/bash

# The list of files to find and copy
readarray -t files <<EOF
modelcmp_collect_csvs_v52.ipynb
interp_featattr.ipynb
eval_model_embeddings.ipynb
eval_error_analysis_v71.ipynb
eval_test_v43_nalab3.csv
eval_v52.csv
eval_test_v43.csv
results_head2head_v52.csv
md_v522_220124.csv
trainer_itv52_InceptionTime_GA5.pkl
213-itv52_InceptionTime_GA5.pkl
213-itv52_InceptionTime_GA5.pkl
trainer_itv52_InceptionTime_GA5.pkl
md_v522_220124.csv
trainer_itv52_InceptionTime_GA5.pkl
213-itv52_InceptionTime_GA5.pkl
embeds_v522_220124.csv
trainer_itrandv52_InceptionTime_GA5.pkl
219-itrandv52_InceptionTime_GA5.pkl
itrand_embds.csv
mdrand.csv
data_dtw.pkl
md_v522_220124.csv
trainer_itv52_InceptionTime_GA5.pkl
213-itv52_InceptionTime_GA5.pkl
embeds_v522_220124.csv
md_220912.csv
trainer_itv71_InceptionTime_GA4.pkl
281-itv71_InceptionTime_GA4.pkl
md_results_train_220912.csv
embeds_train_220912.csv
embeds_val_220912.csv
md_results_test_220912.csv
embeds_test_220912.csv
md_predictability_knn_10wk.pkl
MOD_Data_2021.csv
EOF

# Directory to search in
SEARCH_DIR="/home/ngrav/"

# Destination directory for found files
DEST_DIR="/home/ngrav/project/wearables/missing_files/missing_files"
mkdir -p "$DEST_DIR"

# Archive name
ARCHIVE_NAME="compressed_missing_files.tar.gz"

# Track not found files
declare -a not_found_files=()

# Search, copy, and accumulate not found files
for file in "${files[@]}"; do
    found=$(find "$SEARCH_DIR" -name "$file" -print -quit)
    if [ -n "$found" ]; then
        # File found, copy it to DEST_DIR
        cp "$found" "$DEST_DIR"
        echo "Copied: $file"
    else
        # File not found, add to the list
        not_found_files+=("$file")
    fi
done

# Compress copied files
tar -czvf "$ARCHIVE_NAME" -C "$DEST_DIR" .
echo "Compressed into: $ARCHIVE_NAME"

# Report not found files
if [ ${#not_found_files[@]} -eq 0 ]; then
    echo "All files were found and copied."
else
    echo "The following files were not found:"
    printf ' - %s\n' "${not_found_files[@]}"
fi
