#!/bin/bash

# ----------------------------- Configuration -----------------------------
PROJECT_PATH="."
METRIC_SCRIPT="$PROJECT_PATH/utils/metric.py"
MODEL_NAME="aerial_vla"


RESULTS_DIR="$PROJECT_PATH/eval_results/$MODEL_NAME"
DETAILED_CSV="$PROJECT_PATH/eval_results/$MODEL_NAME/evaluation_detailed.csv"
SUMMARY_CSV="$PROJECT_PATH/eval_results/$MODEL_NAME/evaluation_summary_aggregated.csv"
PATH_TYPE_LIST="full easy hard"
# -----------------------------------------------------------------------

echo "Category,Map,Difficulty,Count,SR(%),OSR(%),NE,SPL(%)" > "$DETAILED_CSV"
export PYTHONPATH="${PROJECT_PATH}"

echo "------------------------------------------------"
echo " Phase 1: Detailed Map Evaluation "
echo "------------------------------------------------"

for category in "seen_valset" "unseen_map_valset" "unseen_object_valset"; do
    CATEGORY_DIR="$RESULTS_DIR/$category"
    if [ ! -d "$CATEGORY_DIR" ]; then continue; fi

    for map_dir in "$CATEGORY_DIR"/*; do
        if [ -d "$map_dir" ]; then
            MAP_NAME=$(basename "$map_dir")
            echo ">>> Evaluating: [$category] -> $MAP_NAME"

            ANALYSIS_ITEM="$category/$MAP_NAME"
            TEMP_LOG=$(mktemp)
            
            CUDA_VISIBLE_DEVICES=0 python3 $METRIC_SCRIPT \
                --root_dir "$RESULTS_DIR" \
                --analysis_list "$ANALYSIS_ITEM" \
                --path_type_list $PATH_TYPE_LIST \
                --map_filter "$MAP_NAME" > "$TEMP_LOG" 2>&1

            SRS=($(grep "Success Rate (SR):" "$TEMP_LOG" | awk -F': ' '{print $2}' | tr -d '%'))
            OSRS=($(grep "Oracle Success Rate (OSR):" "$TEMP_LOG" | awk -F': ' '{print $2}' | tr -d '%'))
            NES=($(grep "Average Normalized Error (NE):" "$TEMP_LOG" | awk -F': ' '{print $2}'))
            SPLS=($(grep "Average Success Path Length (SPL):" "$TEMP_LOG" | awk -F': ' '{print $2}' | tr -d '%'))
            COUNTS=($(grep "Total Count:" "$TEMP_LOG" | awk -F': ' '{print $2}'))

            TYPES=("full" "easy" "hard")
            for i in 0 1 2; do
                TYPE=${TYPES[$i]}
                SR=${SRS[$i]:-"0"}
                OSR=${OSRS[$i]:-"0"}
                NE=${NES[$i]:-"0"}
                SPL=${SPLS[$i]:-"0"}
                COUNT=${COUNTS[$i]:-"0"}

                echo "$category,$MAP_NAME,$TYPE,$COUNT,$SR,$OSR,$NE,$SPL" >> "$DETAILED_CSV"
            done
            echo ",,,,,,," >> "$DETAILED_CSV"
            rm "$TEMP_LOG"
        fi
    done
done

echo -e "\n------------------------------------------------"
echo " Phase 2: Generating Aggregated Tables (Academic Style) "
echo "------------------------------------------------"

echo "Category,Difficulty,Total_Count,Avg_SR(%),Avg_OSR(%),Avg_NE,Avg_SPL(%)" > "$SUMMARY_CSV"

for cat in "seen_valset" "unseen_map_valset" "unseen_object_valset"; do
    echo "Processing summary for $cat..."
    echo "" >> "$SUMMARY_CSV"
    
    for diff in "full" "easy" "hard"; do
        awk -F',' -v c="$cat" -v d="$diff" '
            BEGIN { sum_cnt=0; w_sr=0; w_osr=0; w_ne=0; w_spl=0 }
            $1 == c && $3 == d && $4 > 0 {
                sum_cnt += $4
                w_sr += ($4 * $5)
                w_osr += ($4 * $6)
                w_ne += ($4 * $7)
                w_spl += ($4 * $8)
            }
            END {
                if (sum_cnt > 0) {
                    printf "%s,%s,%d,%.2f,%.2f,%.2f,%.2f\n", c, d, sum_cnt, w_sr/sum_cnt, w_osr/sum_cnt, w_ne/sum_cnt, w_spl/sum_cnt
                } else {
                    printf "%s,%s,0,N/A,N/A,N/A,N/A\n", c, d
                }
            }
        ' "$DETAILED_CSV" >> "$SUMMARY_CSV"
    done
done

echo -e "\nDone!"
echo "1. Detailed results: $DETAILED_CSV"
echo "2. Aggregated Tables (for Paper): $SUMMARY_CSV"
