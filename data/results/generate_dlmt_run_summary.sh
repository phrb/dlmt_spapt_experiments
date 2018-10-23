#! /usr/bin/bash

DLMT_FILES=$(find . | grep -v "random" | grep "log")

ITERATIONS=4

TABLE=$'kernel,run_id,de_step_1,de_step_2,de_step_3,de_step_4,removed_step_1,removed_step_2,removed_step_3,removed_step_4,design_best_step_1,design_best_step_2,design_best_step_3,design_best_step_4,predicted_best_step_1,predicted_best_step_2,predicted_best_step_3,predicted_best_step_4\n'

for f in $DLMT_FILES; do
    RUN_ID="$(echo "$f" | cut -d_ -f6 | cut -d/ -f1),"
    KERNEL="$(echo "$f" | cut -d_ -f6 | cut -d/ -f2 | cut -d. -f1)"
    TABLE+="$KERNEL,$RUN_ID"

    ITEMS=$(grep "D-Eff" $f | wc -l)
    if [[ "$ITEMS" -gt 0 ]]
    then
        TABLE+="$(grep "D-Eff" $f | cut -d" " -f4 | tr $'\n' ',')"
    fi

    MISSING=$(expr $ITERATIONS - $ITEMS)
    if [[ $MISSING -gt 0 ]]
    then
        for (( i=1; i<=$MISSING; i++ )); do
            TABLE+=","
        done
    fi

    ITEMS=$(grep "Predicting Best Values for:" $f | wc -l)
    if [[ "$ITEMS" -gt 0 ]]
    then
        TABLE+="$(grep "Predicting Best Values for:" $f | cut -d"[" -f2 | cut -d"]" -f1 | sed "s/^/\"/" | sed "s/\$/\"/" | sed "s/,/:/g" | tr $'\n' ',')"
    fi

    MISSING=$(expr $ITERATIONS - $ITEMS)
    if [[ $MISSING -gt 0 ]]
    then
        for (( i=1; i<=$MISSING; i++ )); do
            TABLE+=","
        done
    fi

    ITEMS=$(grep "Slowdown (Design Best):" $f | uniq | wc -l)
    if [[ "$ITEMS" -gt 0 ]]
    then
        TABLE+="$(grep "Slowdown (Design Best):" $f | cut -d" " -f4 | uniq | tr $'\n' ',')"
    fi

    MISSING=$(expr $ITERATIONS - $ITEMS)
    if [[ $MISSING -gt 0 ]]
    then
        for (( i=1; i<=$MISSING; i++ )); do
            TABLE+=","
        done
    fi

    ITEMS=$(grep "Slowdown (Predicted Best):" $f | uniq | wc -l)
    if [[ "$ITEMS" -gt 0 ]]
    then
        TABLE+="$(grep "Slowdown (Predicted Best):" $f | cut -d" " -f4 | uniq | tr $'\n' ',')"
    fi

    MISSING=$(expr $ITERATIONS - $ITEMS)
    if [[ $MISSING -gt 0 ]]
    then
        for (( i=1; i<=$MISSING; i++ )); do
            TABLE+=","
        done
    fi

    TABLE=$(sed '$s/,$//' <<< "$TABLE")
    TABLE+=$'\n'
done

#TABLE=$(sed '$s/\\n$//' <<< "$TABLE")
echo "$TABLE"
