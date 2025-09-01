# To see the mapping clearly, find the corresponding columns, e.g.
awk -F, '!seen[$9]++ {print $9": ", $8, " - ", $7}' data/prefi_weaviate_clean-1_import_processed.csv 
awk -F, '!seen[$5]++ {print $5, $6}' data/prefi_weaviate_clean-1_import_processed.csv 
csvcut -c "final_contract_status,final_contract_status_label" data/prefi_weaviate_clean-1_modified_import_processed.csv
in="eval_algos"&&kdiff3 src/${in}.ai src/${in}.py -o src/${in}.mrg
in="eval_algos"&&mv src/${in}.py src/${in}.py.bak&&mv src/${in}.mrg src/${in}.py