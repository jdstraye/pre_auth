#! /bin/bash

# Immediately terminate the script if any command fails
set -e
set -o pipefail
source .venv_pre_auth/bin/activate

input="data/prefi_weaviate_clean-1_modified.json"
imported="data/$(basename ${input} .json)_import_processed.csv"
echo "========" | tee /tmp/log
echo "== Importing JSON: ${input} ==" | tee -a /tmp/log
python -m src.ingest ${input} --column_headers_json src/column_headers.json -o ${imported} 2>&1 | tee -a /tmp/log
echo "========" | tee -a /tmp/log
echo "== Allocating data to data/test*.csv and data/train*.csv ==" | tee -a /tmp/log
python src/allocate.py --input_csv ${imported} --out_train data/splits/train.csv --out_test data/splits/test.csv --add_date --create_links 2>&1 | tee -a /tmp/log
echo "========" | tee -a /tmp/log | tee -a /tmp/log
echo "== Spot Check compare columns in Golden reference, imported results, and training data ==" | tee -a /tmp/log | tee -a /tmp/log
python src/compare_columns.py -csv ${imported} -csv data/splits/train_latest.csv -gold_json src/column_headers.json 2>&1 | tee -a /tmp/log
#20250823a xterm -e 'tail -f /tmp/logs/pre_auth_eval_algos.log' &
xterm -e '/usr/bin/bash -c "tail -f /tmp/log logs/pre_auth_train_latest.log 2>&1 | grep --line-buffered -i -E \"complete|error|critical\"; bash"' &
echo "========" | tee -a /tmp/log
echo "== Evaluating Algorithms" | tee -a /tmp/log
python -m src.eval_algos --column_headers_json src/column_headers.json --train_csv data/splits/train_latest.csv --test_csv data/splits/test_latest.csv 2>&1 | tee -a /tmp/log