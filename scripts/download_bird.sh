#!/usr/bin/env bash
set -euo pipefail

directory=${1:-data/bird}
raw="$directory/raw"
databases="$directory/databases"
mkdir -p "$raw" "$databases"

train_revision=4068469807b255fcfc0816bdd520946fe460d256
mini_revision=f65faf4ae3b638c1fa6df1d3370c8d92c8366301

# Download the question files with Hugging Face's CLI.
if [[ ! -f "$raw/train_filtered.jsonl" ]]; then
    cp "$(uvx hf download "hf://datasets/birdsql/bird23-train-filtered@$train_revision/data/train-00000-of-00001.jsonl" --quiet)" \
        "$raw/train_filtered.jsonl"
fi
if [[ ! -f "$raw/mini_dev_sqlite.json" ]]; then
    cp "$(uvx hf download "hf://datasets/birdsql/bird_mini_dev@$mini_revision/data/mini_dev_sqlite-00000-of-00001.json" --quiet)" \
        "$raw/mini_dev_sqlite.json"
fi

# Download the database packages; the training ZIP contains a nested ZIP.
if [[ ! -f "$raw/train_databases.zip" ]]; then
    uvx gdown https://bird-bench.oss-cn-beijing.aliyuncs.com/train.zip \
        -O "$raw/train.zip" --continue
    unzip -p "$raw/train.zip" train/train_databases.zip > "$raw/train_databases.zip"
    rm "$raw/train.zip"
fi
if [[ ! -f "$raw/minidev_0703.zip" ]]; then
    uvx gdown 13VLWIwpw5E3d5DUkMvzw7hvHE67a4XkG -O "$raw/minidev_0703.zip" --continue
fi

# Extract each question's database into databases/<db_id>/<db_id>.sqlite.
extract_databases() {
    local archive=$1 questions=$2 prefix=$3 db
    while IFS= read -r db; do
        mkdir -p "$databases/$db"
        unzip -j -n -q "$archive" "$prefix/$db/$db.sqlite" -d "$databases/$db"
    done < <(jq -rs 'flatten | .[].db_id' "$questions" | sort -u)
}
extract_databases "$raw/train_databases.zip" "$raw/train_filtered.jsonl" train_databases
extract_databases "$raw/minidev_0703.zip" "$raw/mini_dev_sqlite.json" minidev/MINIDEV/dev_databases

printf 'Downloaded BIRD data to %s\n' "$directory"
