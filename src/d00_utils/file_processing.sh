#!/bin/bash

read -p "Specify path to raw data files: " RAW_DATA_DIR
CLEAN_DATA_DIR=$HOME/data/Xcelera_tablas_clean
mkdir -p $CLEAN_DATA_DIR

FILES=$RAW_DATA_DIR/*

for f in $FILES
do
	if [ ${f: -4} == ".csv" ]
	then
	        filename=$(basename "$f" ".csv")
		filepath="$CLEAN_DATA_DIR/${filename}_utf8.csv"
	        head -n -1 $f | iconv -f ISO88592 -t UTF8 > $filepath;
	        echo "Processed $filename"
	fi
done

echo "Files processed and saved to $CLEAN_DATA_DIR"

