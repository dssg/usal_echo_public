#!/bin/bash
# description:	This shell script processes Xcelera_tablas database csv files so that they can be recreated in postgres.
# 		The following processing steps are performed:
# 		1) The last row of each file is removed, as it contains only NAN values (TODO: should rather use grep to find NAN rows)
# 		2) File encoding is changed from IS08859 to UTF8
# author:	wt
# date: 	18 June 2019
# usage:	./file_processing.sh
# notes:	chmod +x file_processing.sh (make executable to run)
#		saves data to ~/data_usal/Xcelera_tablas_clean	
#=====================================================================

read -p "Specify path to raw data files: " RAW_DATA_DIR
CLEAN_DATA_DIR=$HOME/data_usal/Xcelera_tablas_clean
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

