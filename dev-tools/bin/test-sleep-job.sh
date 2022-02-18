#!/usr/bin/env bash

set -e

job_name=$1
launch_script_name=$2
out_file_path=$3

sleep 4

echo "Value 1" &> "${out_file_path}"
echo "Value 2" &>> "${out_file_path}"
echo "EXP-RESULT: {'fit_predict_time': 115.589062, 'predict_time': 19.987819, 'metric_value': 961597.8767943359, 'test_metric_value': 68959274.66959807}" &>> "${out_file_path}"
echo "something else" &>> "${out_file_path}"
