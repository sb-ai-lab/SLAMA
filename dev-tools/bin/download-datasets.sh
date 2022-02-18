#!/usr/bin/env bash

set -ex

wget https://www.openml.org/data/get_csv/53268/ailerons.arff -O /opt/ailerons.csv
wget https://www.openml.org/data/get_csv/1798106/phpV5QYya -O /opt/PhishingWebsites.csv
wget https://www.openml.org/data/get_csv/53515/kdd_internet_usage.arff -O /opt/kdd_internet_usage.csv
wget https://www.openml.org/data/get_csv/22045221/dataset -O /opt/nasa_phm2008.csv
wget https://www.openml.org/data/get_csv/1798816/php9VSzX6 -O /opt/Buzzinsocialmedia_Twitter.csv
wget https://www.openml.org/data/get_csv/52407/internet_usage.arff -O /opt/internet_usage.csv
wget https://www.openml.org/data/get_csv/1798765/phpYLeydd -O /opt/gesture_segmentation.csv
wget https://www.openml.org/data/get_csv/52422/ipums_la_97-small.arff -O /opt/ipums_97.csv

head -n 25001 /opt/Buzzinsocialmedia_Twitter.csv > /opt/Buzzinsocialmedia_Twitter_25k.csv

#cp examples/data/sampled_app_train.csv /opt
#cp examples/data/small_used_cars_data.csv /opt
