#!/bin/bash

LIST="France
      Australia
      United_States_of_America 
      United_Kingdom
      Italy
      Spain
      Belgium
      India
      Brazil
      Iran
      Japan
      Russia
      Canada
      China
      Germany
      Austria
      Turkey
      South_Africa
      Netherlands
      South_Korea
      Switzerland
      Portugal
      Israel
      Norway
      Slovenia"

mkdir png
    
for COUNTRY in $LIST; do
    notify-send 'computing '$COUNTRY
    echo "Compute "$COUNTRY 
    ./covid.py $COUNTRY
done

notify-send 'computing done'

mv *.png png

echo "# crappy-code-covid [![Build Status](https://cloud.drone.io/api/badges/a-lemonnier/crappy-code-covid/status.svg)](https://cloud.drone.io/a-lemonnier/crappy-code-covid) ![py CI](https://github.com/a-lemonnier/crappy-code-covid/workflows/py%20CI/badge.svg)" > README.md
echo " " >> README.md
echo "A bad py code fitting the covid data from ECDC. ECDC updates his database at 12h00 CEST atm." >> README.md

echo " " >> README.md

for COUNTRY in png/*
do
  echo  "!["$COUNTRY"]("$COUNTRY")" >> README.md
done

echo " " >> README.md

echo "Dependencies:">> README.md
echo "- scipy">> README.md
echo "- numpy">> README.md
echo "- matplotlib">> README.md
echo "- pandas">> README.md
echo "- tk">> README.md
echo "- xlrd">> README.md
echo "- requests">> README.md
echo "- notify-send">> README.md

