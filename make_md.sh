#!/bin/bash

mv *.png png

echo "# crappy-code-covid [![Build Status](https://cloud.drone.io/api/badges/a-lemonnier/crappy-code-covid/status.svg)](https://cloud.drone.io/a-lemonnier/crappy-code-covid) ![py CI](https://github.com/a-lemonnier/crappy-code-covid/workflows/py%20CI/badge.svg)" > README.md
echo " " >> README.md
echo "A bad py code fitting the covid data from ECDC. ECDC updates his database at 12h00 CEST atm. Demographic data are from World Bank." >> README.md

echo " "  >> README.md

echo  "- [ECDC Data](https://www.ecdc.europa.eu/en/publications-data/download-todays-data-geographic-distribution-covid-19-cases-worldwide)" >> README.md
echo  "- [WB Data](https://data.worldbank.org/indicator/sp.pop.totl)" >> README.md

echo " "  >> README.md

echo " " >> README.md
echo "Mise à jour quotidienne automatique par action github." >> README.md

echo " "  >> README.md

echo  "Version pdf: [covid.pdf](https://github.com/a-lemonnier/crappy-code-covid/raw/master/covid.pdf)" >> README.md

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
