# crappy-code-covid [![Build Status](https://cloud.drone.io/api/badges/a-lemonnier/crappy-code-covid/status.svg)](https://cloud.drone.io/a-lemonnier/crappy-code-covid) ![py CI](https://github.com/a-lemonnier/crappy-code-covid/workflows/py%20CI/badge.svg)
 
A bad py code fitting the covid data from ECDC. ECDC updates his database at 12h00 CEST atm. Demographic data are from World Bank.
 
- [ECDC Data](https://www.ecdc.europa.eu/en/publications-data/download-todays-data-geographic-distribution-covid-19-cases-worldwide)
- [WB Data](https://data.worldbank.org/indicator/sp.pop.totl)
 
 
Mise à jour quotidienne automatique par action github.
 
Version pdf: [covid.pdf](https://github.com/a-lemonnier/crappy-code-covid/raw/master/covid.pdf)
 
![png/*](png/*)
 
Dependencies:
- scipy
- numpy
- matplotlib
- pandas
- tk
- xlrd
- requests
- notify-send
