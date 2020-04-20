#!/bin/bash

notify-send 'merging pdf'
rm covid.pdf
pdfunite *.pdf covid.pdf; rm covid_*.pdf;
notify-send 'merging done'



