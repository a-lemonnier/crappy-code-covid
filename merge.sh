#!/bin/bash


notify-send 'merging pdf'
# gs -dBATCH -dNOPAUSE -q -sDEVICE=pdfwrite -sOutputFile=covid.pdf *.pdf
rm covid.pdf
pdfunite *.pdf covid.pdf; rm covid_*.pdf;
notify-send 'merging done'



