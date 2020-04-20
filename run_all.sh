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
      Slovenia
      Ireland
      Sweden
      Mexico
      Peru
      Chile
      Ecuador
      Venezuela
      Colombia
      Niger
      Cameroon"

mkdir png
    
for COUNTRY in $LIST; do
     (notify-send 'computing '$COUNTRY; echo "Compute "$COUNTRY; ./covid.py $COUNTRY) &
done

wait

