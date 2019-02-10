# earthbeaver

## AI - metod
Recurrrent neural network - RNN

## Preprocessing
~~LTA/STA - Long Term Averaging och Short Term Averaging~~ - flyttad till features

## Postprocessing
~~Behövs?? Kanske det, viktningsfunktion/convulution?  ~~
Avfärdad då tvälingsdata är **osorterade** 150k bitar.



## Tankar:
Plocka ut 150K bitar på random för att validera modellen så man ser var den gör bra ifrån sig och var den gör dåligt
~~Preprocessing med STA/LTA kommer att behövas.~~ - Avfärdad

Vad händer om vi tränar modellen på random utvalda 150k bitar?

## Features:
mean  
median(abs)  
std  
max  
min
Classic LTA/STA
Recursive LTA/STA??

## TODO:
Refactor code  
Look into new state of the art kernel: https://www.kaggle.com/artgor/earthquakes-fe-more-features-and-samples   
Check length from peak to failure







