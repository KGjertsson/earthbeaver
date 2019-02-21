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

##  Performance log:

| model         | features              | epochs train  | train mad     | test mad |
| ------------- |:---------------------:|--------------:| -------------:| --------:|
| simpleffnn    | statistical features1 | 1000          |1.2            | 1.8      |
| simpleffnn    | statistical features1 | 100           |1.7            |1.672     |
| simpleffnn    | statistical features1 | 10            |2.0029595604348|1.537     |
| simpleffnn    | statistical features1 | 3             |2.0029595604348|1.540     |
| simpleffnn    | statistical features3 | 1000          |1.2999233295860|          |












