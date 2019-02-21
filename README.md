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

| model         | features              | dropout       |epochs train  | train mad     | test mad |
| ------------- |:---------------------:|--------------:|--------------:| -------------:| --------:|
| simpleffnn    | statistical features1 | 0.25          | 1000          |1.2000000000000| 1.800    |
| simpleffnn    | statistical features1 | 0.25          | 100           |1.7000000000000| 1.672    |
| simpleffnn    | statistical features1 | 0.25          | 10            |2.0029595604348| 1.537    |
| simpleffnn    | statistical features1 | 0.25          | 3             |2.0029595604348| 1.540    |
| simpleffnn    | statistical features1 | 0.5           | 1000          |1.6404929039735|          |
| simpleffnn    | statistical features1 | 0.5           | 100           |1.9273089859860|          |
| simpleffnn    | statistical features1 | 0.5           | 10            |2.0137530461991|          |
| simpleffnn    | statistical features1 | 0.5           | 3             |2.0259630969190|          |
| --------------|-----------------------|---------------|---------------|---------------|----------|
| simpleffnn    | statistical features3 |               | 1000          |1.2999233295860|          |












