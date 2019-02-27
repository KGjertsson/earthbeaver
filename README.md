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
View tensorboard with: tensorboard --logdir=Graph --host=localhost --port=8088

| model         | features              | dropout       |epochs train   | train mae     | test mae |
| ------------- |:---------------------:|--------------:|--------------:| -------------:| --------:|
| simpleffnn    | statistical features1 | 0.25          | 1000          |1.2000000000000| 1.800    |
| simpleffnn    | statistical features1 | 0.25          | 100           |1.7000000000000| 1.672    |
| simpleffnn    | statistical features1 | 0.25          | 10            |2.0029595604348| 1.537    |
| simpleffnn    | statistical features1 | 0.25          | 3             |2.0029595604348| 1.540    |
| simpleffnn    | statistical features1 | 0.5           | 1000          |1.6404929039735| 1.650    |
| simpleffnn    | statistical features1 | 0.5           | 100           |1.9273089859860| 1.596    |
| simpleffnn    | statistical features1 | 0.5           | 10            |2.0137530461991| 1.531    |
| simpleffnn    | statistical features1 | 0.5           | 3             |2.0259630969190| 1.497    |
| --------------|-----------------------|---------------|---------------|---------------|----------|
| simpleffnn    | statistical features3 | 0.25          | 1000          |1.3058205365700| 1.830    |
| simpleffnn    | statistical features3 | 0.25          | 100           |1.8733245993213| 1.622    |
| simpleffnn    | statistical features3 | 0.25          | 10            |2.0358140996289| 1.522    |
| simpleffnn    | statistical features3 | 0.25          | 3             |2.0310845783116| 1.489    |
| simpleffnn    | statistical features3 | 0.5           | 10            |               |          |
| simpleffnn    | statistical features3 | 0.5           | 3             |2.0409161202138| 1.471    |
| --------------|-----------------------|---------------|---------------|---------------|----------|
| simpleffnn~10 | statistical features3 | 0.25          | 3             |2.0229714746295| 1.484    |
| simpleffnn~100| statistical features3 | 0.5           | 3             |2.0308562487956| 1.469    |
| simpleffnn~10 | statistical features3 | 0.5           | 3             |2.0319643617985| 1.473    |
| --------------|-----------------------|---------------|---------------|---------------|----------|
| simpleffnn    | statistical features3 | 0.6           | 3             |               |          |
| simpleffnn    | statistical features3 | 0.7           | 3             |               |          |
| simpleffnn    | statistical features3 | 0.8           | 3             |               |          |













