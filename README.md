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

### Own experiments
| model         | features              | dropout       |epochs train   | activation fun| train mae     | test mae |
|:--------------|:---------------------:|:-------------:|--------------:|:-------------:|:-------------:|:--------:|
| simpleffnn    | statistical features1 | 0.25          | 1000          |    tanh       |1.2000000000000| 1.800    |
| simpleffnn    | statistical features1 | 0.25          | 100           |    tanh       |1.7000000000000| 1.672    |
| simpleffnn    | statistical features1 | 0.25          | 10            |    tanh       |2.0029595604348| 1.537    |
| simpleffnn    | statistical features1 | 0.25          | 3             |    tanh       |2.0029595604348| 1.540    |
| simpleffnn    | statistical features1 | 0.5           | 1000          |    tanh       |1.6404929039735| 1.650    |
| simpleffnn    | statistical features1 | 0.5           | 100           |    tanh       |1.9273089859860| 1.596    |
| simpleffnn    | statistical features1 | 0.5           | 10            |    tanh       |2.0137530461991| 1.531    |
| simpleffnn    | statistical features1 | 0.5           | 3             |    tanh       |2.0259630969190| 1.497    |
| --------------|-----------------------|---------------|---------------|---------------|---------------|----------|
| simpleffnn    | statistical features3 | 0.25          | 1000          |    tanh       |1.3058205365700| 1.830    |
| simpleffnn    | statistical features3 | 0.25          | 100           |    tanh       |1.8733245993213| 1.622    |
| simpleffnn    | statistical features3 | 0.25          | 10            |    tanh       |2.0358140996289| 1.522    |
| simpleffnn    | statistical features3 | 0.25          | 3             |    tanh       |2.0310845783116| 1.489    |
| simpleffnn    | statistical features3 | 0.5           | 10            |    tanh       |2.0118357298538| 1.513    |
| simpleffnn    | statistical features3 | 0.5           | 3             |    tanh       |2.0409161202138| 1.471    |
| simpleffnn    | statistical features3 | 0.6           | 3             |    tanh       |2.0442734291910| 1.483    |
| --------------|-----------------------|---------------|---------------|---------------|---------------|----------|
| simpleffnn~10 | statistical features3 | 0.25          | 3             |    tanh       |2.0229714746295| 1.484    |
| simpleffnn~100| statistical features3 | 0.5           | 3             |    tanh       |2.0308562487956| 1.469    |
| simpleffnn~10 | statistical features3 | 0.5           | 3             |    tanh       |2.0319643617985| 1.473    |
| --------------|-----------------------|---------------|---------------|---------------|---------------|----------|
| heavy_ffn     | statistical features3 | 0.5           | 10            |    relu       |1.9859311313708| 1.516    |
| heavy_ffn     | statistical features3 | 0.5           | 3             |    relu       |2.0442280633093| 1.600    |
| --------------|-----------------------|---------------|---------------|---------------|---------------|----------|
| simpleffnn    | statistical features4 | 0.5           | 10            |    tanh       |2.0113052635706| 1.489    |
| simpleffnn    | statistical features4 | 0.5           | 3             |    tanh       |2.0373891132104| 1.475    |
| simpleffnn    | statistical features5 | 0.5           | 10            |    tanh       |2.0246831776283| 1.510    |
| simpleffnn    | statistical features4 | 0.5           | 3             |    tanh       |2.0402858391355| 1.491    |

### Ensembled
| name                                                                                            | performance           | 
|:------------------------------------------------------------------------------------------------|:---------------------:|
|submission~simple_ffnn~epochs_3~dropout_0.5~gen_statistical_features3~n_networks=100+gpsubmission|  1.451                |












