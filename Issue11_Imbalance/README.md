# Efficientnet model - EfficientNetV2, EfficientNetB0

## TRAIN_SET_A
### EfficientNetV2
#### lr 0.0001, batch 32, epoch 50 
                    (-> lr 0.001 일때, loss값 너무 커짐)
```
Model: EfficientNetV2
Classification Report:
              precision    recall  f1-score   support

        BOLD       0.71      1.00      0.83        10
      BUBBLE       0.90      0.90      0.90        10
        BURR       0.57      0.80      0.67        10
      DAMAGE       0.67      0.20      0.31        10
         DOT       0.00      0.00      0.00        10
        DUST       0.40      1.00      0.57        10
        FOLD       0.50      0.80      0.62        10
        LINE       0.86      0.60      0.71        10
       REACT       1.00      1.00      1.00        10
        RING       1.00      1.00      1.00         1
     SCRATCH       0.86      0.60      0.71        10
         TIP       0.75      0.30      0.43        10

    accuracy                           0.66       111
   macro avg       0.68      0.68      0.64       111
weighted avg       0.66      0.66      0.62       111
```


### EfficientNetB0
#### lr 0.0001, batch 32, epoch 50 
```
Model: EfficientNet B0
Classification Report:
              precision    recall  f1-score   support

        BOLD       0.64      0.90      0.75        10
      BUBBLE       0.83      1.00      0.91        10
        BURR       1.00      0.10      0.18        10
      DAMAGE       0.75      0.30      0.43        10
         DOT       0.00      0.00      0.00        10
        DUST       0.33      1.00      0.50        10
        FOLD       0.43      0.60      0.50        10
        LINE       0.59      1.00      0.74        10
       REACT       0.91      1.00      0.95        10
        RING       0.00      0.00      0.00         1
     SCRATCH       0.60      0.30      0.40        10
         TIP       0.33      0.10      0.15        10

    accuracy                           0.57       111
   macro avg       0.53      0.53      0.46       111
weighted avg       0.58      0.57      0.50       111
```



## TRAIN_SET_R
### EfficientNetV2
#### lr 0.0001, batch 32, epoch 50 
```
Model: EfficientNetV2
Classification Report:
              precision    recall  f1-score   support

        BOLD       0.00      0.00      0.00        10
      BUBBLE       0.00      0.00      0.00        10
        BURR       0.09      1.00      0.17        10
      DAMAGE       0.00      0.00      0.00        10
         DOT       0.00      0.00      0.00        10
        DUST       0.00      0.00      0.00        10
        FOLD       0.00      0.00      0.00        10
        LINE       0.00      0.00      0.00        10
       REACT       0.00      0.00      0.00        10
        RING       0.00      0.00      0.00         1
     SCRATCH       0.00      0.00      0.00        10
         TIP       0.00      0.00      0.00        10

    accuracy                           0.09       111
   macro avg       0.01      0.08      0.01       111
weighted avg       0.01      0.09      0.01       111
```

#### lr 0.0001, batch 32, epoch 100 
-> same result

#### lr 0.0001, batch 8, epoch 50
-> same result

#### lr 0.0001, batch 8, epoch 100


### EfficientNetB0
#### lr 0.0001, batch 32, epoch 50 
```
Model: EfficientNet B0
Classification Report:
              precision    recall  f1-score   support

        BOLD       0.00      0.00      0.00        10
      BUBBLE       0.00      0.00      0.00        10
        BURR       0.00      0.00      0.00        10
      DAMAGE       0.00      0.00      0.00        10
         DOT       0.00      0.00      0.00        10
        DUST       0.00      0.00      0.00        10
        FOLD       0.00      0.00      0.00        10
        LINE       0.00      0.00      0.00        10
       REACT       0.00      0.00      0.00        10
        RING       0.01      1.00      0.02         1
     SCRATCH       0.00      0.00      0.00        10
         TIP       0.00      0.00      0.00        10

    accuracy                           0.01       111
   macro avg       0.00      0.08      0.00       111
weighted avg       0.00      0.01      0.00       111
```

#### lr 0.0001, batch 32, epoch 100 
-> same result

#### lr 0.0001, batch 8, epoch 50
```
Model: EfficientNet B0
Classification Report:
              precision    recall  f1-score   support

        BOLD       0.00      0.00      0.00        10
      BUBBLE       0.00      0.00      0.00        10
        BURR       0.00      0.00      0.00        10
      DAMAGE       0.00      0.00      0.00        10
         DOT       0.00      0.00      0.00        10
        DUST       0.00      0.00      0.00        10
        FOLD       0.09      1.00      0.17        10
        LINE       0.00      0.00      0.00        10
       REACT       0.00      0.00      0.00        10
        RING       0.00      0.00      0.00         1
     SCRATCH       0.00      0.00      0.00        10
         TIP       0.00      0.00      0.00        10

    accuracy                           0.09       111
   macro avg       0.01      0.08      0.01       111
weighted avg       0.01      0.09      0.01       111
```

#### lr 0.0001, batch 8, epoch 100
```
Model: EfficientNet B0
Classification Report:
              precision    recall  f1-score   support

        BOLD       0.44      0.80      0.57        10
      BUBBLE       0.00      0.00      0.00        10
        BURR       0.44      0.70      0.54        10
      DAMAGE       0.06      0.10      0.08        10
         DOT       0.29      0.40      0.33        10
        DUST       0.00      0.00      0.00        10
        FOLD       0.33      0.20      0.25        10
        LINE       1.00      0.30      0.46        10
       REACT       0.00      0.00      0.00        10
        RING       0.07      1.00      0.13         1
     SCRATCH       0.44      0.40      0.42        10
         TIP       0.22      0.20      0.21        10

    accuracy                           0.29       111
   macro avg       0.28      0.34      0.25       111
weighted avg       0.29      0.29      0.26       111
```

## TRAIN_SET_NR
### EfficientNetV2
#### lr 0.0001, batch 32, epoch 50 
```
Model: EfficientNetV2
Classification Report:
              precision    recall  f1-score   support

        BOLD       0.00      0.00      0.00        10
      BUBBLE       0.00      0.00      0.00        10
        BURR       0.00      0.00      0.00        10
      DAMAGE       0.00      0.00      0.00        10
         DOT       0.00      0.00      0.00        10
        DUST       0.00      0.00      0.00        10
        FOLD       0.09      1.00      0.17        10
        LINE       0.00      0.00      0.00        10
       REACT       0.00      0.00      0.00        10
        RING       0.00      0.00      0.00         1
     SCRATCH       0.00      0.00      0.00        10
         TIP       0.00      0.00      0.00        10

    accuracy                           0.09       111
   macro avg       0.01      0.08      0.01       111
weighted avg       0.01      0.09      0.01       111
```

#### lr 0.0001, batch 8, epoch 50 
```
Model: EfficientNetV2
Classification Report:
              precision    recall  f1-score   support

        BOLD       0.89      0.80      0.84        10
      BUBBLE       0.86      0.60      0.71        10
        BURR       0.50      0.30      0.38        10
      DAMAGE       0.50      0.10      0.17        10
         DOT       0.50      0.60      0.55        10
        DUST       0.47      0.80      0.59        10
        FOLD       0.41      0.90      0.56        10
        LINE       0.71      0.50      0.59        10
       REACT       0.73      0.80      0.76        10
        RING       1.00      1.00      1.00         1
     SCRATCH       0.20      0.10      0.13        10
         TIP       0.25      0.30      0.27        10

    accuracy                           0.53       111
   macro avg       0.58      0.57      0.55       111
weighted avg       0.55      0.53      0.51       111
```

#### lr 0.0001, batch 8, epoch 100 
```
Model: EfficientNetV2
Classification Report:
              precision    recall  f1-score   support

        BOLD       0.56      0.50      0.53        10
      BUBBLE       0.86      0.60      0.71        10
        BURR       0.31      0.50      0.38        10
      DAMAGE       0.12      0.10      0.11        10
         DOT       0.44      0.80      0.57        10
        DUST       0.25      0.10      0.14        10
        FOLD       0.47      0.70      0.56        10
        LINE       0.43      0.30      0.35        10
       REACT       0.64      0.70      0.67        10
        RING       1.00      1.00      1.00         1
     SCRATCH       0.38      0.30      0.33        10
         TIP       0.43      0.30      0.35        10

    accuracy                           0.45       111
   macro avg       0.49      0.49      0.48       111
weighted avg       0.45      0.45      0.43       111
```

### EfficientNetB0
#### lr 0.0001, batch 32, epoch 50 
```
Model: EfficientNet B0
Classification Report:
              precision    recall  f1-score   support

        BOLD       0.75      0.30      0.43        10
      BUBBLE       0.86      0.60      0.71        10
        BURR       0.31      0.40      0.35        10
      DAMAGE       0.30      0.30      0.30        10
         DOT       0.43      0.60      0.50        10
        DUST       0.33      0.10      0.15        10
        FOLD       0.50      0.40      0.44        10
        LINE       0.57      0.80      0.67        10
       REACT       0.53      1.00      0.69        10
        RING       1.00      1.00      1.00         1
     SCRATCH       0.00      0.00      0.00        10
         TIP       0.33      0.60      0.43        10

    accuracy                           0.47       111
   macro avg       0.49      0.51      0.47       111
weighted avg       0.45      0.47      0.43       111
```

#### lr 0.0001, batch 8, epoch 50 
```
Model: EfficientNet B0
Classification Report:
              precision    recall  f1-score   support

        BOLD       0.20      0.20      0.20        10
      BUBBLE       0.60      0.30      0.40        10
        BURR       0.40      0.60      0.48        10
      DAMAGE       0.50      0.40      0.44        10
         DOT       0.39      0.90      0.55        10
        DUST       0.00      0.00      0.00        10
        FOLD       0.18      0.20      0.19        10
        LINE       0.50      0.40      0.44        10
       REACT       0.60      0.90      0.72        10
        RING       0.00      0.00      0.00         1
     SCRATCH       0.50      0.10      0.17        10
         TIP       0.08      0.10      0.09        10

    accuracy                           0.37       111
   macro avg       0.33      0.34      0.31       111
weighted avg       0.36      0.37      0.33       111
```

#### lr 0.0001, batch 8, epoch 100 
```
Model: EfficientNet B0
Classification Report:
              precision    recall  f1-score   support

        BOLD       0.62      0.50      0.56        10
      BUBBLE       1.00      0.70      0.82        10
        BURR       0.47      0.70      0.56        10
      DAMAGE       0.75      0.30      0.43        10
         DOT       0.75      0.60      0.67        10
        DUST       0.40      0.60      0.48        10
        FOLD       0.25      0.40      0.31        10
        LINE       0.83      0.50      0.62        10
       REACT       0.56      0.90      0.69        10
        RING       0.00      0.00      0.00         1
     SCRATCH       0.25      0.10      0.14        10
         TIP       0.25      0.30      0.27        10

    accuracy                           0.50       111
   macro avg       0.51      0.47      0.46       111
weighted avg       0.55      0.50      0.50       111
```



## TRAIN_SET_HL
### EfficientNetV2
#### lr 0.0001, batch 32, epoch 50 
```
Model: EfficientNetV2
```


```
Model: EfficientNetV2
```


```
Model: EfficientNetV2
```



### EfficientNetB0
#### lr 0.0001, batch 32, epoch 50 
```
Model: EfficientNet B0
```


```
Model: EfficientNet B0
```


```
Model: EfficientNet B0
```

### EfficientNetV2
#### lr 0.0001, batch 32, epoch 50 
```
Model: EfficientNetV2
```


```
Model: EfficientNetV2
```


```
Model: EfficientNetV2
```



### EfficientNetB0
#### lr 0.0001, batch 32, epoch 50 
```
Model: EfficientNet B0
```


```
Model: EfficientNet B0
```


```
Model: EfficientNet B0
```




### EfficientNetV2
#### lr 0.0001, batch 32, epoch 50 
```
Model: EfficientNetV2
```


```
Model: EfficientNetV2
```


```
Model: EfficientNetV2
```



### EfficientNetB0
#### lr 0.0001, batch 32, epoch 50 
```
Model: EfficientNet B0
```


```
Model: EfficientNet B0
```


```
Model: EfficientNet B0
```






### EfficientNetV2
#### lr 0.0001, batch 32, epoch 50 
```
Model: EfficientNetV2
```


```
Model: EfficientNetV2
```


```
Model: EfficientNetV2
```



### EfficientNetB0
#### lr 0.0001, batch 32, epoch 50 
```
Model: EfficientNet B0
```


```
Model: EfficientNet B0
```


```
Model: EfficientNet B0
```






### EfficientNetV2
#### lr 0.0001, batch 32, epoch 50 
```
Model: EfficientNetV2
```


```
Model: EfficientNetV2
```


```
Model: EfficientNetV2
```



### EfficientNetB0
#### lr 0.0001, batch 32, epoch 50 
```
Model: EfficientNet B0
```


```
Model: EfficientNet B0
```


```
Model: EfficientNet B0
```




### EfficientNetV2
#### lr 0.0001, batch 32, epoch 50 
```
Model: EfficientNetV2
```


```
Model: EfficientNetV2
```


```
Model: EfficientNetV2
```



### EfficientNetB0
#### lr 0.0001, batch 32, epoch 50 
```
Model: EfficientNet B0
```


```
Model: EfficientNet B0
```


```
Model: EfficientNet B0
```
