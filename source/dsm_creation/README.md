## Note about dependency-based DSMs:

Line 76 in `create_dep_based_cooc_file.py` was changed from:

```
parents = [token for token in sentence if token[-2] == parent]
```

to:

```
parents = [token for token in sentence if token[-3] == parent]
```

The implementation for which the paper reports results erroneously included as contexts a word's siblings instead of its parents and children. 
