"""
author: Jet Chien
GitHub: https://github.com/jet-c-21
Create Date: 11/19/21
"""
# coding: utf-8
from collections import Counter
import pandas as pd
import numpy as np
from string import ascii_uppercase

d = dict()
for i, v in enumerate(list(ascii_uppercase)):
    d[v] = i

classes = np.array(list(d))
print(classes)


