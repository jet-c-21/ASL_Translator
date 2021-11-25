# coding: utf-8
"""
author: Jet Chien
GitHub: https://github.com/jet-c-21
Create Date: 11/24/21
"""
import numpy as np
from string import ascii_uppercase

arr = np.array([0, 1, 2])
# arr = np.apply_along_axis(lambda x: np.array(list(ascii_uppercase))[x], 0, arr)
# print(arr)

arr = [ascii_uppercase[i] for i in arr]
print(arr)

# z = np.array(list(ascii_uppercase))
# print(z)
