# coding: utf-8
"""
author: Jet Chien
GitHub: https://github.com/jet-c-21
Create Date: 11/22/21
"""
import asl_model
from inspect import getmembers, isfunction
from pprint import pprint as pp
from asl_model import get_stn_a_model_1
from asl_model import tool

# for f in getmembers(asl_model, isfunction):
#     if f[0] in ['stn']:
#         continue
#
#     print(f"func name : {f[0]}")
#     model = f[1]()
#     model.summary()

for model_func_name, model_func_obj in tool.all_models.items():
    pass