"""
author: Jet Chien
GitHub: https://github.com/jet-c-21
Create Date: 11/19/21
"""


# coding: utf-8

def get_proper_test_sample_count(train_sample_count: int, test_proportion=0.2):
    """
    x / (train_sample_count + x) = test_proportion

    :param train_sample_count:
    :param test_proportion:
    :return:
    """
    return test_proportion * train_sample_count / (1 - test_proportion)


if __name__ == '__main__':
    y = get_proper_test_sample_count(500, 0.2)
    print(y)
