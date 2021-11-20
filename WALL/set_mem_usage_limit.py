# coding: utf-8
"""
author: Jet Chien
GitHub: https://github.com/jet-c-21
Create Date: 11/20/21
"""

import resource
import platform
import sys


def memory_limit():
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    # set usage to half
    resource.setrlimit(resource.RLIMIT_AS, (get_memory() * 1024 / 2, hard))


def get_memory():
    with open('/proc/meminfo', 'r') as mem:
        free_memory = 0
        for i in mem:
            s_line = i.split()
            if str(s_line[0]) in ('MemFree:', 'Buffers:', 'Cached:'):
                free_memory += int(s_line[1])
    return free_memory


def main():
    pass


if __name__ == '__main__':
    memory_limit()
    try:
        main()
    except MemoryError:
        sys.stderr.write('\n\nERROR: Memory Exception\n')
        sys.exit(1)
