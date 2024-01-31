import math
import re
from pathlib import Path

import numpy as np

# From https://en.wikipedia.org/wiki/Paley_construction (construction II for q = 5)

had_12_paley = """
+-++++++++++
--+-+-+-+-+-
+++-++----++
+---+--+-++-
+++++-++----
+-+---+--+-+
++--+++-++--
+--++---+--+
++----+++-++
+--+-++---+-
++++----+++-
+-+--+-++---
""" 

# From http://neilsloane.com/hadamard/

had_20_will = """
+----+----++--++-++-
-+----+---+++---+-++
--+----+---+++-+-+-+
---+----+---+++++-+-
----+----++--++-++-+
-+++++-----+--+++--+
+-+++-+---+-+--+++--
++-++--+---+-+--+++-
+++-+---+---+-+--+++
++++-----++--+-+--++
--++-+-++-+-----++++
---++-+-++-+---+-+++
+---++-+-+--+--++-++
++---++-+----+-+++-+
-++---++-+----+++++-
-+--+--++-+----+----
+-+-----++-+----+---
-+-+-+---+--+----+--
--+-+++------+----+-
+--+--++------+----+
"""


had_28_will = """
+------++----++-+--+-+--++--
-+-----+++-----+-+--+-+--++-
--+-----+++---+-+-+----+--++
---+-----+++---+-+-+-+--+--+
----+-----+++---+-+-+++--+--
-----+-----++++--+-+--++--+-
------++----++-+--+-+--++--+
--++++-+-------++--+++-+--+-
---++++-+-----+-++--+-+-+--+
+---+++--+----++-++--+-+-+--
++---++---+----++-++--+-+-+-
+++---+----+----++-++--+-+-+
++++--------+-+--++-++--+-+-
-++++--------+++--++--+--+-+
-+-++-++--++--+--------++++-
+-+-++--+--++--+--------++++
-+-+-++--+--++--+----+---+++
+-+-+-++--+--+---+---++---++
++-+-+-++--+------+--+++---+
-++-+-+-++--+------+-++++---
+-++-+---++--+------+-++++--
-++--++-+-++-+++----++------
+-++--++-+-++-+++-----+-----
++-++---+-+-++-+++-----+----
-++-++-+-+-+-+--+++-----+---
--++-++++-+-+----+++-----+--
+--++-+-++-+-+----+++-----+-
++--++-+-++-+-+----++------+
"""

header = """
/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

// This file is auto-generated. See "generator.py"\n

#pragma once

"""

template = """
__device__ __forceinline__ void hadamard_mult_thread_{N}(float x[{N}]) {
    float out[{N}];
    {code}
    #pragma unroll
    for (int i = 0; i < {N}; i++) { x[i] = out[i]; }
}

"""


def string_to_array(string):
    # Convert strings of + and - to bool arrays
    string = string.strip().replace('+', '1').replace('-', '-1').split()
    return np.stack([np.fromstring(" ".join(string[i]), dtype=np.int32, sep=' ') for i in range(len(string))])


def array_code_gen(arr):
    N = arr.shape[0]
    assert arr.shape[0] == arr.shape[1]
    out = []
    for i in range(N):
        out.append(f"out[{i}] = " + " ".join([f"{'+' if arr[i, j] == 1 else '-'} x[{j}]" for j in range(N)]) + ";")
    return template.replace("{N}", str(N)).replace("{code}", '\n    '.join(out))



def main():
    output_dir = Path(__file__).parent / "fast_hadamard_transform_special.h"
    output_dir.write_text(header + array_code_gen(string_to_array(had_12_paley)) + array_code_gen(string_to_array(had_20_will)) + array_code_gen(string_to_array(had_28_will)))

if __name__ == '__main__':
    main()
