#!/bin/bash
if [ ! -f "conda/envs/aphrodite-runtime/bin/python" ]; then
./update-runtime.sh
fi
if [ $# -eq 0 ]
  then
    bin/micromamba run -r conda -n aphrodite-runtime bash
    exit
fi
bin/micromamba run -r conda -n aphrodite-runtime $*
