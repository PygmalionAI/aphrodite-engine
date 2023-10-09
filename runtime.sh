#!/bin/bash
if [ ! -f "conda/envs/linux/bin/python" ]; then
./update-runtime.sh
fi
if [ $# -eq 0 ]
  then
    bin/micromamba run -r conda -n linux bash
    exit
fi
bin/micromamba run -r conda -n linux $*
