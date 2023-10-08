#!/bin/bash

wget -qO- https://micromamba.snakepit.net/api/micromamba/linux-64/latest | tar -xvj bin/micromamba
if [ ! -f "conda/envs/linux/bin/python" ]; then
 bin/micromamba create --no-shortcuts -r conda -n linux -f environment.yaml -y
fi
bin/micromamba create --no-shortcuts -r conda -n linux -f environment.yaml -y
bin/micromamba install -r conda -n linux gxx=10 -c conda-forge -y
bin/micromamba run -r conda -n linux pip install -r requirements.txt
bin/micromamba run -r conda -n linux pip install -e .
