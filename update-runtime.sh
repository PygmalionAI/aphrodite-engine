#!/bin/bash

wget -qO- https://micromamba.snakepit.net/api/micromamba/linux-64/latest | tar -xvj bin/micromamba
if [ ! -f "conda/envs/linux/bin/python" ]; then
 bin/micromamba create --no-shortcuts -r conda -n linux -f environment.yaml -y
fi
bin/micromamba create --no-shortcuts -r conda -n linux -f environment.yaml -y
bin/micromamba install -r conda -n linux gxx=10 -c conda-forge -y
bin/micromamba run -r conda -n linux pip install -r requirements.txt
read -p "Do you want to install aphrodite from source? (y/n): " answer
case ${answer:0:1} in
    y|Y )
        bin/micromamba run -r conda -n linux pip install -e .
    ;;
    * )
        bin/micromamba run -r conda -n linux pip install aphrodite-engine
    ;;
esac
