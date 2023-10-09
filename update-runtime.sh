#!/bin/bash

wget -qO- https://micromamba.snakepit.net/api/micromamba/linux-64/latest | tar -xvj bin/micromamba
if [ ! -f "conda/envs/linux/bin/python" ]; then
 bin/micromamba create --no-shortcuts -r conda -n linux -f environment.yaml -y
fi
bin/micromamba create --no-shortcuts -r conda -n linux -f environment.yaml -y
bin/micromamba install -r conda -n linux gxx=10 -c conda-forge -y
bin/micromamba run -r conda -n linux pip install -r requirements.txt
# Make it so the correct NVCC is found. Will do a quick search that shouldn't take more than a few seconds and get the first result.
export CUDA_HOME=$(find / -type d -path "*/conda/envs/linux" 2>/dev/null | head -n 1)
read -p "Do you want to install aphrodite from source? (y/n): " answer
case ${answer:0:1} in
    y|Y )
        bin/micromamba run -r conda -n linux pip install -e .
    ;;
    * )
        bin/micromamba run -r conda -n linux pip install aphrodite-engine
    ;;
esac

