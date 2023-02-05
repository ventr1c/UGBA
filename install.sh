conda install pytorch==1.12.1 torchvision=0.13.1 torchaudio=0.12.1 pytorch-cuda=11.3 -c pytorch -c nvidia
pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.12.0+cu113.html
pip install scikit-learn-extra