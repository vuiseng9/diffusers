# dev setup

# install diffuser
git clone https://github.com/vuiseng9/diffusers
cd diffusers
git checkout -b local-dev
pip install -e .[torch]

# install transformer
git clone https://github.com/vuiseng9/transformers
cd transformers
git checkout -b v4.24-release
pip install -e .

# other dependency required for stable diffusion
pip install accelerate scipy ftfy
