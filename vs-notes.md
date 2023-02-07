# Setup (ldm-profile)

### install diffuser
```bash
git clone https://github.com/vuiseng9/diffusers
cd diffusers
git checkout ldm-profile
pip install -e .[torch]
```

### install transformer
```bash
git clone https://github.com/vuiseng9/transformers
cd transformers
git checkout v4.24-ldm-profile
pip install -e .
```

### other dependency required for stable diffusion
```bash
pip install accelerate scipy ftfy
```

### tools for benchmark/analysis
```bash
git clone https://github.com/vuiseng9/torchinfo
cd torchinfo
git checkout ldm-profile
pip install -e .
```

# Usages
```bash
python3 diffusers/examples/text_to_image/stable_diffusion.py
# this should report the latency of major stages in inference and for each function wrapped with timeit
```
To print out model architecture summary,
set `DUMP_MODELSUMMARY = True` in `torchinfo/torchinfo/benchutils.py`

Do play with summary arguments (e.g. depth, col_width) to get desired breakdown
```python
summary(self.unet, 
        input_data=[latent_model_input, t, text_embeddings], 
        depth=4, 
        col_width=33,
        col_names=("mult_adds", "input_size", "output_size", "num_params"))
```
