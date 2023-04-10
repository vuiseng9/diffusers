# Setup (ldm-profile)

### install diffuser
```bash
git clone https://github.com/vuiseng9/diffusers
cd diffusers
git remote add upstream https://github.com/huggingface/diffusers
git checkout ldm-profile-by-scope
pip install -e .[torch]
```

### install transformer
```bash
git clone https://github.com/vuiseng9/transformers
cd transformers
git remote add upstream https://github.com/huggingface/transformers
git checkout v4.27.4 -b v4.27.4
pip install -e .
```

### Sample profiling
```
cd diffusers
python3 profile_text2img.py 2>&1 | tee -a log.text2img-sdv1.5-diffstep20-res512
```