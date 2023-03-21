


import torch
from diffusers import StableDiffusionPipeline
from torchinfo.benchutils import report_timeit_stats

from functools import partial
from collections import OrderedDict

pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipe = pipe.to("cuda")


global order_id
global leaf_layer_list
leaf_layer_list=[]
order_id = 0

def tensor_size(t):
    # convert tensor shape to list
    return list(t.shape)

def summarize_tensor(t):
    return list(t.shape), t.numel()

def get_hook(label):
    def summarize_leaf_layer(module, input, output, label):
        global leaf_layer_list
        global order_id 

        assert len(input) == 1, "Requires new input handling"
        assert isinstance(output, torch.Tensor), "Requires new output handling"

        d = {}
        d['forward_order'] = order_id
        d['torch_name'] = label
        d['module_type'] = module.__class__.__name__
        d['ifm_shape'], d['ifm_numel'] = summarize_tensor(input[0])
        d['weight_shape'], d['weight_numel'] = summarize_tensor(module.weight)
        d['ofm_shape'], d['ofm_numel'] = summarize_tensor(output)
        d['module_str'] = str(module)

        leaf_layer_list.append(d)
        order_id += 1

    return partial(summarize_leaf_layer, label=label)

hook_list =[]
for n, m in pipe.unet.named_modules():
    if len(list(m.children())) == 0 and hasattr(m, 'weight'):
        hook_list.append(
            m.register_forward_hook(get_hook(n))
        )

# Duck taping approach - not recommended for production, for our own analysis
# so that we can access internally
pipe.leaf_layer_list = leaf_layer_list
pipe.dump_leaf_csv = True

prompt = "a photo of an astronaut riding a horse on mars"
image = pipe(prompt).images[0]  

report_timeit_stats()
print("End of script.")