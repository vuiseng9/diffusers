


import torch
from diffusers import StableDiffusionPipeline
from functools import wraps
import time
from functools import partial
from collections import OrderedDict, defaultdict
import numpy as np

pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipe = pipe.to("cuda")

# pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float32)
# pipe = pipe.to("cuda")

# pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float32)
# pipe = pipe.to("cpu")
torch.nn.ModuleList.register_forward_hook

def annotate_module_static_attr(top_module, family_name=None):
    # static attr: 
    # first_name, last_name, class_name, is_leaf_module, leaf_has_weight
    if family_name is None:
        family = top_module.__class__.__name__.lower() + "class_as_family_name"
    else:
        family = family_name

    for parent_name, parent_module in top_module.named_modules():
        # handle top level because children loop below operate one level below, top level module will be missed 
        if parent_name == "":
            parent_module.first_name = family
            parent_module.last_name = ""

        for child_name, child_module in parent_module.named_children():
            child_module.first_name = child_name
            if parent_name == "":
                # just to handle the period if we dont do this conditional loop
                child_module.last_name = f"{family}"
            else:
                child_module.last_name = f"{family}.{parent_name}"
            
        # Following applies to every module
        parent_module.leaf_module = False
        if len(list(parent_module.children())) == 0:
            parent_module.is_leaf_module = True
            parent_module.leaf_has_weight = False
            if len(list(parent_module.parameters())) > 0:
                parent_module.leaf_has_weight = True

        parent_module.class_name = parent_module.__class__.__name__
        parent_module.full_name = f"{parent_module.last_name}.{parent_module.first_name}" # must be put at last

def register_timeit_hook(module, tracker_dict: OrderedDict):
    def pre_hook(module, args, registry):
        registry[module.full_name]['ts_start'].append(time.perf_counter())
    
    def post_hook(module, args, output, registry):
        registry[module.full_name]['ts_end'].append(time.perf_counter())
    
    tracker_dict[module.full_name] = defaultdict(list)
    tracker_dict[module.full_name]['ts_start'] = []
    tracker_dict[module.full_name]['ts_end'] = []

    prehk = module.register_forward_pre_hook(partial(pre_hook, registry=tracker_dict))
    posthk = module.register_forward_hook(partial(post_hook, registry=tracker_dict))
    
    return prehk, posthk

# # sample usage of timeit hook
# annotate_module_static_attr(top_module=pipe.unet, family_name="unet")
#
# logdict = OrderedDict()
#
# prehook, posthook = register_timeit_hook(pipe.unet.mid_block, logdict)
# 
# prompt = "a photo of an astronaut riding a horse on mars"
# image = pipe(prompt).images[0]
#
# prehook.remove()
# posthook.remove()
#
# prompt = "a photo of an astronaut riding a horse on mars"
# image = pipe(prompt).images[0]
# for i, (s, e) in enumerate(zip(logdict['unet.mid_block']['ts_start'],logdict['unet.mid_block']['ts_end'])):
#   print(i, '{:.4f}'.format(e-s))

top_module=pipe.unet
annotate_module_static_attr(top_module=top_module, family_name="unet")

modtype_to_modlist = defaultdict(list)
modname_to_modtype = OrderedDict()
modname_to_module = OrderedDict()

for n, m in top_module.named_modules():
    modtype_to_modlist[m.class_name].append(f"{m.last_name}.{m.first_name}")
    modname_to_modtype[m.full_name] = m.class_name
    modname_to_module[m.full_name] = m

# following function dependent on lookup above
def wrap_timeit_by_namelist(namelist, logdict):
    prehook_list=[]
    posthook_list=[]

    for modname in namelist:
        print(f"Registering timeit hooks to: {modname}")
        prehook, posthook = register_timeit_hook(modname_to_module[modname], logdict)
        prehook_list.append(prehook)
        posthook_list.append(posthook)
    return prehook_list, posthook_list

def wrap_timeit_by_modtypelist(modtypelist, logdict):
    prehook_list=[]
    posthook_list=[]

    for modtype in modtypelist:
        for modname in modtype_to_modlist[modtype]:
            print(f"Registering timeit hooks to: {modname}")
            prehook, posthook = register_timeit_hook(modname_to_module[modname], logdict)
            prehook_list.append(prehook)
            posthook_list.append(posthook)
    return prehook_list, posthook_list

# for k, v in modtype_to_modlist.items():
#     print(v[0], k)

# for k, v in modtype_to_modlist.items():
#     if k == "ModuleList":
#         print('\n'.join(v))

# ["unet.down_blocks", "unet.up_blocks", "unet.mid_block"]
# ["CrossAttnDownBlock2D", "DownBlock2D", "UpBlock2D", "UNetMidBlock2DCrossAttn", "CrossAttnUpBlock2D"]
# Transformer2DModel
# ResnetBlock2D

timedict = OrderedDict()

superblock_list = ["unet.down_blocks", "unet.up_blocks", "unet.mid_block"]
# hook_tuple = wrap_timeit_by_namelist(superblock_list, timedict)

# majorblock_list = ["CrossAttnDownBlock2D", "DownBlock2D", "UpBlock2D", "UNetMidBlock2DCrossAttn", "CrossAttnUpBlock2D"]
# hook_tuple = wrap_timeit_by_modtypelist(majorblock_list, timedict)

keyblock_list = ["Transformer2DModel", "ResnetBlock2D"]
hook_tuple = wrap_timeit_by_modtypelist(keyblock_list, timedict)


prompt = "a photo of an astronaut riding a horse on mars"
image = pipe(prompt).images[0]
image.save("astronaut.png",'PNG')

summary_delta = OrderedDict()

for k, v in timedict.items():
    for i, (s, e) in enumerate(zip(v['ts_start'], v['ts_end'])):
        if k not in summary_delta:
            summary_delta[k] = {'history': [e-s]}
        else:
            summary_delta[k]['history'].append(e-s)

for k, v in summary_delta.items():
    v['mean'] = np.array(v['history']).mean()
    v['sum'] = np.array(v['history']).sum()
    print(f"sum: {v['sum']:.4f} | mean: {v['mean']:.4f} | {k}")

print("End of script.")