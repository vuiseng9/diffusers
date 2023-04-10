


import torch
from diffusers import StableDiffusionPipeline
from functools import wraps
import time
from functools import partial
from collections import OrderedDict, defaultdict
import numpy as np
import pandas as pd
import os

pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16).to("cuda")

def print_divider():
    print('-'*100)

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

def remove_hook(hook_tuple):
    for hooklist in hook_tuple:
        for hook in hooklist:
            hook.remove()

def summarize_timedict(timedict, pipe_mean, nloop=10):
    summary_delta = OrderedDict()

    for k, v in timedict.items():
        for i, (s, e) in enumerate(zip(v['ts_start'], v['ts_end'])):
            if k not in summary_delta:
                summary_delta[k] = {'history': [e-s]}
            else:
                summary_delta[k]['history'].append(e-s)

    for k, v in summary_delta.items():
        v['count'] = len(v['history']) #  count here is by entry, i.e. number of generation * expected entry of a hook
        v['scope_mean'] = np.array(v['history']).sum()/nloop # average time spent over number of generation, total timestep of denoising affect outcome 
        v['module_type'] = modname_to_modtype[k]
        print(f"count: {v['count']:3} | scope_mean: {v['scope_mean']:.4f} | {v['module_type']} | {k}")
    
    df = pd.DataFrame.from_dict(summary_delta).T
    df=df[['module_type', 'scope_mean']]
    top_scope_mean = df.scope_mean['.unet'] # hardcoding
    df = df.drop(['.unet']) #hardcoding
    df.loc["others"] = ['misc', top_scope_mean-df['scope_mean'].sum()] # nloop must be carefully checked to align func
    df['scope_percentage'] = df['scope_mean']/top_scope_mean*100
    df.reset_index(inplace=True)
    df = df.rename(columns = {'index':'scope'})
    return df, top_scope_mean, pipe_mean

# for k, v in modtype_to_modlist.items():
#     print(v[0], k)
# for k, v in modtype_to_modlist.items():
#     if k == "ModuleList":
#         print('\n'.join(v))

# ["unet.down_blocks", "unet.up_blocks", "unet.mid_block"]
# ["CrossAttnDownBlock2D", "DownBlock2D", "UpBlock2D", "UNetMidBlock2DCrossAttn", "CrossAttnUpBlock2D"]
# [Transformer2DModel]
# [ResnetBlock2D]
ROOTDIR=os.path.dirname(os.path.abspath(__file__))
WORKDIR=os.path.join(ROOTDIR, "text2img-sdv1.5-diffstep20-res512")
os.makedirs(WORKDIR, exist_ok=True)

# this function require access to WORKDIR
def postprocess(df, top_mean_latency, pipe_mean_latency, label_csv):
    if label_csv is None:
        raise ValueError("label_csv is None")

    print(f"pipe elapsed time/ses ({label_csv}): {pipe_mean_latency:.3f}")
    print(f"unet elapsed time/sec | % ({label_csv}): {top_mean_latency:.3f} | {top_mean_latency/pipe_mean_latency*100:.2f}")
    print(df)
    df.to_csv(os.path.join(WORKDIR, f'{label_csv}.csv'))


prompt = "a photo of an astronaut riding a horse on mars"
def elapsed_time(pipeline, nb_pass=10, num_inference_steps=20):
    start = time.time()
    for _ in range(nb_pass):
        image = pipeline(prompt, num_inference_steps=num_inference_steps)
    end = time.time()
    return (end - start) / nb_pass, image.images[0]

# warmup
elapsed_time(pipe)

print_divider()
time_original_model, image = elapsed_time(pipe)
image.save(f"{WORKDIR}/pipe.0.png", "PNG")
actual_dtype = next(pipe.unet.parameters()).dtype
actual_device = next(pipe.unet.parameters()).device
print(f"pipe elapsed time: {time_original_model:.3f} | {actual_device} | {actual_dtype}")
print_divider()

def profile_by_block(block_type_list, timedict, top_module, pipe,  label):
    hook_tuple = wrap_timeit_by_modtypelist(block_type_list, timedict)
    top_module_hook = wrap_timeit_by_modtypelist([top_module.__class__.__name__], timedict)

    # profiling loop
    block_elapse, image = elapsed_time(pipe)
    image.save(f"{WORKDIR}/{label}.png", "PNG")
    df, top_mean_latency, pipe_mean_latency = summarize_timedict(timedict, block_elapse)
    print_divider()
    postprocess(df, top_mean_latency, pipe_mean_latency, label)
    remove_hook(hook_tuple)
    remove_hook(top_module_hook)
    print_divider()
    return df

# ModuleList forward are not called!
# superblock_list = ["unet.down_blocks", "unet.up_blocks", "unet.mid_block"]
# hook_tuple = wrap_timeit_by_namelist(superblock_list, timedict)

# -----------------------------
majorblock_list = ["CrossAttnDownBlock2D", "DownBlock2D", "UpBlock2D", "UNetMidBlock2DCrossAttn", "CrossAttnUpBlock2D"]
majorblock_timedict = OrderedDict()
df_majorblock = profile_by_block(majorblock_list, majorblock_timedict, top_module, pipe, "majorblock_latency")
# -----------------------------
resblock_type = ["ResnetBlock2D"]
resblock_timedict = OrderedDict()
df_resblock = profile_by_block(resblock_type, resblock_timedict, top_module, pipe, "resblock_latency")
# -----------------------------
txblock_type = ["Transformer2DModel"]
txblock_timedict = OrderedDict()
df_txblock = profile_by_block(txblock_type, txblock_timedict, top_module, pipe, "txblock_latency")
# -----------------------------
conv2d_type = ["Conv2d"]
conv2d_timedict = OrderedDict()
df_conv2d = profile_by_block(conv2d_type, conv2d_timedict, top_module, pipe, "conv2d_latency")
# -----------------------------
linear_type = ["Linear"]
linear_timedict = OrderedDict()
df_linear = profile_by_block(linear_type, linear_timedict, top_module, pipe, "linear_latency")
# -----------------------------
print("End of script.")