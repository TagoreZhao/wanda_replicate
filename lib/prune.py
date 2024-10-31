import time 
import heapq 
import torch 
import torch.nn as nn 
from .sparsegpt import SparseGPT 
from .layerwrapper import WrappedGPT
from .data import get_loaders 

from .ablate import AblateGPT 

def find_layers(module, layers=[nn.Linear], name=''):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

def check_sparsity(model):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    layers = model.model.layers
    count = 0 
    total_params = 0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        sub_count = 0
        sub_params = 0
        for name in subset:
            W = subset[name].weight.data
            count += (W==0).sum().item()
            total_params += W.numel()

            sub_count += (W==0).sum().item()
            sub_params += W.numel()

        print(f"layer {i} sparsity {float(sub_count)/sub_params:.6f}")

    model.config.use_cache = use_cache 
    return float(count)/total_params 

# def prepare_calibration_input(model, dataloader, device):
#     use_cache = model.config.use_cache
#     model.config.use_cache = False
#     layers = model.model.layers

#     # dev = model.hf_device_map["model.embed_tokens"]
#     if "model.embed_tokens" in model.hf_device_map:
#         device = model.hf_device_map["model.embed_tokens"]

#     dtype = next(iter(model.parameters())).dtype
#     inps = torch.zeros((128, model.seqlen, model.config.hidden_size), dtype=dtype, device=device)
#     inps.requires_grad = False
#     cache = {'i': 0, 'attention_mask': None, "position_ids": None}

#     class Catcher(nn.Module):
#         def __init__(self, module):
#             super().__init__()
#             self.module = module
#         def forward(self, inp, **kwargs):
#             inps[cache['i']] = inp
#             cache['i'] += 1
#             cache['attention_mask'] = kwargs['attention_mask']
#             cache['position_ids'] = kwargs['position_ids']
#             raise ValueError
#     layers[0] = Catcher(layers[0])
#     for batch in dataloader:
#         try:
#             model(batch[0].to(device))
#         except ValueError:
#             pass 
#     layers[0] = layers[0].module

#     outs = torch.zeros_like(inps)
#     attention_mask = cache['attention_mask']
#     position_ids = cache['position_ids']
#     model.config.use_cache = use_cache

#     return inps, outs, attention_mask, position_ids 

def prepare_calibration_input(model, dataloader, device):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    # Set the appropriate device if "embed_tokens" is mapped
    if "model.embed_tokens" in model.hf_device_map:
        device = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps_cpu = torch.zeros((128, model.seqlen, model.config.hidden_size), dtype=dtype, device="cpu")
    outs_cpu = torch.zeros((128, model.seqlen, model.config.hidden_size), dtype=dtype, device="cpu")
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps_cpu[cache['i']] = inp.cpu()  # Store on CPU to save GPU memory
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask'].cpu()
            cache['position_ids'] = kwargs['position_ids'].cpu()
            raise ValueError

    # Wrap the first layer to intercept input/output
    layers[0] = Catcher(layers[0])

    for batch in dataloader:
        try:
            batch_gpu = batch[0].to(device)
            model(batch_gpu)
        except ValueError:
            pass
        torch.cuda.empty_cache()  # Clear GPU cache after each batch

    # Restore the original model layer
    layers[0] = layers[0].module
    model.config.use_cache = use_cache

    # Move only necessary data to GPU in a batched manner later in the pipeline
    return inps_cpu, outs_cpu, cache['attention_mask'], cache['position_ids']


def return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before):
    thres_cumsum = sum_before * alpha 
    sort_mask = tmp_metric <= thres_cumsum.reshape((-1,1))
    thres = torch.gather(sort_res[0], dim=1, index=sort_mask.sum(dim=1, keepdims=True)-1)
    W_mask = (W_metric <= thres)
    cur_sparsity = (W_mask==True).sum() / W_mask.numel()
    return W_mask, cur_sparsity

def prune_magnitude(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    layers = model.model.layers 

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        for name in subset:
            W = subset[name].weight.data 
            W_metric = torch.abs(W)
            if prune_n != 0:
                W_mask = (torch.zeros_like(W)==1)
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:,ii:(ii+prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
            else:
                thresh = torch.sort(W_metric.flatten().cuda())[0][int(W.numel()*args.sparsity_ratio)].cpu()
                W_mask = (W_metric<=thresh)

            W[W_mask] = 0

# def prune_wanda(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
#     use_cache = model.config.use_cache 
#     model.config.use_cache = False 

#     print("loading calibdation data")
#     dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)
#     print("dataset loading complete")
#     with torch.no_grad():
#         inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device)

#     layers = model.model.layers
#     for i in range(len(layers)):
#         layer = layers[i]
#         subset = find_layers(layer)

#         if f"model.layers.{i}" in model.hf_device_map:   ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
#             dev = model.hf_device_map[f"model.layers.{i}"]
#             inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

#         wrapped_layers = {}
#         for name in subset:
#             wrapped_layers[name] = WrappedGPT(subset[name])

#         def add_batch(name):
#             def tmp(_, inp, out):
#                 wrapped_layers[name].add_batch(inp[0].data, out.data)
#             return tmp

#         handles = []
#         for name in wrapped_layers:
#             handles.append(subset[name].register_forward_hook(add_batch(name)))
#         for j in range(args.nsamples):
#             with torch.no_grad():
#                 outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
#         for h in handles:
#             h.remove()

#         for name in subset:
#             print(f"pruning layer {i} name {name}")
#             W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))

#             W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False
#             if prune_n != 0:
#                 # structured n:m sparsity
#                 for ii in range(W_metric.shape[1]):
#                     if ii % prune_m == 0:
#                         tmp = W_metric[:,ii:(ii+prune_m)].float()
#                         W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
#             else:
#                 sort_res = torch.sort(W_metric, dim=-1, stable=True)

#                 if args.use_variant:
#                     # wanda variant 
#                     tmp_metric = torch.cumsum(sort_res[0], dim=1)
#                     sum_before = W_metric.sum(dim=1)

#                     alpha = 0.4
#                     alpha_hist = [0., 0.8]
#                     W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
#                     while (torch.abs(cur_sparsity - args.sparsity_ratio)>0.001) and (alpha_hist[1]-alpha_hist[0]>=0.001):
#                         if cur_sparsity > args.sparsity_ratio:
#                             alpha_new = (alpha + alpha_hist[0]) / 2.0
#                             alpha_hist[1] = alpha
#                         else:
#                             alpha_new = (alpha + alpha_hist[1]) / 2.0
#                             alpha_hist[0] = alpha

#                         alpha = alpha_new 
#                         W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
#                     print(f"alpha found {alpha} sparsity {cur_sparsity:.6f}")
#                 else:
#                     # unstructured pruning
#                     indices = sort_res[1][:,:int(W_metric.shape[1]*args.sparsity_ratio)]
#                     W_mask.scatter_(1, indices, True)

#             subset[name].weight.data[W_mask] = 0  ## set weights to zero 

#         for j in range(args.nsamples):
#             with torch.no_grad():
#                 outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
#         inps, outs = outs, inps

#     model.config.use_cache = use_cache 
#     torch.cuda.empty_cache()

def prune_wanda(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    use_cache = model.config.use_cache
    model.config.use_cache = False

    print("loading calibration data")
    dataloader, _ = get_loaders("c4", nsamples=args.nsamples, seed=args.seed, seqlen=model.seqlen, tokenizer=tokenizer)
    print("dataset loading complete")

    # Prepare calibration data on CPU, only moving to GPU when required
    inps_cpu, outs_cpu, attention_mask_cpu, position_ids_cpu = prepare_calibration_input(model, dataloader, "cpu")

    layers = model.model.layers
    for i, layer in enumerate(layers):
        subset = find_layers(layer)

        # Determine device for the current layer
        dev = model.hf_device_map.get(f"model.layers.{i}", device)

        # Move only required data to GPU for this specific layer
        inps = inps_cpu.to(dev, non_blocking=True)
        outs = outs_cpu.to(dev, non_blocking=True)
        attention_mask = attention_mask_cpu.to(dev, non_blocking=True)
        position_ids = position_ids_cpu.to(dev, non_blocking=True)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        # Capture intermediate activations on CPU
        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data.cpu(), out.data.cpu())  # Store activations on CPU
            return tmp

        # Register forward hooks for capturing activations
        handles = [subset[name].register_forward_hook(add_batch(name)) for name in wrapped_layers]

        # Iterate over each sample, one by one, to reduce memory load
        for j in range(args.nsamples):
            with torch.no_grad():
                # Move each sample to GPU only when needed
                inps_j = inps[j].unsqueeze(0).to(dev, non_blocking=True)
                outs[j] = layer(inps_j, attention_mask=attention_mask, position_ids=position_ids)[0]

        # Remove hooks to free up memory
        for h in handles:
            h.remove()

        # Apply pruning based on calculated metrics
        for name in subset:
            print(f"pruning layer {i} name {name}")
            # Compute W_metric on CPU to avoid GPU memory overload
            W_metric = torch.abs(subset[name].weight.data.cpu()) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1, -1)).cpu())
            W_mask = torch.zeros_like(W_metric, dtype=torch.bool)  # Initialize mask as all False

            # Structured n:m sparsity
            if prune_n != 0:
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:, ii:(ii + prune_m)].float()
                        W_mask.scatter_(1, ii + torch.topk(tmp, prune_n, dim=1, largest=False)[1], True)
            else:
                # Unstructured pruning with wanda variant
                sort_res = torch.sort(W_metric, dim=-1, stable=True)
                if args.use_variant:
                    tmp_metric = torch.cumsum(sort_res[0], dim=1)
                    sum_before = W_metric.sum(dim=1)
                    alpha = 0.4
                    alpha_hist = [0., 0.8]
                    W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    while abs(cur_sparsity - args.sparsity_ratio) > 0.001 and (alpha_hist[1] - alpha_hist[0] >= 0.001):
                        if cur_sparsity > args.sparsity_ratio:
                            alpha = (alpha + alpha_hist[0]) / 2.0
                            alpha_hist[1] = alpha
                        else:
                            alpha = (alpha + alpha_hist[1]) / 2.0
                            alpha_hist[0] = alpha
                        W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    print(f"alpha found {alpha} sparsity {cur_sparsity:.6f}")
                else:
                    indices = sort_res[1][:, :int(W_metric.shape[1] * args.sparsity_ratio)]
                    W_mask.scatter_(1, indices, True)

            # Apply mask on GPU, then move back to CPU to save GPU memory
            subset[name].weight.data[W_mask.to(dev)] = 0  # Apply mask and zero-out pruned weights

        # Update CPU tensors for the next layer, freeing GPU memory immediately
        inps_cpu = outs_cpu.clone()
        outs_cpu.zero_()
        torch.cuda.empty_cache()  # Explicitly free GPU memory to prevent memory overflow

    # Restore original cache setting and clear GPU cache at the end
    model.config.use_cache = use_cache
    torch.cuda.empty_cache()



@torch.no_grad()
def prune_sparsegpt(args, model, tokenizer, dev, prune_n=0, prune_m=0):
    ## SparseGPT code available at: https://github.com/IST-DASLab/sparsegpt/tree/f5c25005a61f96a0933ca2f95705a963585aafaa
    print('Starting ...')
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    if "model.embed_tokens" in model.hf_device_map:
        dev = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    print('Ready.')

    for i in range(len(layers)):
        layer = layers[i]
        if f"model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.layers.{i}"]
            print(f"layer {i} device {dev}")
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

        subset = find_layers(layer)

        gpts = {}
        for name in subset:
            gpts[name] = SparseGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        for name in gpts:
            print(i, name)
            print('Pruning ...')

            gpts[name].fasterprune(args.sparsity_ratio, prune_n=prune_n, prune_m=prune_m, percdamp=0.01, blocksize=128)
            gpts[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        layers[i] = layer 
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()



@torch.no_grad()
def prune_ablate(args, model, tokenizer, dev, prune_n=0, prune_m=0):
    ## SparseGPT code available at: https://github.com/IST-DASLab/sparsegpt/tree/f5c25005a61f96a0933ca2f95705a963585aafaa
    print('Starting ...')
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    if "model.embed_tokens" in model.hf_device_map:
        dev = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    print('Ready.')

    for i in range(len(layers)):
        layer = layers[i]
        if f"model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.layers.{i}"]
            print(f"layer {i} device {dev}")
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

        subset = find_layers(layer)

        gpts = {}
        for name in subset:
            gpts[name] = AblateGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        for name in gpts:
            print(i, name)
            print('Pruning ...')

            if args.prune_method == "ablate_wanda_seq":
                prune_mask = gpts[name].get_wanda_mask(args.sparsity_ratio, prune_n, prune_m)
            elif args.prune_method == "ablate_mag_seq":
                prune_mask = gpts[name].get_mag_mask(args.sparsity_ratio, prune_n, prune_m)
            elif "iter" in args.prune_method:
                prune_mask = None 

            gpts[name].fasterprune(args, args.sparsity_ratio, mask=prune_mask, prune_n=prune_n, prune_m=prune_m, percdamp=0.01, blocksize=128)
            gpts[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        layers[i] = layer 
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()