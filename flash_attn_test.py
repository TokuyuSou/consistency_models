import torch
from flash_attn import flash_attn_qkvpacked_func
b,s,h,d=1,128,16,128
import torch as T
qkv=T.randn(b,s,3,h,d,device="cuda",dtype=T.float16)
print(flash_attn_qkvpacked_func(qkv,dropout_p=0.0,causal=False).shape)
print("OK on:", torch.__version__, "CUDA", torch.version.cuda)
