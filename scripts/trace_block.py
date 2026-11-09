import torch
from transformers import AutoModel
m = AutoModel.from_pretrained("facebook/dinov2-small").eval()
c = m.config
print("hidden", c.hidden_size, "heads", c.num_attention_heads, "layers", c.num_hidden_layers, "patch", c.patch_size)
layer0 = m.encoder.layer[0]
for name, mod in layer0.named_modules():
    print(f"{name or '<layer>':35s} {type(mod).__name__}")
def hook(name):
    def f(mod, inp, out):
        print(f"{name:35s} in {tuple(inp[0].shape)} -> out {tuple(out.shape)}")
    return f
for name, mod in layer0.named_modules():
    if isinstance(mod, torch.nn.Linear):
        mod.register_forward_hook(hook(name))
with torch.no_grad():
    out = m(pixel_values=torch.randn(1, 3, 224, 224))
print("output", tuple(out.last_hidden_state.shape))
print("total params", sum(p.numel() for p in m.parameters()))
