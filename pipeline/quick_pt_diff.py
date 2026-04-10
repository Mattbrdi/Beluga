import torch
from pathlib import Path

paths = [
    Path(r".\jsons\models\mobile_net_overlaps.pt"),
    Path(r".\jsons\models\mobile_net_8_layers_qat.pt"),
]

def summarize(obj, name):
    print(f"\n=== {name} ===")
    print("python type:", type(obj))

    if isinstance(obj, dict):
        print("dict keys (preview):", list(obj.keys())[:30])
        # est-ce que ça ressemble à un state_dict ?
        tensor_like = [k for k,v in obj.items() if hasattr(v, "shape") and hasattr(v, "dtype")]
        print("tensor-like entries:", len(tensor_like), "/", len(obj))
        if tensor_like:
            k0 = tensor_like[0]
            v0 = obj[k0]
            print("example tensor key:", k0)
            print("  shape:", tuple(v0.shape), "dtype:", v0.dtype)
        # state_dict embedded ?
        for k in ["state_dict", "model_state", "model", "net", "weights"]:
            if k in obj and isinstance(obj[k], dict):
                print(f"embedded dict at key='{k}' with {len(obj[k])} entries")
                ten2 = [kk for kk,vv in obj[k].items() if hasattr(vv,"shape") and hasattr(vv,"dtype")]
                print("  tensor-like in embedded:", len(ten2))
                if ten2:
                    kk0 = ten2[0]
                    vv0 = obj[k][kk0]
                    print("  example:", kk0, tuple(vv0.shape), vv0.dtype)

    elif hasattr(obj, "shape") and hasattr(obj, "dtype"):
        print("tensor shape:", tuple(obj.shape), "dtype:", obj.dtype)

for p in paths:
    obj = torch.load(p, map_location="cpu")
    print(f"\nFILE: {p}  size={p.stat().st_size/1024:.1f} KB")
    summarize(obj, p.name)
