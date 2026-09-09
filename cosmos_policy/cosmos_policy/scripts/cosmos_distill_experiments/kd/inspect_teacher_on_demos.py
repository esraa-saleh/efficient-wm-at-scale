"""Sanity-check a teacher_on_demos sidecar. Run with the venv:
    .venv/bin/python -m cosmos_policy.scripts.cosmos_distill_experiments.kd.inspect_teacher_on_demos <sidecar_dir>
(defaults to the smoketest dir). Writes /tmp/tod_<demo>_{primary,wrist}.png for a visual look."""
import glob, io, os, sys
import numpy as np
import h5py
from PIL import Image

d = sys.argv[1] if len(sys.argv) > 1 else (
    os.environ["COSMOS_POLICY_STORAGE"] + "/LIBERO-Cosmos-Policy/success_only/libero_object_regen__teacher_on_demos_smoketest"
)
OUT = sys.argv[2] if len(sys.argv) > 2 else "/tmp"
os.makedirs(OUT, exist_ok=True)
paths = glob.glob(os.path.join(d, "**", "*.teacher_on_demos.hdf5"), recursive=True)
assert paths, f"no sidecar under {d}"

for p in paths:
    f = h5py.File(p)
    print(f"\n=== {p}")
    print("root attrs:", {k: (list(f.attrs[k]) if hasattr(f.attrs[k], '__len__') and not isinstance(f.attrs[k], str) else f.attrs[k])
                           for k in ("complete", "demo_keys", "num_denoising_steps", "seed", "build_git_sha", "source_relpath")
                           if k in f.attrs})
    for dk in f:
        g = f[dk]
        ac, fp, v = g["action_chunks"][:], g["future_proprio"][:], g["value"][:]
        T = g.attrs["num_steps"]
        print(f"\n[{dk}] num_steps={T} chunk_size={g.attrs['chunk_size']}")
        print("  shapes:", {k: g[k].shape for k in g})
        print("  action_chunks  min/max/mean  %.3f / %.3f / %.3f" % (ac.min(), ac.max(), ac.mean()))
        print("  future_proprio min/max       %.3f / %.3f" % (fp.min(), fp.max()))
        print("  value          min/max       %.3f / %.3f   |  first3=%s  last3=%s"
              % (v.min(), v.max(), np.round(v[:3], 3), np.round(v[-3:], 3)))
        assert ac.shape == (T, 16, 7) and fp.shape == (T, 9) and v.shape == (T,)
        assert -1.5 < ac.min() and ac.max() < 1.5, "action_chunks wildly out of [-1,1]"
        assert -1.05 <= v.min() and v.max() <= 1.05, "value out of [-1,1] -- *2-1 remap bug?"
        t = T // 2
        prim = np.array(Image.open(io.BytesIO(bytes(g["future_image_jpeg"][t]))))
        wr = np.array(Image.open(io.BytesIO(bytes(g["future_wrist_image_jpeg"][t]))))
        print("  future_image  ", prim.shape, prim.dtype, " future_wrist ", wr.shape)
        assert prim.shape == (224, 224, 3) and prim.dtype == np.uint8, "teacher decode not 224x224x3 uint8"
        Image.fromarray(prim).save(f"{OUT}/tod_{dk}_primary.png")
        Image.fromarray(wr).save(f"{OUT}/tod_{dk}_wrist.png")
        print(f"  wrote {OUT}/tod_{dk}_primary.png  {OUT}/tod_{dk}_wrist.png")
    f.close()

print("\nSHAPE/RANGE CHECKS PASSED -- now eyeball the PNGs: upright, natural colours, plausible LIBERO scene.")
