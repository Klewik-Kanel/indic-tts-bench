"""Kaggle smoke test for indic-tts-bench. Generated; do not edit here."""
import hashlib, json, os, platform, subprocess, sys, time

RESULT = {"ok": False, "stages": {}}
OUT = "/kaggle/working/smoke_result.json"


def stage(name, fn):
    t0 = time.time()
    try:
        value = fn()
        RESULT["stages"][name] = {"ok": True, "value": value,
                                  "seconds": round(time.time() - t0, 2)}
        print(f"[ok]   {name}: {value}", flush=True)
        return value
    except Exception as exc:
        RESULT["stages"][name] = {"ok": False, "error": f"{type(exc).__name__}: {exc}",
                                  "seconds": round(time.time() - t0, 2)}
        print(f"[FAIL] {name}: {type(exc).__name__}: {exc}", flush=True)
        return None


def gpu():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError(
            "no CUDA device. The usual cause is an account that is not phone "
            "verified, which silently drops the notebook to CPU. Check "
            "kaggle.com/settings -> Accelerators before scheduling any run.")
    return {"name": torch.cuda.get_device_name(0),
            "count": torch.cuda.device_count(),
            "capability": list(torch.cuda.get_device_capability(0)),
            "total_memory_gb": round(
                torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 2),
            "torch": torch.__version__, "cuda": torch.version.cuda}


def throughput():
    """A fixed matmul, so T4 and P100 runs are comparable and the step budget
    can be sized from a measurement rather than a guess."""
    import torch
    d = torch.device("cuda")
    a = torch.randn(4096, 4096, device=d, dtype=torch.float16)
    b = torch.randn(4096, 4096, device=d, dtype=torch.float16)
    for _ in range(5):
        a @ b
    torch.cuda.synchronize()
    t0 = time.time()
    n = 50
    for _ in range(n):
        a @ b
    torch.cuda.synchronize()
    dt = time.time() - t0
    tflops = (2 * 4096 ** 3 * n) / dt / 1e12
    return {"fp16_matmul_tflops": round(tflops, 2), "seconds": round(dt, 3)}


def internet():
    import urllib.request
    with urllib.request.urlopen("https://huggingface.co/api/whoami-v2",
                                timeout=30) as r:
        return {"status": r.status}


def hf_auth():
    from kaggle_secrets import UserSecretsClient
    token = UserSecretsClient().get_secret("HF_TOKEN")
    os.environ["HF_TOKEN"] = token
    from huggingface_hub import HfApi
    who = HfApi(token=token).whoami()
    return {"name": who.get("name"), "type": who.get("type")}


def checkpoint_roundtrip():
    """Push a file to the Hub, delete it, pull it back, compare bytes.

    This is the mechanism that makes a killed Kaggle session cost N steps
    instead of a whole run, so it is tested before it is relied on.
    """
    from huggingface_hub import HfApi, hf_hub_download
    token = os.environ["HF_TOKEN"]
    api = HfApi(token=token)
    user = api.whoami()["name"]
    repo = f"{user}/indic-tts-bench-ckpt"
    api.create_repo(repo, private=True, exist_ok=True)

    payload = os.urandom(1 << 20)          # 1 MiB, enough to exercise LFS paths
    digest = hashlib.sha256(payload).hexdigest()
    local = "/kaggle/working/_roundtrip.bin"
    with open(local, "wb") as fh:
        fh.write(payload)

    api.upload_file(path_or_fileobj=local, path_in_repo="smoke/_roundtrip.bin",
                    repo_id=repo, commit_message="smoke test round trip")
    os.remove(local)

    back = hf_hub_download(repo_id=repo, filename="smoke/_roundtrip.bin",
                           token=token, force_download=True)
    with open(back, "rb") as fh:
        got = hashlib.sha256(fh.read()).hexdigest()
    if got != digest:
        raise RuntimeError(f"checkpoint corrupted in transit: {got} != {digest}")
    api.delete_file("smoke/_roundtrip.bin", repo_id=repo,
                    commit_message="smoke test cleanup")
    return {"repo": repo, "sha256": digest[:16], "bytes": len(payload)}


print("python", platform.python_version(), flush=True)
stage("gpu", gpu)
stage("throughput", throughput)
stage("internet", internet)
stage("hf_auth", hf_auth)
stage("checkpoint_roundtrip", checkpoint_roundtrip)

RESULT["ok"] = all(s["ok"] for s in RESULT["stages"].values())
with open(OUT, "w") as fh:
    json.dump(RESULT, fh, indent=2)
print("\n" + json.dumps(RESULT, indent=2), flush=True)
if not RESULT["ok"]:
    sys.exit(1)
