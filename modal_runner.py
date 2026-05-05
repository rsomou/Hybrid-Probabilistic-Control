import modal

app = modal.App("mppi-pf-gpu")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install(
        "gymnasium[mujoco]",
        "cupy-cuda12x",
        "numpy",
    )
    .add_local_dir("./mppi_pf_gpu", remote_path="/root/mppi_pf_gpu")
)


@app.function(
    image=image,
    gpu="T4",
    timeout=600,
)
def run_episode(
    K: int = 1024,
    N: int = 1000,
    H: int = 30,
    deadline: float = 50.0,
    sigma: float = 1.0,
    lambda_: float = 1.0,
    steps: int = 300,
    record: bool = True,
    no_pf: bool = False,
    no_timing: bool = False,
):
    import os
    import sys

    os.chdir("/root/mppi_pf_gpu")
    sys.path.insert(0, "/root/mppi_pf_gpu")

    from config import Config
    from runner import run

    cfg = Config(
        K=K, N=N, H=H,
        deadline_ms=deadline,
        sigma=sigma,
        lambda_=lambda_,
        max_steps=steps,
        device_id=0,
        enable_timing=not no_timing,
    )

    total_reward, timing_log = run(cfg, render=False, record=record, no_pf=no_pf)

    results = {
        "total_reward": total_reward,
        "timing_log": timing_log,
    }

    # If video was recorded, read the file bytes
    if record:
        import glob

        vids = glob.glob("./videos/*.mp4")
        if vids:
            with open(vids[0], "rb") as f:
                results["video_bytes"] = f.read()
    return results


@app.local_entrypoint()
def main(
    K: int = 1024,
    N: int = 1000,
    H: int = 30,
    deadline: float = 50.0,
    sigma: float = 1.0,
    lambda_: float = 1.0,
    steps: int = 300,
    record: bool = True,
    no_pf: bool = False,
    no_timing: bool = False,
):
    results = run_episode.remote(
        K=K, N=N, H=H,
        deadline=deadline,
        sigma=sigma,
        lambda_=lambda_,
        steps=steps,
        record=record,
        no_pf=no_pf,
        no_timing=no_timing,
    )

    print(f"\nTotal reward: {results['total_reward']:.3f}")

    # Save timing log locally
    import numpy as np

    np.save("timing_log.npy", results["timing_log"])
    print("Saved timing_log.npy")

    # Save video locally
    if "video_bytes" in results:
        with open("episode.mp4", "wb") as f:
            f.write(results["video_bytes"])
        print("Saved episode.mp4")