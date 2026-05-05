import modal

app = modal.App("mppi-pf-gpu")

REMOTE_PROJECT_DIR = "/root/mppi_pf_gpu"

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(
        "ffmpeg",
        "libgl1",
        "libglib2.0-0",
        "libegl1",
        "libgles2",
        "libglvnd0",
        "libosmesa6",
        "patchelf",
    )
    .pip_install(
        "numpy",
        "gymnasium[mujoco]",
        "gymnasium[other]",
        "mujoco",
        "cupy-cuda12x",
        "nvidia-cuda-nvrtc-cu12",
        "nvidia-cuda-runtime-cu12",
        "imageio",
        "imageio-ffmpeg",
        "moviepy",
    )
    .env(
        {
            # Important for headless MuJoCo rendering on Modal.
            "MUJOCO_GL": "egl",
            "PYOPENGL_PLATFORM": "egl",

            # Keeps CuPy kernel cache somewhere writable.
            "CUPY_CACHE_DIR": "/tmp/cupy_cache",

            # Avoid noisy Python buffering when watching logs.
            "PYTHONUNBUFFERED": "1",
        }
    )
    .add_local_dir(
        "./mppi_pf_gpu",
        remote_path=REMOTE_PROJECT_DIR,
        ignore=[
            "__pycache__",
            "*.pyc",
            ".git",
            "videos",
            "timing_log.npy",
        ],
    )
)


@app.function(
    image=image,
    gpu="T4",
    timeout=120,
)
def smoke_test():
    import os
    import subprocess

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    print("Running nvidia-smi...")
    subprocess.run(["nvidia-smi"], check=True)

    print("Testing CuPy...")
    import cupy as cp

    x = cp.arange(10, dtype=cp.float32)
    print("cupy sum:", float(cp.sum(x)))
    print("cupy device:", cp.cuda.runtime.getDevice())

    print("Testing Gymnasium/MuJoCo Pusher-v5...")
    import gymnasium as gym

    env = gym.make("Pusher-v5", render_mode="rgb_array", max_episode_steps=5)
    obs, info = env.reset()
    frame = env.render()
    env.close()

    return {
        "obs_shape": tuple(obs.shape),
        "frame_shape": tuple(frame.shape),
        "cupy_sum": float(cp.sum(x)),
    }


@app.function(
    image=image,
    gpu="T4",
    timeout=900,
)
def run_episode(
    k: int = 1024,
    n: int = 1000,
    h: int = 30,
    deadline: float = 50.0,
    sigma: float = 1.0,
    lambda_: float = 1.0,
    steps: int = 300,
    record: bool = False,
    no_pf: bool = False,
    no_timing: bool = False,
):
    import os
    import sys
    import glob
    import shutil
    from pathlib import Path

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("CUPY_CACHE_DIR", "/tmp/cupy_cache")

    os.chdir(REMOTE_PROJECT_DIR)
    sys.path.insert(0, REMOTE_PROJECT_DIR)

    # Avoid accidentally returning stale files from previous container reuse.
    videos_dir = Path(REMOTE_PROJECT_DIR) / "videos"
    if videos_dir.exists():
        shutil.rmtree(videos_dir)
    videos_dir.mkdir(parents=True, exist_ok=True)

    timing_path = Path(REMOTE_PROJECT_DIR) / "timing_log.npy"
    if timing_path.exists():
        timing_path.unlink()

    from config import Config
    from runner import run

    cfg = Config(
        K=k,
        N=n,
        H=h,
        deadline_ms=deadline,
        sigma=sigma,
        lambda_=lambda_,
        max_steps=steps,
        device_id=0,
        enable_timing=not no_timing,
    )

    total_reward, timing_log = run(
        cfg,
        render=False,
        record=record,
        no_pf=no_pf,
    )

    results = {
        "total_reward": float(total_reward),
        "timing_log": timing_log,
    }

    # runner.py already saves timing_log.npy; return it as bytes too.
    if timing_path.exists():
        results["timing_log_npy_bytes"] = timing_path.read_bytes()

    if record:
        mp4s = sorted(glob.glob(str(videos_dir / "*.mp4")))
        if mp4s:
            video_path = mp4s[-1]
            with open(video_path, "rb") as f:
                results["video_bytes"] = f.read()
            results["video_filename"] = os.path.basename(video_path)
        else:
            results["video_error"] = "record=True, but no mp4 file was produced."

    return results


@app.local_entrypoint()
def main(
    k: int = 1024,
    n: int = 1000,
    h: int = 30,
    deadline: float = 50.0,
    sigma: float = 1.0,
    lambda_: float = 1.0,
    steps: int = 300,
    record: bool = False,
    no_pf: bool = False,
    no_timing: bool = False,
    smoke: bool = False,
):
    if smoke:
        result = smoke_test.remote()
        print("Smoke test passed:")
        print(result)
        return

    results = run_episode.remote(
        k=k,
        n=n,
        h=h,
        deadline=deadline,
        sigma=sigma,
        lambda_=lambda_,
        steps=steps,
        record=record,
        no_pf=no_pf,
        no_timing=no_timing,
    )

    print(f"\nTotal reward: {results['total_reward']:.3f}")

    # Save timing log locally.
    if "timing_log_npy_bytes" in results:
        with open("timing_log.npy", "wb") as f:
            f.write(results["timing_log_npy_bytes"])
        print("Saved timing_log.npy")
    else:
        # Fallback: save pickled/object NumPy array locally.
        import numpy as np

        np.save("timing_log.npy", results["timing_log"], allow_pickle=True)
        print("Saved timing_log.npy from returned timing_log")

    # Save video locally.
    if "video_bytes" in results:
        with open("episode.mp4", "wb") as f:
            f.write(results["video_bytes"])
        print(f"Saved episode.mp4 from {results.get('video_filename', 'remote mp4')}")
    elif "video_error" in results:
        print(results["video_error"])