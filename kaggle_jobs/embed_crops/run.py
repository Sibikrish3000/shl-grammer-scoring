"""Kaggle GPU job (2x T4): kaggle.embed --crops, WavLM on GPU 0 and Whisper encoder on GPU 1.

Runs the exact local code from the sibikrish3000/shl-code dataset.
Outputs (/kaggle/working/emb): wavlm_crops_mean.npy, whisper_crops_mean.npy (rows = crops.csv order).
"""
import glob
import os
import shutil
import subprocess
import sys

code = os.path.dirname(glob.glob("/kaggle/input/**/embed.py", recursive=True)[0])
pkg = "/kaggle/working/pkg/kaggle"
shutil.copytree(code, pkg, dirs_exist_ok=True)
data = os.path.dirname(glob.glob("/kaggle/input/**/train.csv", recursive=True)[0])
feat = "/kaggle/working"
shutil.copy(glob.glob("/kaggle/input/**/crops.csv", recursive=True)[0], f"{feat}/crops.csv")
os.makedirs(f"{feat}/emb", exist_ok=True)

env = {**os.environ, "PYTHONPATH": "/kaggle/working/pkg", "DATA_DIR": data, "FEAT_DIR": feat}
jobs = [subprocess.Popen([sys.executable, "-m", "kaggle.embed", "--crops", m], env={**env, "CUDA_VISIBLE_DEVICES": str(g)})
        for g, m in enumerate(["wavlm", "whisper"])]
codes = [j.wait() for j in jobs]
shutil.rmtree("/kaggle/working/pkg")
assert codes == [0, 0], codes
print(os.listdir(f"{feat}/emb"))
