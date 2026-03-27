#!/usr/bin/env python3
import os
import subprocess
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--script_path", type=str, required=True)
    parser.add_argument("--world_size", type=int, required=True)
    parser.add_argument("--job_name", type=str, required=True)
    args = parser.parse_args()

    cmd = [
        "bash",
        "nebula_scripts/launch_ray_cluster.sh",
        f"--script_path={args.script_path}",
        f"--world_size={args.world_size}",
        f"--job_name={args.job_name}",
    ]
    # 显式传入当前完整环境，确保 Nebula --env 注入的变量被继承
    ret = subprocess.run(cmd, env=os.environ.copy())
    exit(ret.returncode)
