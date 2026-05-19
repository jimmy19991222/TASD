#!/usr/bin/env python3
import os
import subprocess
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--script_path", type=str, required=True)
    parser.add_argument("--world_size", type=int, required=True)
    parser.add_argument("--job_name", type=str, required=True)
    # 接收所有超参（KEY=VALUE 格式），注入到当前进程环境
    parser.add_argument("--env", type=str, action="append", default=[],
                        help="KEY=VALUE pairs to inject into environment")
    args = parser.parse_args()

    # 将 --env KEY=VALUE 写入 os.environ
    env = os.environ.copy()
    for kv in args.env:
        if "=" in kv:
            k, v = kv.split("=", 1)
            env[k] = v
            print(f"[entry.py] inject: {k}={v}")
        else:
            print(f"[entry.py] WARNING: invalid --env format: {kv}")

    cmd = [
        "bash",
        "nebula_scripts/launch_ray_cluster.sh",
        f"--script_path={args.script_path}",
        f"--world_size={args.world_size}",
        f"--job_name={args.job_name}",
    ]
    ret = subprocess.run(cmd, env=env)
    exit(ret.returncode)
