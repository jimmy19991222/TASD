#!/usr/bin/env python3
import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--script_path", type=str, required=True)
    parser.add_argument("--world_size", type=int, required=True)
    parser.add_argument("--job_name", type=str, required=True)
    args = parser.parse_args()

    # 构造命令
    cmd = (
        f"bash nebula_scripts/launch_ray_cluster.sh --script_path={args.script_path} "
        f"--world_size={args.world_size} "
        f"--job_name={args.job_name}"
    )
    os.system(cmd)
