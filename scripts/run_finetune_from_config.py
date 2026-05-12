#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
根据给定的 config.pkl 直接调用 src.model_a.finetune.finetune_model 进行一次微调。
"""

import argparse
import pickle
import os
import sys


def main():
    parser = argparse.ArgumentParser(description="Run finetune_model with a pickled config.")
    parser.add_argument("--config_path", type=str, required=True, help="Path to config.pkl")
    args = parser.parse_args()

    config_path = args.config_path
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config_path not found: {config_path}")

    with open(config_path, "rb") as f:
        config = pickle.load(f)

    sys.path.insert(0, os.path.dirname(__file__))
    from src.model_a.finetune import finetune_model

    finetune_model(config)


if __name__ == "__main__":
    main()

