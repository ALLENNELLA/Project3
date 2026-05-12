"""
rebuild_ctc_dataset.py

生成/重建 ptDecoder_ctc{nDays} 数据文件（pickle，无后缀）。

支持两种模式：
1) source=mat  : 调用 make_dataset() 从 competitionData 的 .mat 重新构建（需要 g2p_en 依赖）
2) source=slice: 从一个“更大天数”的已存在 ptDecoder_ctc* 文件切片得到前 nDays（不依赖 g2p_en）

运行示例：
  # 直接从已有 ctc17 切出前10天（推荐：无额外依赖）
  PYTHONPATH=/root/25S151115/project3/scripts python3 -m src.utils.rebuild_ctc_dataset --ndays 10 --source slice

  # 强制重建前12天（会先备份旧文件）
  PYTHONPATH=/root/25S151115/project3/scripts python3 -m src.utils.rebuild_ctc_dataset --ndays 12 --source slice --force
"""

from __future__ import annotations

import argparse
import os
import shutil
from datetime import datetime


def _backup_if_exists(path: str) -> str | None:
    if not os.path.exists(path):
        return None
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{path}.bak.{ts}"
    shutil.copy2(path, backup_path)
    return backup_path


def _quick_validate(dataset_path: str, expected_days: int) -> None:
    # 只做轻量结构校验：确保 train/test 的天数符合预期
    import pickle

    with open(dataset_path, "rb") as f:
        obj = pickle.load(f)

    if not isinstance(obj, dict):
        raise ValueError(f"{dataset_path}: not a dict, got {type(obj)}")

    for split in ("train", "test"):
        if split not in obj:
            raise ValueError(f"{dataset_path}: missing split '{split}'")
        if not isinstance(obj[split], list):
            raise ValueError(f"{dataset_path}: split '{split}' not a list, got {type(obj[split])}")
        if len(obj[split]) != expected_days:
            raise ValueError(
                f"{dataset_path}: split '{split}' has {len(obj[split])} days, expected {expected_days}"
            )


def _get_session_names_sorted() -> list[str]:
    # 与训练配置一致：config.py 内会 sort()
    from ..model_a.config import get_base_config

    cfg = get_base_config()
    names = list(cfg["sessionNames"])
    names.sort()
    return names


def _slice_competition_list(
    source_competition: list,
    session_names_first_n: list[str],
    competition_data_dir: str,
) -> list:
    """
    make_dataset 的逻辑：按 day 顺序遍历 session_names，若存在 competitionHoldOut/{session}.mat 则 append。
    所以 competition list 是“有 holdout 的 session 子序列”，顺序与 session_names 保持一致。
    """
    holdout_dir = os.path.join(competition_data_dir, "competitionHoldOut")
    need = 0
    for s in session_names_first_n:
        if os.path.exists(os.path.join(holdout_dir, f"{s}.mat")):
            need += 1
    return list(source_competition[:need])


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild ptDecoder_ctc{nDays} dataset.")
    parser.add_argument("--ndays", type=int, required=True, help="汇总天数，例如 10、12")
    parser.add_argument("--model-name", type=str, default="conformer", help="用于生成 config 的 model_name")
    parser.add_argument("--seed", type=int, default=0, help="训练/数据生成用 seed（影响输出目录名，但不影响 dataset 内容）")
    parser.add_argument(
        "--base-dir",
        type=str,
        default="/root/25S151115/project3",
        help="project3 根目录（用于确定输出 ptDecoder_ctc{nDays} 路径）",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/root/25S151115/project2/data/competitionData",
        help="competitionData 根目录（包含 train/test/competitionHoldOut）",
    )
    parser.add_argument(
        "--source",
        type=str,
        choices=("slice", "mat"),
        default="slice",
        help="slice=从已有ptDecoder_ctc*切片生成；mat=从.mat重建（需要g2p_en）",
    )
    parser.add_argument(
        "--source-dataset",
        type=str,
        default="/root/25S151115/project3/data/ptDecoder_ctc17",
        help="当 source=slice 时，作为切片来源的 ptDecoder_ctc* 路径（需天数>=ndays）",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="若输出已存在则覆盖（覆盖前自动备份）",
    )

    args = parser.parse_args()

    # 延迟 import，避免 argparse --help 时引入 heavy deps
    from ..model_a.config import get_train_config

    config = get_train_config(
        nDays=args.ndays,
        model_name=args.model_name,
        base_dir=args.base_dir,
        seed=args.seed,
    )

    out_path = config["datasetPath"]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if os.path.exists(out_path) and not args.force:
        raise SystemExit(f"❌ 输出已存在：{out_path}（如需重建请加 --force）")

    backup_path = _backup_if_exists(out_path)
    if backup_path:
        print(f"🧷 已备份旧文件 -> {backup_path}")

    if args.source == "mat":
        # 需要 g2p_en
        from .make_dataset import make_dataset

        print(f"🚧 开始生成（from .mat）: ndays={args.ndays} -> {out_path}")
        make_dataset(config, data_dir=args.data_dir)
    else:
        import pickle

        print(f"🚧 开始生成（slice）: ndays={args.ndays} <- {args.source_dataset} -> {out_path}")
        with open(args.source_dataset, "rb") as f:
            src = pickle.load(f)

        if len(src.get("train", [])) < args.ndays or len(src.get("test", [])) < args.ndays:
            raise ValueError(
                f"source dataset days insufficient: "
                f"train={len(src.get('train', []))} test={len(src.get('test', []))} need={args.ndays}"
            )

        session_names = _get_session_names_sorted()
        session_first_n = session_names[: args.ndays]

        out = {
            "train": list(src["train"][: args.ndays]),
            "test": list(src["test"][: args.ndays]),
            "competition": _slice_competition_list(
                source_competition=list(src.get("competition", [])),
                session_names_first_n=session_first_n,
                competition_data_dir=args.data_dir,
            ),
        }

        with open(out_path, "wb") as f:
            pickle.dump(out, f)
        print(f"✅ Dataset saved to {out_path}")

    print("🔎 生成后校验中...")
    _quick_validate(out_path, expected_days=args.ndays)
    print("✅ 校验通过")


if __name__ == "__main__":
    main()

