#!/usr/bin/env python3
"""
使用 ResNeXt-101 从 MUStARD 视频提取视觉特征，并按 CCG-Net 数据格式存成 pkl。
配置：
- 输入视频：/home/ubuntu/csp/MUStARD-master/data/videos/{train,val,test}
- 标注 JSON：/home/ubuntu/csp/CCG-Net/Data_MMSD/Text/{split}_data.json
- 输出 pkl：/home/ubuntu/csp/CCG-Net/Data_MMSD/Video/{split}_visual_features.pkl

特征设置（参考论文描述）：
- 以 ffmpeg 抽帧，fps=1.5，缩放至高度 720（等比例，宽度自动对齐为偶数）
- 窗口长度 16 帧，步长 1 帧（约 10.7s 的时间窗滑动）
- 使用 torchvision 的 ResNeXt-101-32x8d 预训练模型，取 fc 前 2048 维
- 每个 16 帧窗口对帧级特征做均值，得到 (T, 2048) 的片段特征
- 与已有 pkl 一致，video_feats 字段为 [np.ndarray] 的列表封装
"""

import json
import os
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torchvision import models


# ---------------------------------------------------------
# 配置
# ---------------------------------------------------------
DATASETS = [
    {
        "name": "TEST",
        "video_dir": "/home/ubuntu/csp/MUStARD-master/data/videos/test",
        "json": "/home/ubuntu/csp/CCG-Net/Data_MMSD/Text_after/test_data_final_plain.json",
        "output": "/home/ubuntu/csp/CCG-Net/Data_MMSD/Video/test_visual_features.pkl",
    },
    {
        "name": "TRAIN",
        "video_dir": "/home/ubuntu/csp/MUStARD-master/data/videos/train",
        "json": "/home/ubuntu/csp/CCG-Net/Data_MMSD/Text_after/train_data_final_plain.json",
        "output": "/home/ubuntu/csp/CCG-Net/Data_MMSD/Video/train_visual_features.pkl",
    },
    {
        "name": "VAL",
        "video_dir": "/home/ubuntu/csp/MUStARD-master/data/videos/val",
        "json": "/home/ubuntu/csp/CCG-Net/Data_MMSD/Text_after/val_data_final_plain.json",
        "output": "/home/ubuntu/csp/CCG-Net/Data_MMSD/Video/val_visual_features.pkl",
    },
]

FPS = 1.5
WINDOW = 8
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------
# 工具函数
# ---------------------------------------------------------
def extract_frames_ffmpeg(video_path: str, frames_dir: str) -> bool:
    """使用 ffmpeg 抽帧，fps=FPS，缩放到 720p 高度，宽度按比例缩放为偶数。"""
    Path(frames_dir).mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-vf",
        f"fps={FPS},scale=-2:720",
        os.path.join(frames_dir, "%05d.jpg"),
    ]
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except subprocess.CalledProcessError:
        return False


def load_resnet152() -> torch.nn.Module:
    """加载预训练 ResNet-152 (MUStARD 原论文方法)，去掉最后分类层，返回 2048 维。"""
    # 使用 torchvision 的 ResNet-152
    try:
        # 新版 torchvision 写法
        weights = models.ResNet152_Weights.IMAGENET1K_V1
        model = models.resnet152(weights=weights)
    except Exception:
        # 旧版 torchvision 写法
        model = models.resnet152(pretrained=True)

    model.eval()

    # 替换 fc 层为直通，提取 pool5 特征 (2048维)
    class Identity(torch.nn.Module):
        def forward(self, x):
            return x

    model.fc = Identity()

    for p in model.parameters():
        p.requires_grad = False
    return model.to(DEVICE)


def frames_to_tensor(frame_paths: List[str], transform: transforms.Compose) -> torch.Tensor:
    """读取多帧并变换为 tensor，输出形状 (N, 3, H, W)。"""
    tensors = []
    for fp in frame_paths:
        img = Image.open(fp).convert("RGB")
        tensors.append(transform(img))
    return torch.stack(tensors) if tensors else torch.empty((0,))


def extract_clip_features(
    frame_dir: str, model: torch.nn.Module, transform: transforms.Compose
) -> np.ndarray:
    """
    读取帧 -> 提取帧级特征 -> 按滑窗均值 -> (T, 2048) ndarray。
    - 当帧数 >= WINDOW 时使用 WINDOW=16
    - 当帧数 < 16 时，将窗口长度下调为帧数本身（至少 1），避免丢弃短视频
    """
    frame_paths = sorted(
        [os.path.join(frame_dir, f) for f in os.listdir(frame_dir) if f.lower().endswith(".jpg")]
    )
    n_frames = len(frame_paths)
    if n_frames == 0:
        return np.empty((0, 2048), dtype=np.float32)

    window = WINDOW if n_frames >= WINDOW else n_frames

    # 帧级特征
    frame_feats: List[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(frame_paths), BATCH_SIZE):
            batch_paths = frame_paths[start : start + BATCH_SIZE]
            batch = frames_to_tensor(batch_paths, transform)
            if batch.numel() == 0:
                continue
            batch = batch.to(DEVICE)
            feats = model(batch)  # (B, 2048)
            frame_feats.append(feats.cpu())

    if not frame_feats:
        return np.empty((0, 2048), dtype=np.float32)

    frame_feats = torch.cat(frame_feats, dim=0)  # (N, 2048)

    # 滑窗聚合
    clip_list: List[torch.Tensor] = []
    for i in range(0, frame_feats.shape[0] - window + 1):
        clip_feat = frame_feats[i : i + window].mean(dim=0)
        clip_list.append(clip_feat)

    if not clip_list:
        return np.empty((0, 2048), dtype=np.float32)

    clips = torch.stack(clip_list, dim=0)  # (T, 2048)
    return clips.numpy().astype(np.float32)


def process_split(split_cfg: Dict[str, str], model: torch.nn.Module, transform: transforms.Compose) -> Tuple[int, int]:
    """处理单个划分，返回 (success, failed)。"""
    video_dir = split_cfg["video_dir"]
    json_path = split_cfg["json"]
    output_path = split_cfg["output"]

    if not os.path.exists(video_dir):
        print(f"[{split_cfg['name']}] 视频目录不存在: {video_dir}")
        return 0, 0
    if not os.path.exists(json_path):
        print(f"[{split_cfg['name']}] JSON 不存在: {json_path}")
        return 0, 0

    with open(json_path, "r", encoding="utf-8") as f:
        data_dict = json.load(f)

    rows = []
    success = 0
    failed = 0

    # 临时帧目录
    tmp_root = tempfile.mkdtemp(prefix=f"mmsd_frames_{split_cfg['name'].lower()}_")

    try:
        for idx, (vid, info) in enumerate(data_dict.items(), 1):
            video_path = os.path.join(video_dir, f"{vid}.mp4")
            frame_dir = os.path.join(tmp_root, vid)

            print(f"[{split_cfg['name']}] ({idx}/{len(data_dict)}) {vid}", end=" ... ")

            if not os.path.exists(video_path):
                print("视频缺失")
                failed += 1
                continue

            # 抽帧
            if not extract_frames_ffmpeg(video_path, frame_dir):
                print("抽帧失败")
                failed += 1
                continue

            # 特征
            feats = extract_clip_features(frame_dir, model, transform)
            if feats.size == 0:
                print("特征为空/失败")
                failed += 1
                continue

            row = {
                "episode_name": info.get("episode_name", ""),
                "target_speaker": info.get("target_speaker", ""),
                "target_utterance": info.get("target_utterance", ""),
                "context_speakers": info.get("context_speakers", []),
                "context_utterances": info.get("context_utterances", []),
                "sarcasm_target": info.get("sarcasm_target", "UNKNOWN"),
                "sarcasm_type": info.get("sarcasm_type", "UNKNOWN"),
                "english_explanation": info.get("english_explanation", ""),
                "code_mixed_explanation": info.get("code_mixed_explanation", ""),
                "start_time": info.get("start_time", "00:00:00"),
                "end_time": info.get("end_time", "00:00:00"),
                # 与现有 pkl 保持一致，列表包裹 numpy 数组
                "video_feats": [feats],
            }
            rows.append(row)
            success += 1
            print(f"✓ clips={feats.shape[0]} dim={feats.shape[1]}")

        if rows:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            df = pd.DataFrame(rows)
            with open(output_path, "wb") as f:
                pickle.dump(df, f)
            print(
                f"[{split_cfg['name']}] 完成: 成功 {success}, 失败 {failed}, "
                f"保存 {len(df)} 条至 {output_path}"
            )
            print(
                f"[{split_cfg['name']}] 平均 clip 数: {np.mean([r.shape[0] for r in df['video_feats'].str[0]]):.1f}"
            )
        else:
            print(f"[{split_cfg['name']}] 未生成任何样本")
    finally:
        # 清理临时帧
        import shutil

        shutil.rmtree(tmp_root, ignore_errors=True)

    return success, failed


def main() -> None:
    print("=" * 80)
    print("MUStARD -> CCG-Net 视觉特征提取 (ResNet-152, fps=1.5, window=8)")
    print(f"设备: {DEVICE}")
    print("=" * 80)

    # MUStARD 原论文预处理：Resize(256) -> CenterCrop(224)
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    model = load_resnet152()

    total_s, total_f = 0, 0
    for cfg in DATASETS:
        s, f = process_split(cfg, model, transform)
        total_s += s
        total_f += f

    print("=" * 80)
    print(f"全部完成，总成功 {total_s}, 总失败 {total_f}")
    if total_s + total_f > 0:
        rate = total_s / (total_s + total_f) * 100
        print(f"成功率: {rate:.1f}%")


if __name__ == "__main__":
    sys.exit(main())


