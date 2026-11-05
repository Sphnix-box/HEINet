#!/usr/bin/env python3
"""
为 MMSD 数据集提取音频特征 (train/test/val)
生成 154 维音频特征 pkl 文件
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import librosa
import warnings
warnings.filterwarnings('ignore')

def extract_audio_from_video_ffmpeg(video_path, audio_output_path):
    """使用 ffmpeg 提取音频"""
    import subprocess
    try:
        cmd = [
            'ffmpeg', '-i', video_path,
            '-vn', '-acodec', 'pcm_s16le',
            '-ar', '22050', '-ac', '1',
            '-y', audio_output_path
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except:
        return False

def extract_acoustic_features(audio_path, sr=22050, hop_length=512):
    """提取154维音频特征"""
    try:
        y, sr = librosa.load(audio_path, sr=sr)
        features_list = []
        
        # MFCC - 40维
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40, hop_length=hop_length)
        features_list.append(mfcc)
        
        # Mel spectrogram - 40维
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40, hop_length=hop_length)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        features_list.append(mel_db)
        
        # Chroma - 12维
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)
        features_list.append(chroma)
        
        # Spectral contrast - 7维
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=hop_length)
        features_list.append(contrast)
        
        # Tonnetz - 6维
        tonnetz = librosa.feature.tonnetz(y=y, sr=sr, hop_length=hop_length)
        features_list.append(tonnetz)
        
        # Zero crossing rate - 1维
        zcr = librosa.feature.zero_crossing_rate(y, hop_length=hop_length)
        features_list.append(zcr)
        
        # Spectral features - 4维
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop_length)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop_length)
        spectral_flatness = librosa.feature.spectral_flatness(y=y, hop_length=hop_length)
        features_list.extend([spectral_centroid, spectral_rolloff, spectral_bandwidth, spectral_flatness])
        
        # RMS Energy - 1维
        rms = librosa.feature.rms(y=y, hop_length=hop_length)
        features_list.append(rms)
        
        # 合并特征 (总共: 40+40+12+7+6+1+4+1 = 111维)
        features = np.vstack(features_list)
        
        # 添加 delta 特征来达到 154 维
        # 111 + 43 = 154
        delta = librosa.feature.delta(features)
        features = np.vstack([features, delta[:43, :]])
        
        return features.T  # (n_frames, 154)
        
    except Exception as e:
        print(f"    特征提取错误: {e}", file=sys.stderr)
        return None

def process_dataset(video_dir, json_path, output_pkl_path, temp_audio_dir, dataset_name):
    """处理单个数据集"""
    print("\n" + "=" * 70)
    print(f"处理 {dataset_name} 数据集")
    print("=" * 70)
    print(f"  视频目录: {video_dir}")
    print(f"  JSON文件: {json_path}")
    print(f"  输出文件: {output_pkl_path}")
    
    # 加载 JSON
    with open(json_path, 'r', encoding='utf-8') as f:
        data_dict = json.load(f)
    print(f"  JSON包含 {len(data_dict)} 个样本\n")
    
    # 处理视频
    data_list = []
    success_count = 0
    failed_count = 0
    
    for idx, (video_id, info) in enumerate(data_dict.items(), 1):
        video_path = os.path.join(video_dir, f"{video_id}.mp4")
        audio_path = os.path.join(temp_audio_dir, f"{video_id}.wav")
        
        print(f"  [{idx}/{len(data_dict)}] {video_id}", end=" ... ")
        
        if not os.path.exists(video_path):
            print("视频不存在")
            failed_count += 1
            continue
        
        # 提取音频
        if not os.path.exists(audio_path):
            if not extract_audio_from_video_ffmpeg(video_path, audio_path):
                print("音频提取失败")
                failed_count += 1
                continue
        
        # 提取特征
        features = extract_acoustic_features(audio_path)
        if features is None:
            print("特征提取失败")
            failed_count += 1
            continue
        
        # 获取时长
        try:
            y, sr = librosa.load(audio_path, sr=None)
            duration = librosa.get_duration(y=y, sr=sr)
        except:
            duration = 0.0
        
        # 解析时间字符串 (格式: "00:00:00")
        start_time_str = info.get('start_time', '00:00:00')
        end_time_str = info.get('end_time', '00:00:00')
        
        # 构建数据行 (与 CCG-Net 格式一致)
        row = {
            'episode_name': info.get('episode_name', ''),
            'target_speaker': info.get('target_speaker', ''),
            'target_utterance': info.get('target_utterance', ''),
            'context_speakers': info.get('context_speakers', []),
            'context_utterances': info.get('context_utterances', []),
            'sarcasm_target': info.get('sarcasm_target', 'UNKNOWN'),
            'sarcasm_type': info.get('sarcasm_type', 'UNKNOWN'),
            'english_explanation': info.get('english_explanation', ''),
            'code_mixed_explanation': info.get('code_mixed_explanation', ''),
            'start_time': start_time_str,
            'end_time': end_time_str,
            'audio_feats': features
        }
        
        data_list.append(row)
        success_count += 1
        print(f"✓ {features.shape}")
    
    # 保存结果
    print("\n" + "-" * 70)
    print(f"  成功: {success_count}, 失败: {failed_count}")
    
    if data_list:
        df = pd.DataFrame(data_list)
        print(f"  DataFrame 形状: {df.shape}")
        
        with open(output_pkl_path, 'wb') as f:
            pickle.dump(df, f)
        
        print(f"  ✓ 已保存到: {output_pkl_path}")
        
        # 显示统计信息
        print(f"\n  特征统计:")
        print(f"    平均帧数: {np.mean([f.shape[0] for f in df['audio_feats']]):.1f}")
        print(f"    特征维度: {df['audio_feats'].iloc[0].shape[1]}")
        file_size_mb = os.path.getsize(output_pkl_path) / (1024**2)
        print(f"    文件大小: {file_size_mb:.2f} MB")
    else:
        print("  ✗ 没有成功处理任何样本")
    
    return success_count, failed_count

def main():
    """主函数"""
    print("=" * 70)
    print("MMSD 数据集音频特征提取")
    print("=" * 70)
    
    # JSON 和输出配置 (每个数据集有自己的视频目录)
    datasets = [
        {
            'name': 'TEST',
            'video_dir': '/home/ubuntu/csp/MUStARD-master/data/videos/test',
            'json': '/home/ubuntu/csp/CCG-Net/Data_MMSD/Text_after/test_data_final_plain.json',
            'output': '/home/ubuntu/csp/CCG-Net/Data_MMSD/Audio/test_acoustic_features.pkl'
        },
        {
            'name': 'TRAIN',
            'video_dir': '/home/ubuntu/csp/MUStARD-master/data/videos/train',
            'json': '/home/ubuntu/csp/CCG-Net/Data_MMSD/Text_after/train_data_final_plain.json',
            'output': '/home/ubuntu/csp/CCG-Net/Data_MMSD/Audio/train_acoustic_features.pkl'
        },
        {
            'name': 'VAL',
            'video_dir': '/home/ubuntu/csp/MUStARD-master/data/videos/val',
            'json': '/home/ubuntu/csp/CCG-Net/Data_MMSD/Text_after/val_data_final_plain.json',
            'output': '/home/ubuntu/csp/CCG-Net/Data_MMSD/Audio/val_acoustic_features.pkl'
        }
    ]
    
    # 临时音频目录
    temp_audio_dir = '/home/ubuntu/csp/temp_audio_mmsd'
    Path(temp_audio_dir).mkdir(exist_ok=True)
    print(f"\n临时音频目录: {temp_audio_dir}\n")
    
    # 统计信息
    total_success = 0
    total_failed = 0
    
    # 处理每个数据集
    for ds in datasets:
        # 检查视频目录
        if not os.path.exists(ds['video_dir']):
            print(f"错误: 视频目录不存在 - {ds['video_dir']}")
            continue
        
        if not os.path.exists(ds['json']):
            print(f"警告: JSON 文件不存在 - {ds['json']}")
            continue
        
        success, failed = process_dataset(
            ds['video_dir'],
            ds['json'],
            ds['output'],
            temp_audio_dir,
            ds['name']
        )
        
        total_success += success
        total_failed += failed
    
    # 总结
    print("\n" + "=" * 70)
    print("处理完成!")
    print("=" * 70)
    print(f"  总成功: {total_success}")
    print(f"  总失败: {total_failed}")
    print(f"  成功率: {total_success/(total_success+total_failed)*100:.1f}%")
    
    # 清理临时文件
    print(f"\n清理临时文件: {temp_audio_dir}")
    import shutil
    shutil.rmtree(temp_audio_dir, ignore_errors=True)
    
    print("\n✓ 全部完成！")

if __name__ == '__main__':
    main()

