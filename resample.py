import os
import argparse
import librosa
import numpy as np
from multiprocessing import Pool, cpu_count
from scipy.io import wavfile
from tqdm import tqdm


def process(item):
    spkdir, wav_name, args = item
    # speaker 's5', 'p280', 'p315' are excluded,
    speaker = spkdir.replace("\\", "/").split("/")[-1]
    wav_path = os.path.join(args.in_dir, speaker, wav_name)
    if os.path.exists(wav_path) and '.wav' in wav_path:
        os.makedirs(os.path.join(args.out_dir2, speaker), exist_ok=True)
        wav, sr = librosa.load(wav_path, sr=None)
        wav, _ = librosa.effects.trim(wav, top_db=40)
        peak = np.abs(wav).max()
        if peak > 1.0:
            wav = 0.98 * wav / peak
        wav2 = librosa.resample(wav, orig_sr=sr, target_sr=args.sr2)
        if not args.skip_loudnorm:
            wav2 /= max(wav2.max(), -wav2.min())
        save_name = wav_name
        save_path2 = os.path.join(args.out_dir2, speaker, save_name)
        wavfile.write(
            save_path2,
            args.sr2,
            (wav2 * np.iinfo(np.int16).max).astype(np.int16)
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sr2", type=int, default=44100, help="sampling rate")
    parser.add_argument("--in_dir", type=str, default="audio_files/dataset", help="path to source dir")
    parser.add_argument("--out_dir2", type=str, default="./dataset/44k", help="path to target dir")
    parser.add_argument("--skip_loudnorm", action="store_true", help="Skip loudness matching if you have done it")
    args = parser.parse_args()
    processs = 30 if cpu_count() > 60 else (cpu_count()-2 if cpu_count() > 4 else 1)
    pool = Pool(processes=processs)

    for speaker in os.listdir(args.in_dir):
        spk_dir = os.path.join(args.in_dir, speaker)
        if os.path.isdir(spk_dir):
            print(spk_dir)
            for _ in tqdm(pool.imap_unordered(process, [(spk_dir, i, args) for i in os.listdir(spk_dir) if i.endswith("wav")])):
                pass