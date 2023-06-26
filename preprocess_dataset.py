import os, shutil, argparse
from utils.get_vocals import get_vocals
from utils.create_dataset import create_dataset

SONGS_PATH = "./audio_files/songs"
VOCALS_PATH = "./audio_files/vocals_only"
DATASET_PATH = "./audio_files/dataset"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--songs_path", type=str, default=SONGS_PATH, help="songs directory")
    parser.add_argument("--vocals_path", type=str, default=VOCALS_PATH, help="vocals only directory")
    parser.add_argument("--dataset_path", type=str, default=DATASET_PATH, help="dataset directory")
    args = parser.parse_args()

    get_vocals(args.songs_path, args.vocals_path)
    create_dataset(args.vocals_path, args.dataset_path)
    if os.path.exists(args.vocals_path): shutil.rmtree(args.vocals_path)