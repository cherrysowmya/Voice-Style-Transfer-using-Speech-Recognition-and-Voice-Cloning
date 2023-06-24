import os, shutil
from utils.get_vocals import get_vocals
from utils.create_dataset import create_dataset

SONGS_PATH = "audio_files/songs"
VOCALS_PATH = "audio_files/vocals_only"
DATASET_PATH = "audio_files/dataset"

if __name__ == "__main__":
    get_vocals(SONGS_PATH, VOCALS_PATH)
    create_dataset(VOCALS_PATH, DATASET_PATH)
    if os.path.exists(VOCALS_PATH): shutil.rmtree(VOCALS_PATH)