import os
import subprocess
import shutil
import sys

def get_vocals(songs_path, vocals_path):

    for i, (dirpath, _, filenames) in enumerate(os.walk(songs_path)):

        if dirpath is not songs_path:

            singer = dirpath.split("\\")[-1]
            singer_path = vocals_path + '/' + singer
            if not os.path.exists(singer_path):
                os.makedirs(singer_path)

            print(f"Processing {singer} directory...")

            for mp3file in filenames:

                file = mp3file.replace('.mp3', '')
                filepath  = songs_path + "/" + singer + "/" + mp3file

                print(f"Extracting vocals from {file} file...")

                vocals_old = singer_path + "/vocals.wav"
                vocals_new = singer_path + f"/{file}.wav"

                if os.path.exists(vocals_old):
                    print(f'ERROR: Delete "vocals.wav" in "{singer_path}"')
                    sys.exit()

                elif os.path.exists(vocals_new):
                   print(f'"{file}.wav" already exists in "{singer_path}"')

                else: 
                    command = f"spleeter separate -o temp/ {filepath}"
                    process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
                    _, _ = process.communicate()

                    vocals_temp = f"temp/{file}/vocals.wav"

                    shutil.move(vocals_temp, singer_path)
                    os.rename(vocals_old, vocals_new)

                    shutil.rmtree(f"temp/{file}", ignore_errors=True)


    if os.path.exists("temp"):
        shutil.rmtree("temp", ignore_errors=True)