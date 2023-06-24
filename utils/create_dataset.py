import os
from pydub import AudioSegment

CLIP_DURATION = 10 

def create_dataset(vocals_path, dataset_path):

    for i, (dirpath, _, filenames) in enumerate(os.walk(vocals_path)):

        if dirpath is not vocals_path:

            singer = dirpath.split("\\")[-1]
            singer_path = dataset_path + '/' + singer
            if not os.path.exists(singer_path):
                os.makedirs(singer_path)

            concat_audio = AudioSegment.empty()

            print(f"Concatenating {singer}'s vocal audio files...")

            for file in filenames:
                filepath  = vocals_path + "/" + singer + "/" + file
                audio = AudioSegment.from_file(filepath)
                concat_audio += audio

            total_duration = len(concat_audio)
            duration = CLIP_DURATION * 1000

            num_chunks = total_duration // duration

            print(f"Splitting dataset into {num_chunks} audio clips, each {CLIP_DURATION} sec...")

            for i in range(num_chunks):

                start_time = i * duration
                end_time = start_time + duration

                clip = concat_audio[start_time:end_time]

                audio_file = os.path.join(singer_path, f"{i}.wav")
                clip.export(audio_file, format='wav')