# Voice Style Transfer using Speech Recognition and Voice Cloning
Major Project for B.Tech IVth Year

Step 0.1: Make sure the following libraries are installed in conda env: pytorch, torchvision, torchaudio, pytorch-cuda=11.7
          If not then run => conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
          
Step 0.2: Download the required packages using the following command => pip install -r requirements.txt

Step 1: Run python preprocess_dataset.py

Step 2: Run python resample.py

Step 3: Run python preprocess_flist_config.py --speech_encoder hubertsoft --vol_aug

Step 4: Run python preprocess_hubert_f0.py --f0_predictor crepe

Step 5: Run python train.py -c configs/config.json -m 44k

Step 6: Run python inference_main.py -m "logs/44k/G_example.pth" -c "configs/config.json" -n "raw/example_song.wav" -t 0 -s "singer_name"
