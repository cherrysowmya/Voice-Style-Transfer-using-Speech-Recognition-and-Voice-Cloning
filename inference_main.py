import io
import logging
import time
from pathlib import Path
# from spkmix import spk_mix_map
import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile
from inference import infer_tool
from inference import slicer
from inference.infer_tool import Svc

logging.getLogger('numba').setLevel(logging.WARNING)
chunks_dict = infer_tool.read_temp("inference/chunks_temp.json")



def main():
    import argparse

    parser = argparse.ArgumentParser(description='sovits4 inference')

    # Required parameters
    parser.add_argument('-m', '--model_path', type=str, default="logs/44k/G_37600.pth", help='model path')
    parser.add_argument('-c', '--config_path', type=str, default="logs/44k/config.json", help='config file path')
    parser.add_argument('-cl', '--clip', type=float, default=0, help='Forced audio slice, default 0 is automatic slice, unit is second/s')
    parser.add_argument('-n', '--clean_names', type=str, nargs='+', default=["input.wav"], help='List of wav file names, placed in the raw folder')
    parser.add_argument('-t', '--trans', type=int, nargs='+', default=[0], help='Pitch adjustment, support positive and negative (semitone)')
    parser.add_argument('-s', '--spk_list', type=str, nargs='+', default=['buyizi'], help='synthetic target speaker name')
    
    # Optional parameters
    parser.add_argument('-a', '--auto_predict_f0', action='store_true', default=False, help='The voice conversion automatically predicts the pitch, do not turn on this when converting the singing voice, it will be seriously out of tune')
    parser.add_argument('-cm', '--cluster_model_path', type=str, default="logs/44k/kmeans_10000.pt", help='Clustering model or feature retrieval index path, if no clustering or feature retrieval is trained, fill it in casually')
    parser.add_argument('-cr', '--cluster_infer_ratio', type=float, default=0, help='Clustering scheme or feature retrieval ratio, range 0-1, if no clustering model or feature retrieval is trained, the default is 0')
    parser.add_argument('-lg', '--linear_gradient', type=float, default=0, help='The cross-fade length of two audio slices. If the human voice is incoherent after forcing the slice, you can adjust this value. If it is coherent, it is recommended to use the default value of 0, and the unit is second')
    parser.add_argument('-f0p', '--f0_predictor', type=str, default="crepe", help='Select the F0 predictor, you can choose crepe, crepe, dio, harvest, the default is pm (note: crepe uses the mean filter for the original F0)')
    parser.add_argument('-eh', '--enhance', action='store_true', default=False, help='Whether to use the NSF_HIFIGAN enhancer, this option has a certain sound quality enhancement effect on some models with a small training set, but has a negative effect on the trained model, and it is disabled by default')
    parser.add_argument('-shd', '--shallow_diffusion', action='store_true', default=False, help='Whether to use shallow diffusion. After using it, it can solve some electronic audio problems. It is disabled by default. When this option is enabled, the NSF_HIFIGAN enhancer will be disabled')
    parser.add_argument('-usm', '--use_spk_mix', action='store_true', default=False, help='Whether to use role fusion')
    parser.add_argument('-lea', '--loudness_envelope_adjustment', type=float, default=1, help='The input source loudness envelope replaces the output loudness envelope fusion ratio, the closer to 1, the more the output loudness envelope is used')
    parser.add_argument('-fr', '--feature_retrieval', action='store_true', default=False, help='Whether to use feature retrieval, if the clustering model is used, it will be disabled, and the cm and cr parameters will become the index path and mixing ratio of feature retrieval')

    # Shallow diffusion settings
    parser.add_argument('-dm', '--diffusion_model_path', type=str, default="logs/44k/diffusion/model_0.pt", help='Diffusion Model Path')
    parser.add_argument('-dc', '--diffusion_config_path', type=str, default="logs/44k/diffusion/config.yaml", help='Diffusion model config file path')
    parser.add_argument('-ks', '--k_step', type=int, default=100, help='The number of diffusion steps, the larger the result is closer to the diffusion model, the default is 100')
    parser.add_argument('-se', '--second_encoding', action='store_true', default=False, help='Secondary encoding, the original audio will be encoded twice before shallow diffusion, metaphysical option, sometimes the effect is good, sometimes the effect is poor')
    parser.add_argument('-od', '--only_diffusion', action='store_true', default=False, help='Pure diffusion mode, this mode will not load the sovits model, reasoning with the diffusion model')
    

    # Fixed settings
    parser.add_argument('-sd', '--slice_db', type=int, default=-40, help='Default -40, noisy audio can be -30, dry sound can keep breathing -50')
    parser.add_argument('-d', '--device', type=str, default=None, help='Inference device, if None is to automatically select cpu and gpu')
    parser.add_argument('-ns', '--noice_scale', type=float, default=0.4, help='The noise level will affect the articulation and sound quality, which is more metaphysical')
    parser.add_argument('-p', '--pad_seconds', type=float, default=0.5, help='Inferring the number of seconds of the audio pad, there will be abnormal noise at the beginning and end due to unknown reasons, and the pad will not appear after a short period of silence')
    parser.add_argument('-wf', '--wav_format', type=str, default='wav', help='audio output format')
    parser.add_argument('-lgr', '--linear_gradient_retain', type=float, default=0.75, help='After automatic audio slicing, the head and tail of each slice need to be discarded. This parameter sets the ratio of cross length retention, range 0-1, left open and right closed')
    parser.add_argument('-eak', '--enhancer_adaptive_key', type=int, default=0, help='Adapt the enhancer to a higher register (in semitones) | default 0')
    parser.add_argument('-ft', '--f0_filter_threshold', type=float, default=0.05,help='F0 filter threshold, only valid when using crepe. The value ranges from 0-1. Lowering this value can reduce the probability of out-of-tune, but it will increase mute')


    args = parser.parse_args()

    clean_names = args.clean_names
    trans = args.trans
    spk_list = args.spk_list
    slice_db = args.slice_db
    wav_format = args.wav_format
    auto_predict_f0 = args.auto_predict_f0
    cluster_infer_ratio = args.cluster_infer_ratio
    noice_scale = args.noice_scale
    pad_seconds = args.pad_seconds
    clip = args.clip
    lg = args.linear_gradient
    lgr = args.linear_gradient_retain
    f0p = args.f0_predictor
    enhance = args.enhance
    enhancer_adaptive_key = args.enhancer_adaptive_key
    cr_threshold = args.f0_filter_threshold
    diffusion_model_path = args.diffusion_model_path
    diffusion_config_path = args.diffusion_config_path
    k_step = args.k_step
    only_diffusion = args.only_diffusion
    shallow_diffusion = args.shallow_diffusion
    use_spk_mix = args.use_spk_mix
    second_encoding = args.second_encoding
    loudness_envelope_adjustment = args.loudness_envelope_adjustment

    svc_model = Svc(args.model_path,
                    args.config_path,
                    args.device,
                    args.cluster_model_path,
                    enhance,
                    diffusion_model_path,
                    diffusion_config_path,
                    shallow_diffusion,
                    only_diffusion,
                    use_spk_mix,
                    args.feature_retrieval)
    
    infer_tool.mkdir(["raw", "results"])
    
    # if len(spk_mix_map)<=1:
    #     use_spk_mix = False
    if use_spk_mix:
        spk_list = [spk_mix_map]
    
    infer_tool.fill_a_to_b(trans, clean_names)
    for clean_name, tran in zip(clean_names, trans):
        raw_audio_path = f"raw/{clean_name}"
        if "." not in raw_audio_path:
            raw_audio_path += ".wav"
        infer_tool.format_wav(raw_audio_path)
        for spk in spk_list:
            kwarg = {
                "raw_audio_path" : raw_audio_path,
                "spk" : spk,
                "tran" : tran,
                "slice_db" : slice_db,
                "cluster_infer_ratio" : cluster_infer_ratio,
                "auto_predict_f0" : auto_predict_f0,
                "noice_scale" : noice_scale,
                "pad_seconds" : pad_seconds,
                "clip_seconds" : clip,
                "lg_num": lg,
                "lgr_num" : lgr,
                "f0_predictor" : f0p,
                "enhancer_adaptive_key" : enhancer_adaptive_key,
                "cr_threshold" : cr_threshold,
                "k_step":k_step,
                "use_spk_mix":use_spk_mix,
                "second_encoding":second_encoding,
                "loudness_envelope_adjustment":loudness_envelope_adjustment
            }
            audio = svc_model.slice_inference(**kwarg)
            key = "auto" if auto_predict_f0 else f"{tran}key"
            cluster_name = "" if cluster_infer_ratio == 0 else f"_{cluster_infer_ratio}"
            isdiffusion = "sovits"
            if shallow_diffusion : isdiffusion = "sovdiff"
            if only_diffusion : isdiffusion = "diff"
            if use_spk_mix:
                spk = "spk_mix"
            song_name = clean_name.replace('.wav', '')
            res_path = f'results/{song_name}_{spk}.{wav_format}'
            soundfile.write(res_path, audio, svc_model.target_sample, format=wav_format)
            svc_model.clear_empty()
            
if __name__ == '__main__':
    main()