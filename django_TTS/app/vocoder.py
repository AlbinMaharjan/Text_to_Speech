import os
import sys
import time
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import resampy
import scipy.signal
import soundfile as sf  # For saving audio files
from tqdm import tqdm


def text_to_speech_synthesis(input_text):
    """
    Synthesizes speech from text using Tacotron2 and HiFi-GAN.

    Parameters:
        input_text (str): The text to be synthesized into speech.

    Outputs:
        Saves the generated audio to 'output.wav' and displays two mel spectrograms.
    """
    # Set paths to models
    BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))  # Adjust if running elsewhere
    TTS_PATH = os.path.join(BASE_PATH, "TTS_TT2")
    HIFI_GAN_PATH = os.path.join(BASE_PATH, "hifi_gan")

    # Add project directories to Python path
    sys.path.append('D:\\\\Project\\\\text_to_speech\\\\hifi_gan')
    sys.path.append('D:\\\\Project\\\\text_to_speech\\\\TTS_TT2')
    sys.path.append(BASE_PATH)

    # Import Tacotron2 and HiFi-GAN components
    from text_to_speech.TTS_TT2.hparams import create_hparams
    from text_to_speech.TTS_TT2.model import Tacotron2
    from text_to_speech.TTS_TT2.layers import TacotronSTFT
    from text_to_speech.hifi_gan.audio_processing import griffin_lim
    from text_to_speech.TTS_TT2.text import text_to_sequence
    from text_to_speech.hifi_gan.env import AttrDict
    from text_to_speech.hifi_gan.meldataset import mel_spectrogram, MAX_WAV_VALUE
    from text_to_speech.hifi_gan.models import Generator
    from text_to_speech.hifi_gan.denoiser import Denoiser

    # Initialize and load pronunciation dictionary
    
    PRON_DICT_FILE = os.path.join(BASE_PATH, "merged.dict.txt")

    if not os.path.exists(PRON_DICT_FILE):
        print("Downloading pronunciation dictionary...")
        
#maps words to ARPAbetpronouncation dictionary
    pronunciation_dict = {}
    with open(PRON_DICT_FILE, "r") as f:
        for line in f:
            key, value = line.strip().split(" ", 1)
            pronunciation_dict[key] = value.strip()
#Converts input text to ARPAbet format for improved phoneme-level accuracy.
#Words not in the dictionary are left unchanged.

    # Pronunciation conversion function
    def ARPA(text, punctuation=r"!?,.;", EOS_Token=True):
        out = ''
        for word in text.split():
            end_chars = ''.join(c for c in word if c in punctuation)
            word = word.rstrip(end_chars)
            try:
                word = "{" + pronunciation_dict[word.upper()] + "}"
            except KeyError:
                pass
            out += " " + word + end_chars
        return (out + ";") if EOS_Token and out[-1] != ";" else out

    # Load HiFi-GAN model
    def get_hifigan(config_name):
        model_file = os.path.join("D:\\\\Project\\\\text_to_speech\\\\hifi_gan\\\\", f"{config_name}.pth")
        
        with open(os.path.join("D:\\\\Project\\\\text_to_speech\\\\hifi_gan\\\\", f"{config_name}.json")) as f:
            h = AttrDict(json.load(f)) # Loads model hyperparameters

        hifigan = Generator(h).to("cpu") # Initializes HiFi-GAN generator
        state_dict = torch.load(model_file, map_location="cpu")
        hifigan.load_state_dict(state_dict["generator"])
        hifigan.eval()
        hifigan.remove_weight_norm()   # Removes weight normalization (important for inference)

        return hifigan, h, Denoiser(hifigan, mode="normal")

    # Load Tacotron2 model
    def get_tacotron2(model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Tacotron2 model not found: {model_path}")
# Provides the core Tacotron2 architecture used to generate mel spectrograms from input text.
        hparams = create_hparams()
        hparams.sampling_rate = 22050
        hparams.max_decoder_steps = 3000 #model lai indefinitely run huna bata bachauxa
        hparams.gate_threshold = 0.25 #if predicted value yo bhanda badi gaye stop hunxa

        model = Tacotron2(hparams).to("cpu")
        state_dict = torch.load(model_path, map_location="cpu")["state_dict"]
        model.load_state_dict(state_dict)
        model.eval()

        return model, hparams

    # Inference function
    #Synthesizes speech using Tacotron2 and HiFi-GAN.
    def infer_tts(model, hparams, hifigan, text, pre_net_output="pre_net_mel_spectrogram.png", post_net_output="post_net_mel_spectrogram.png"):
       #Converts input text to a sequence of numerical IDs.
        sequence = np.array(text_to_sequence(text, ["english_cleaners"]))[None, :]
        sequence = torch.from_numpy(sequence).long() #64 bit ma lagxa

        # Tacotron2 inference
        with torch.no_grad():
            mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence)

        # Save pre-net Mel spectrogram as PNG
        plt.figure(figsize=(10, 5))
        plt.imshow(mel_outputs.cpu().numpy()[0], aspect="auto", origin="lower", cmap="inferno")
        plt.title("Pre-net Mel Spectrogram")
        plt.colorbar()
        plt.savefig(pre_net_output)
        plt.close()
        print(f"Pre-net mel spectrogram saved as {pre_net_output}")

        # Save post-net Mel spectrogram as PNG
        plt.figure(figsize=(10, 5))
        plt.imshow(mel_outputs_postnet.cpu().numpy()[0], aspect="auto", origin="lower", cmap="inferno")
        plt.title("Post-net Mel Spectrogram")
        plt.colorbar()
        plt.savefig(post_net_output)
        plt.close()
        print(f"Post-net mel spectrogram saved as {post_net_output}")

        # HiFi-GAN inference
        with torch.no_grad():
            audio = hifigan(mel_outputs_postnet).squeeze().cpu().numpy()

        return audio, hparams.sampling_rate

    # Paths to models
    TACO_MODEL_PATH = os.path.join(BASE_PATH, "tacotron2_model.pth")
    HIFI_GAN_CONFIG = "config_v1"

    print("Loading Tacotron2...")
    tacotron2, hparams = get_tacotron2(TACO_MODEL_PATH)

    print("Loading HiFi-GAN...")
    hifigan, h, denoiser = get_hifigan(HIFI_GAN_CONFIG)

    # Convert input text
    text = ARPA(input_text)

    print("Synthesizing speech...")
    audio, sampling_rate = infer_tts(tacotron2, hparams, hifigan, text)

    # Save output audio
    OUTPUT_FILE = "D:\\Project\\text_to_speech\\Scripts\\django_TTS\\app\\static\\output.wav"
    sf.write(OUTPUT_FILE, audio, sampling_rate)
    print(f"Audio saved to {OUTPUT_FILE}")
