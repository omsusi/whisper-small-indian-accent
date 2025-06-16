Whisper Small Model - Indian Accent
This repository hosts a fine-tuned version of OpenAI's Whisper Small model, specifically adapted for improved speech recognition of Indian English accents. This model aims to provide higher accuracy for transcribing audio that contains the unique phonetic and linguistic characteristics of Indian English speech.

🚀 Model Details
Base Model: OpenAI Whisper Small

Fine-tuning Objective: Enhanced transcription accuracy for Indian English.

Model Weights Format: safetensors (for improved security and loading speed)

📊 Training Details
Dataset Used for Training
Due to resource constraints (the full Opus version being ~100GB and not feasible for free Google Colab training), we utilized a sampled subset of the NPTEL-2020 Indian English Speech Dataset.

Sample Dataset Download: https://github.com/AI4Bharat/NPTEL2020-Indian-English-Speech-Dataset/releases/download/v0.1/nptel-pure-set.tar.gz

Dataset Description: The NPTEL-2020 dataset comprises speech from lectures delivered by Indian professors, providing a valuable source of Indian English speech.

Training Environment
[Optional: Briefly mention the hardware used, e.g., "Fine-tuned on Google Colab Pro+ GPU (A100)" or "Trained on a custom GPU setup with NVIDIA RTX 3090."]

[Optional: Briefly mention libraries or frameworks used, e.g., "Developed using Hugging Face Transformers and PyTorch."]

✨ Performance
Our custom-trained Whisper-small model significantly outperforms the pre-trained Whisper-small model on a custom validation dataset. The substantial reduction in both Word Error Rate (WER) and Character Error Rate (CER) demonstrates the effectiveness of domain-specific fine-tuning for Indian English accents.

Evaluation Results on Custom Validation Dataset
Model

WER (Word Error Rate)

CER (Character Error Rate)

Pre-trained Whisper-small

32.1

12.3

Custom-trained Whisper-small

15.6

7.8

Custom Dataset Used for Testing
To validate the model's performance on realistic Indian English speech, we used a custom dataset comprising our own voices.

Custom Audio Data: https://drive.google.com/drive/folders/1bKVak_v3T-qtyEzdIY7AkDQ57ap18J2o?usp=sharing

Required JSON File for Testing: https://drive.google.com/file/d/1bX1sjeRVEVhqoWwfVYm-IEE7K-v-AevR/view?usp=sharing

Testing Colab Notebook
You can reproduce our testing and evaluate the model's performance yourself using the following Google Colab notebook:

Colab Notebook: https://colab.research.google.com/drive/1zcL9dbifU2ZenJIjYOeSVkOwheB_u281?usp=sharing

🛠️ Usage
This model can be easily loaded and used with the Hugging Face transformers library.

1. Installation:

First, ensure you have the necessary libraries installed:

pip install transformers accelerate safetensors datasets soundfile # Or just pip install transformers[torch] accelerate

2. Load the Model and Processor:

You can load the model directly from this GitHub repository (once uploaded) or from its corresponding Hugging Face Hub page.

from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
import torch
import librosa # You might need to pip install librosa

# Specify the model ID (replace with your actual GitHub/Hugging Face path)
# After uploading to GitHub, this would be:
model_id = "your-github-username/whisper-small-indian-accent"
# If you also upload to Hugging Face Hub (highly recommended for ML models):
# model_id = "your-huggingface-username/whisper-small-indian-accent"

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else "cpu" # Changed float32 to cpu for consistency
# Make sure your audio is sampled at 16kHz
sample_rate = 16000
duration = 5 # seconds
dummy_audio = torch.randn(1, sample_rate * duration).numpy() # Example: 5 seconds of random noise


processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
).to(device)


# Process dummy audio
input_features = processor(
    dummy_audio,
    sampling_rate=sample_rate,
    return_tensors="pt"
).input_features.to(device)

# Generate transcription
predicted_ids = model.generate(input_features)
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

print(f"Transcription: {transcription}")

# Example for a real audio file (uncomment and replace path):
# audio, sampling_rate = librosa.load("path/to/your/indian_accent_audio.wav", sr=16000)
# input_features = processor(audio, sampling_rate=sampling_rate, return_tensors="pt").input_features.to(device)
# predicted_ids = model.generate(input_features)
# transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
# print(f"Transcription: {transcription}")

🙏 Attribution
If you use this model in your research, projects, or applications, please ensure you provide appropriate credit to the original creators, as required by the CC BY 4.0 license.

This includes:

Omsubhra Singha

Aman Kumar

You can provide credit by:

Linking to this GitHub repository.

Mentioning our names in your project's documentation, acknowledgements, or credits section.

Citing the model using the information provided in the CITATION.cff file (if applicable for academic use).

Your adherence to these attribution requirements, as per the CC BY 4.0 license, is greatly appreciated.

📜 License
This Whisper Small Model (Indian Accent) is distributed under the Creative Commons Attribution 4.0 International Public License (CC BY 4.0).

You are free to:

Share — copy and redistribute the material in any medium or format for any purpose, even commercially.

Adapt — remix, transform, and build upon the material for any purpose, even commercially.

Under the following terms:

Attribution — You must give appropriate credit, provide a link to the license, and indicate if changes were made. You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.

For the full text of the license, please see the LICENSE file in this repository or visit: https://creativecommons.org/licenses/by/4.0/

📞 Contact
For questions or inquiries, please open an issue in this repository or contact https://www.linkedin.com/in/omsubhra-singha-30447a254/ and amnkmr2098@gmail.com.