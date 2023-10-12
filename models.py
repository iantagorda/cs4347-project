import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import Dataset


class DataCollator:
    def __init__(self, processor, padding, device):
        self.processor = processor
        self.padding = padding
        self.device = device

    def __call__(self, data):
        waveforms, lm_labels, accent_labels, gender_labels = zip(*data)
        accent_labels = torch.tensor(accent_labels, device=self.device)
        gender_labels = torch.tensor(gender_labels, device=self.device)

        input_features = [
            {"input_values": waveform.squeeze()} for waveform in waveforms
        ]
        label_features = [{"input_ids": lm_label} for lm_label in lm_labels]

        padded_waveforms = self.processor.pad(
            input_features,
            padding=True,
            return_tensors="pt",
        )["input_values"]
        padded_waveforms = padded_waveforms.to(self.device)

        with self.processor.as_target_processor():
            padded_lm_labels = self.processor.pad(
                label_features,
                padding=True,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        padded_lm_labels = padded_lm_labels["input_ids"].masked_fill(
            padded_lm_labels.attention_mask.ne(1), -100
        )
        padded_lm_labels = padded_lm_labels.to(self.device)

        return padded_waveforms, padded_lm_labels, accent_labels, gender_labels


class L2ArcticDataset(Dataset):
    def __init__(self, processor, audio_paths, lm_labels, accent_labels, gender_labels):
        orig_sampling_rate = 44100
        new_sampling_rate = 16000
        resample_transform = torchaudio.transforms.Resample(
            orig_sampling_rate, new_sampling_rate
        )

        self.waveforms = []
        self.lm_labels = []
        self.accent_labels = accent_labels
        self.gender_labels = gender_labels

        for audio_path in audio_paths:
            waveform, _ = torchaudio.load(audio_path)
            waveform = resample_transform(waveform)
            self.waveforms.append(
                processor(waveform, sampling_rate=new_sampling_rate).input_values[0]
            )

        with processor.as_target_processor():
            for lm_label in lm_labels:
                self.lm_labels.append(processor(lm_label).input_ids)

    def __getitem__(self, index):
        return (
            self.waveforms[index],
            self.lm_labels[index],
            self.accent_labels[index],
            self.gender_labels[index],
        )

    def __len__(self):
        return len(self.waveforms)


class MultiTaskWav2Vec2(nn.Module):
    def __init__(self, wav2vec2_backbone, num_accent_class):
        super().__init__()
        self.wav2vec2 = wav2vec2_backbone
        self.accent_projector = nn.Linear(768, 256)
        self.accent_classifier = nn.Linear(256, num_accent_class)
        self.gender_projector = nn.Linear(768, 256)
        self.gender_classifier = nn.Linear(256, 2)

    def forward(self, waveform, lm_labels):
        # use hugging face wav2vecc2
        wav2vec2_output = self.wav2vec2(input_values=waveform, labels=lm_labels)

        # get partial loss based (lm_head loss or the ctc loss)
        ctc_loss = wav2vec2_output.loss

        # get features from wav2vec2
        features = wav2vec2_output.hidden_states[-1]

        # get output lm logits
        lm_logits = wav2vec2_output.logits

        # get output accent logits
        accent_projected = self.accent_projector(features)
        accent_projected = accent_projected.mean(dim=1)
        accent_logits = self.accent_classifier(accent_projected)

        # get output gender logits
        gender_projected = self.gender_projector(features)
        gender_projected = gender_projected.mean(dim=1)
        gender_logits = self.gender_classifier(gender_projected)

        return ctc_loss, lm_logits, accent_logits, gender_logits
