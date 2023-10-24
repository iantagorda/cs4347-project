import random

import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import Dataset


class DataCollator:
    def __init__(self, processor, padding, device, augment):
        self.processor = processor
        self.padding = padding
        self.device = device
        self.sampling_rate = 16000
        self.augment = augment

        atempos = (0.8, 1.0, 1.25)  # audio tempo atempo=tempo
        audio_effects = (
            ("highpass=frequency=1500",),
            (
                "vibrato=f=5:d=0.4",
                "volume=1.5",
            ),
            (
                "aecho=0.8:0.88:30:0.3",
                "volume=1.5",
            ),
        )

        self.effectors = [None]
        for atempo in atempos:
            for audio_effect in audio_effects:
                effect = f"atempo={atempo}," + ",".join(audio_effect)
                self.effectors.append(torchaudio.io.AudioEffector(effect=effect))

    def __call__(self, data):
        waveforms, lm_labels, accent_labels, gender_labels = zip(*data)
        accent_labels = torch.tensor(accent_labels, device=self.device)
        gender_labels = torch.tensor(gender_labels, device=self.device)

        input_features = [
            {"input_values": self.random_augment(waveform).squeeze()}
            for waveform in waveforms
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

    def random_augment(self, waveform):
        if not self.augment:
            return waveform

        waveform = torch.tensor(waveform)
        waveform = torch.transpose(waveform, 0, 1)
        effector = random.choice(self.effectors)
        if effector is None:
            return waveform

        augmented_waveform = effector.apply(waveform, self.sampling_rate)
        if augmented_waveform.isnan().any() | augmented_waveform.isinf().any():
            return waveform

        return augmented_waveform


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
    def __init__(
        self,
        wav2vec2_backbone,
        backbone_hidden_size,
        projection_hidden_size,
        num_accent_class,
    ):
        super().__init__()
        self.wav2vec2 = wav2vec2_backbone
        self.accent_projector = nn.Linear(backbone_hidden_size, projection_hidden_size)
        self.accent_classifier = nn.Linear(projection_hidden_size, num_accent_class)
        self.gender_projector = nn.Linear(backbone_hidden_size, projection_hidden_size)
        self.gender_classifier = nn.Linear(projection_hidden_size, 2)

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
