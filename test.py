import os
import glob

import torch
from torch.utils.data import DataLoader
from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
)

from constants import *
from utils import *
from models import *


batch_size = 64
checkpoint_paths = glob.glob(os.path.join("checkpoints", "*"))


projection_hidden_size = 256
model_path = "facebook/wav2vec2-xls-r-300m"
backbone_hidden_size = 1024  # this depends on the pre trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


print("Loading the data...")
# get the data and segregate them
speaker_paths = glob.glob(os.path.join(TEST_DATA, "*", "*"))
waveform_paths, lm_labels, accent_labels, gender_labels = get_data_from_speaker_paths(
    speaker_paths
)

print("Transforming the data...")
# create tokenizer, feature extractor and processor
tokenizer = Wav2Vec2CTCTokenizer(
    "phoneme_to_id", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|"
)
feature_extractor = Wav2Vec2FeatureExtractor(
    feature_size=1,
    sampling_rate=16000,
    padding_value=0.0,
    do_normalize=True,
    return_attention_mask=False,
)
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)


# Create the dataset for testing
test_dataset = L2ArcticDataset(
    processor, waveform_paths, lm_labels, accent_labels, gender_labels
)


# Create the data loaders
data_collator = DataCollator(
    processor=processor, padding=True, device=device, augment=False
)
test_dataloader = DataLoader(
    test_dataset, batch_size=batch_size, collate_fn=data_collator, shuffle=False
)

print("Creating the model...")
# creat the backbone and the model
wav2vec2_backbone = Wav2Vec2ForCTC.from_pretrained(
    pretrained_model_name_or_path=model_path,
    ignore_mismatched_sizes=True,
    ctc_loss_reduction="mean",
    pad_token_id=processor.tokenizer.pad_token_id,
    vocab_size=len(processor.tokenizer),
    output_hidden_states=True,
)
wav2vec2_backbone = wav2vec2_backbone.to(device)
model = MultiTaskWav2Vec2(
    wav2vec2_backbone=wav2vec2_backbone,
    backbone_hidden_size=backbone_hidden_size,
    projection_hidden_size=projection_hidden_size,
    num_accent_class=len(accents),
)

for checkpoint_path in checkpoint_paths:
    print(f"Loading checkpoint: {checkpoint_path}")

    model.load_state_dict(torch.load(checkpoint_path))
    model.to(device)
    model.eval()

    per_list = []
    with torch.no_grad():
        for waveform, lm_labels, accent_labels, gender_labels in test_dataloader:
            _, lm_logits, accent_logits, gender_logits = model(waveform, lm_labels)
            per = compute_per(lm_logits, lm_labels, tokenizer)
            per_list.append(per)
    print("Average PER: ", sum(per_list) / len(per_list))
    print()
