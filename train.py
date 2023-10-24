import os
import glob

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
)

from constants import *
from utils import *
from models import *


file_prefix = "multitask"
augment = False

# Hyperparameters
learning_rate = 3e-4
num_epochs = 400
batch_size = 8
projection_hidden_size = 256

model_path = "facebook/wav2vec2-xls-r-300m"
backbone_hidden_size = 1024  # this depends on the pre trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading the data...")
# get the data and segregate them
speaker_paths = glob.glob(os.path.join(TRAIN_DATA, "*", "*"))
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


# Create the dataset and split it to train and val
l2_artic_dataset = L2ArcticDataset(
    processor, waveform_paths, lm_labels, accent_labels, gender_labels
)
train_split_generator = torch.Generator().manual_seed(42)
train_dataset, val_dataset = random_split(
    l2_artic_dataset, [0.9, 0.1], generator=train_split_generator
)


# Create the data loaders
train_data_collator = DataCollator(
    processor=processor, padding=True, device=device, augment=augment
)
train_dataloader = DataLoader(
    train_dataset, batch_size=batch_size, collate_fn=train_data_collator, shuffle=True
)

val_data_collator = DataCollator(
    processor=processor, padding=True, device=device, augment=False
)
val_dataloader = DataLoader(
    val_dataset, batch_size=batch_size, collate_fn=val_data_collator, shuffle=False
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


print("Starting the training...")
# Training Loop
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
min_val_per = 9999
for epoch in range(num_epochs):
    print()
    print(f"Epoch {epoch + 1}:")

    # Training
    model.train()
    epoch_total_loss = 0
    epoch_ctc_loss = 0
    epoch_accent_loss = 0
    epoch_gender_loss = 0
    per_list = []
    for waveform, lm_labels, accent_labels, gender_labels in train_dataloader:
        # Forward pass and loss calculation
        ctc_loss, lm_logits, accent_logits, gender_logits = model(waveform, lm_labels)
        accent_loss = criterion(accent_logits, accent_labels)
        gender_loss = criterion(gender_logits, gender_labels)
        total_loss = ctc_loss + accent_loss + gender_loss

        # Optimize parameters
        optimizer.zero_grad()
        total_loss.backward()
        # ctc_loss.backward()
        optimizer.step()

        # Accumulate losses for the epoch (for logging)
        epoch_total_loss += total_loss.item()
        epoch_ctc_loss += ctc_loss.item()
        epoch_accent_loss += accent_loss.item()
        epoch_gender_loss += gender_loss.item()
        per_batch = compute_per(lm_logits, lm_labels, tokenizer)
        per_list.extend(per_batch)

    epoch_per = sum(per_list) / len(per_list)

    print("---Training Losses---")
    print(f"Total Loss:  {epoch_total_loss/len(train_dataloader)}")
    print(f"CTC Loss:    {epoch_ctc_loss/len(train_dataloader)}")
    print(f"Accent Loss: {epoch_accent_loss/len(train_dataloader)}")
    print(f"Gender Loss: {epoch_gender_loss/len(train_dataloader)}")
    print(f"Average PER: {epoch_per}")

    # Validation
    model.eval()
    epoch_total_loss = 0
    epoch_ctc_loss = 0
    epoch_accent_loss = 0
    epoch_gender_loss = 0
    per_list = []
    with torch.no_grad():
        for waveform, lm_labels, accent_labels, gender_labels in val_dataloader:
            # Forward pass and loss calculation
            ctc_loss, lm_logits, accent_logits, gender_logits = model(
                waveform, lm_labels
            )
            accent_loss = criterion(accent_logits, accent_labels)
            gender_loss = criterion(gender_logits, gender_labels)
            total_loss = ctc_loss + accent_loss + gender_loss

            # Accumulate losses for the epoch (for logging)
            epoch_total_loss += total_loss.item()
            epoch_ctc_loss += ctc_loss.item()
            epoch_accent_loss += accent_loss.item()
            epoch_gender_loss += gender_loss.item()
            per_batch = compute_per(lm_logits, lm_labels, tokenizer)
            per_list.extend(per_batch)

    epoch_per = sum(per_list) / len(per_list)

    print("---Validation Losses---")
    print(f"Total Loss:  {epoch_total_loss/len(val_dataloader)}")
    print(f"CTC Loss:    {epoch_ctc_loss/len(val_dataloader)}")
    print(f"Accent Loss: {epoch_accent_loss/len(val_dataloader)}")
    print(f"Gender Loss: {epoch_gender_loss/len(val_dataloader)}")
    print(f"Average PER: {epoch_per}")

    # Save model with the lowest validation ctc loss
    if epoch_per < min_val_per:
        min_val_per = epoch_per
        torch.save(model.state_dict(), f"checkpoints/{file_prefix}_best.pt")

    torch.save(model.state_dict(), f"checkpoints/{file_prefix}_last.pt")

print()
print(f"Minimum PER: {min_val_per}")
print("Done!")
