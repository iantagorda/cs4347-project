import textgrid
import os
from constants import *
import jiwer
import torch


def get_phones_annotation_data(textgrid_filepath):
    """
    Ignores the word and IPA annotations -- with word annotations, it is possible to add word delimiters
    Ignores the start and end time annotations
    Outputs a dictionary containing the list of correct phonemes (how the text should have been pronounced natively),
    pronouned_phones (how the text was pronouned by L2 speaker) and a list of errors (substition, addition or deletion) corresponding
    to index of phones
    """

    tg_dict = {"correct_phones": [], "pronounced_phones": [], "errors": []}
    try:
        tg = textgrid.TextGrid.fromFile(textgrid_filepath)
    except:
        print("annotation Error loading textgrid file")
        return
    for interval_tier in tg:
        if interval_tier.name != "phones":
            continue
        results = []
        for interval in interval_tier:
            start_time = interval.minTime
            end_time = interval.maxTime
            annotation = interval.mark
            if annotation == "":
                annotation = "sil"
            if "," in annotation:
                # either s,a or d pronounciation error by L2 speaker
                correct_phoneme, detected_phoneme, error_type = annotation.split(",")
                # we do not add the artificial sil token for L2 speaker in the case of phone deletion
                if error_type.strip() != "d":
                    tg_dict["pronounced_phones"].append(detected_phoneme.strip())
                tg_dict["correct_phones"].append(correct_phoneme.strip())
                tg_dict["errors"].append(error_type.strip())
            else:
                # l2 speaker pronounced correctly -- so annotation is just a phone e.g. "AH"
                tg_dict["correct_phones"].append(annotation)
                tg_dict["pronounced_phones"].append(annotation)
                tg_dict["errors"].append("")

    return tg_dict


def get_phonetic_transcription_data(textgrid_filepath):
    tg_dict = {"pronounced_phones": []}
    try:
        tg = textgrid.TextGrid.fromFile(textgrid_filepath)
    except:
        print("transcription Error loading textgrid file")
        return

    for interval_tier in tg:
        if interval_tier.name != "phones":
            continue
        results = []
        for interval in interval_tier:
            start_time = interval.minTime
            end_time = interval.maxTime
            annotation = interval.mark
            if annotation == "":
                annotation = "sil"

            tg_dict["pronounced_phones"].append(annotation)

    return tg_dict


def get_data_from_speaker_paths(speaker_paths):
    # currently not returned
    text_transcription = []

    waveform_paths = []
    lm_labels = []
    accent_labels = []
    gender_labels = []

    for speaker_path in speaker_paths:
        speaker = os.path.basename(speaker_path)

        accent_str, gender_str = speaker_label_lookup[speaker]
        accent_label = accent_to_id[accent_str]
        gender_label = gender_to_id[gender_str]

        train_path = os.path.join(speaker_path, "")
        filenames = [
            os.path.splitext(filename)[0]
            for filename in os.listdir(f"{speaker_path}/wav")
        ]

        wav_path = os.path.join(train_path, WAV_FOLDER_NAME)
        transcript_path = os.path.join(train_path, TRANSCRIPT_FOLDER_NAME)
        annotation_path = os.path.join(train_path, ANNOTATIONS_FOLDER_NAME)
        textgrid_path = os.path.join(train_path, TEXTGRID_FOLDER_NAME)

        for filename in filenames:
            wav_filename = os.path.join(wav_path, f"{filename}.wav")
            transcript_filename = os.path.join(transcript_path, f"{filename}.txt")
            annotation_filename = os.path.join(annotation_path, f"{filename}.TextGrid")
            textgrid_filename = os.path.join(textgrid_path, f"{filename}.TextGrid")

            with open(transcript_filename) as f:
                contents = f.read()
            f.close()

            phone_annotation = None
            if os.path.exists(annotation_filename):  # manual annotations are rare
                phone_annotation = get_phones_annotation_data(annotation_filename)
            # we use force alignment transcriptions if there are no manual annotations

            if phone_annotation == None:
                phone_annotation = get_phonetic_transcription_data(textgrid_filename)

            if phone_annotation == None:
                continue
            pronounced_phones = phone_annotation["pronounced_phones"]

            text_transcription.append(contents)

            waveform_paths.append(wav_filename)
            lm_labels.append("|".join(pronounced_phones))
            accent_labels.append(accent_label)
            gender_labels.append(gender_label)

    return waveform_paths, lm_labels, accent_labels, gender_labels


def get_vocab_from_lm_labels(lm_labels):
    phoneme_vocab = set()
    for lm_label in lm_labels:
        for phoneme in lm_label.split("|"):
            phoneme_vocab.add(phoneme)
    phoneme_vocab = ["|", "[UNK]", "[PAD]"] + list(phoneme_vocab)

    return phoneme_vocab


def compute_per(lm_preds, lm_labels, tokenizer):
    lm_preds = torch.argmax(lm_preds, dim=-1)
    lm_labels[lm_labels == -100] = tokenizer.pad_token_id
    per_list = []
    for i in range(len(lm_labels)):
        pred_decoded = [phoneme for phoneme in tokenizer.batch_decode(lm_preds[i])]
        label_decoded = [
            phoneme
            for phoneme in tokenizer.batch_decode(lm_labels[i], group_tokens=False)
        ]

        pred_str = " ".join(pred_decoded)
        label_str = " ".join(label_decoded)

        per = jiwer.wer(pred_str, label_str)
        per_list.append(per)

    return per_list
