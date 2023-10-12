# Folder names
DATA_ROOT = "train_data"
ANNOTATIONS_FOLDER_NAME = "annotation"
WAV_FOLDER_NAME = "wav"
TRANSCRIPT_FOLDER_NAME = "transcript"
TEXTGRID_FOLDER_NAME = "textgrid"


# Speaker look up
speaker_label_lookup = {
    "BWC": ("Mandarin", "M"),
    "LXC": ("Mandarin", "F"),
    "NCC": ("Mandarin", "F"),
    "TXHC": ("Mandarin", "M"),
    "HJK": ("Korean", "F"),
    "HKK": ("Korean", "M"),
    "YDCK": ("Korean", "F"),
    "YKWK": ("Korean", "M"),
    "HQTV": ("Vietnamese", "M"),
    "PNV": ("Vietnamese", "F"),
    "THV": ("Vietnamese", "F"),
    "TLV": ("Vietnamese", "M"),
}
accents = ["Korean", "Vietnamese", "Mandarin"]
accent_to_id = {accent: idx for idx, accent in enumerate(accents)}
id_to_accent = {idx: accent for idx, accent in enumerate(accents)}

genders = ["M", "F"]
gender_to_id = {gender: idx for idx, gender in enumerate(genders)}
id_to_gender = {idx: gender for idx, gender in enumerate(genders)}
