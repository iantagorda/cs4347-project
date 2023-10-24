import glob
import json

from constants import *
from utils import *

speaker_paths = glob.glob(os.path.join(TRAIN_DATA, "*", "*"))
waveform_paths, lm_labels, accent_labels, gender_labels = get_data_from_speaker_paths(
    speaker_paths
)

# prepare the vocab and dictionary
phoneme_vocab = get_vocab_from_lm_labels(lm_labels)
phoneme_to_id = {phoneme: idx for idx, phoneme in enumerate(phoneme_vocab)}

# save the vocab file to be used by tokenizer
with open("phoneme_to_id", "w") as file:
    json.dump(phoneme_to_id, file)
