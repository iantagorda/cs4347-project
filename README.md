# cs4347-project

## Hugging Face Demo
The most intuitive was to test the model is to try the [HuggingFace Demo](https://huggingface.co/spaces/peproject/pronounciationevaluation).

## Data Preparation
Training and testing path must first be defined in the `constants.py` file. Inside the training/testing directory, there should be a subdirectory for each speaker L1/origin country. Inside each L1 subdirectory is a folder for annotations, wav, transcript and textgrid (these subfolders are taken directly from L2 Arctic Dataset). A sample of the directory structure for both train and test for one Korean speaker is provided in the repo. 

## Training
Before training, the vocabulary to be used must be generated first. this can be done by running `create_vocab.py`. After creating the vocab, training the model is as simple as running `train.py`. The compute resource to run this is quite heavy. When running the script on Slurm, `run.sh` can be used as a basis.

To choose whether or not to run with augmentations, set the `augment` variable accordingly. It is also possible to change the feature extractor used by swapping the backbone. The resulting checkpoints of the training are stored in the checkpoints folder.

## Testing
To evaluate the model on the testing data, simply run `test.py`. `run_test.sh` is a useful basis when running the script on Slurm.

## Notebook files
There are a couple of notebooks files that were used in the data preparation and exploration. 
- `Data Augmentation of L2-ARCTIC.ipynb` was used to explore the different augmentation techniques that were used.
- `Train_Test_Split_of_L2_Artic_Korean,_Vietnamese_and_Mandarin_Speakers.ipynb` was used to separate the dataset into train and test data.
- `Wav2Vec2_Baseline.ipynb` was used to create the baseline model. This also includes design decisions made in the data processing stage.
- `l2 artic dataset split with TTS samples.ipynb` was used to prototype the TTS used in HuggingFace.
