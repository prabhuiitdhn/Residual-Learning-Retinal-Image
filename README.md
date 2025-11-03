This project is about understanding the residual between the original retinal image and noisy residual image. Using Encoder-decoder architecture based.
to run this project
Install the required packages

pip install -r requirements.txt

- run data_preprocess.py to augment the images and add random noise.
- model.py is for model architecture designing
- train.py is for training the model on pair images.
- trainer.py is for hitting the model training with saving the intermediate features on each epochs
- test.py is for testing.

