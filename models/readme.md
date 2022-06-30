# Models

The models used in this project can be obtained by using the following scripts/commands:

* `finetune_sbert.py` to **fine-tune SBERT** for matching learning objectives. Uncomment lines in the definition of QueryLinks class to choose from different input types (just learning objective text, plus document titles, plus first sentence of document summaries, or plus _n_ sentences of the document summaries in which at least one term from the anchor appears). Also, please make sure that the data files defined in the def main() are in the data folder. To obtain the results reported in the thesis, repeat this script with random seeds 7,13 and 42 and with all reported combination of input features.

* `train_ltr.py --model_save --features --random_seed` to **train learning-to-rank model** for re-ranking module. `--model_save` is the filepath to save trained model and `--features` are the higher layers to be used as features.`

* `train_classifier.py` to **train classifier** for matching leraning objectives with DistilBERT as cross-encoder for learning objective text (and document titles in the case of the candidates), topic and subject, one-hot encodings with embedding layer for age, and a neural network with softmax to classify learning objective pairs as either a match or a mismatch. 

* To use the **TF-IDF** encoder, a special authorized key is needed. This is because the model was built by the Wizenoze Science Team and is not available for free. If you would like reproduce this experimental step, please contact adrielli.drica@gmail.com.

* To use the **Fasttext** encoder, please download the freely available pre-trained embeddings `wiki-news-300d-1M.vec.zip` on https://fasttext.cc/docs/en/english-vectors.html.

* To use the **pre-trained SBERT** encoder, please install and import the transformers huggingface library, specified on `requirements.txt`.

