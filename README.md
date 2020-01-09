# Sarcasm-Detection-using-LSTM-Models
Using Long Short-Term Memory(LSTM) models to find out sarcastic contents in texts.

# Datasets
A subset of Prinston's SARC 2.0 corpus for sarcasm detection with size of 100,000 samples. (Reference: nlp.cs.princeton.edu/SARC/2.0/)
With the project's focus on the influence of conversational context, only 'label', 'comment' and 'parent_comment' columns are saved ('author' column is also saved but actually not used).
 - train.csv: training set with size of 80,000 samples.
 - val.csv: validation set with size of 10,000 samples.
 - test.csv: test set with size of 10,000 samples.
 
# Implementation
Text Data Preprocessing: TorchText(0.4.0)
Pre-trained word vector: GloVe.6B.300d
Model Implementation: PyTorch(1.3.1)
Models are with/without conversational context and with/without attention mechanism. In total 4 LSTM models.

# Running the program
1. Download all .csv and .py files
2. Run training.py -- change parameters/models in training.py and load_data.py
3. See results in the console

# Evaluation
Precisions:
1. LSTM: 69.39%
2. LSTM(context): 65.4%
3. LSTM(attention): 69.73%
4. LSTM(attention+context): 66.49%

# Future Work
- Try to separate comments and parent_comments into different networks (using Seq2Seq models)
