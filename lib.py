from gensim.utils import simple_preprocess
import re
import nltk
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer

MAX_SEQ_LENGTH = 512

class Manager:

    def __init__(self, model_name):
        self.model_name = model_name
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.max_seq_len = 512

        nltk.download("stopwords")
        self.stop_words = set(nltk.corpus.stopwords.words("english"))

    def seq_preprocess(self, sequence):
        processed_words = simple_preprocess(
            re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(#[A-Za-z0-9]+)|(\w+:\/\/\S+)", " ", sequence))
        processed_words = [word.lower() for word in processed_words if word.lower() not in self.stop_words]

        return ' '.join(processed_words)

    def get_bert_tokens(self, texts):

        texts = [self.seq_preprocess(seq) for seq in texts]
        ct = len(texts)
        input_ids = np.ones((ct, self.max_seq_len), dtype='int32')
        attention_mask = np.zeros((ct, self.max_seq_len), dtype='int32')
        token_type_ids = np.zeros((ct, self.max_seq_len), dtype='int32')

        for k, text in tqdm(enumerate(texts), total=len(texts)):
            # Tokenize
            tok_text = self.tokenizer.tokenize(text)

            # Truncate and convert tokens to numerical IDs
            enc_text = self.tokenizer.convert_tokens_to_ids(tok_text[:(self.max_seq_len - 2)])

            input_length = len(enc_text) + 2
            input_length = input_length if input_length < self.max_seq_len else self.max_seq_len

            # Add tokens [CLS] and [SEP] at the beginning and the end
            input_ids[k, :input_length] = np.asarray([0] + enc_text + [2], dtype='int32')

            # Set to 1s in the attention input
            attention_mask[k, :input_length] = 1

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids
        }
