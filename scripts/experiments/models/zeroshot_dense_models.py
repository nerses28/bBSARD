import sys
import logging

from tqdm import tqdm

import numpy as np
from sklearn.metrics import pairwise_distances

import torch
import torch.nn.functional as F
import fasttext
import transformers
from gensim.models import KeyedVectors
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer

from utils.common import log_step
from typing import List

transformers.logging.set_verbosity_error()
fasttext.FastText.eprint = lambda x: None
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

class DenseRetriever:
    def __init__(self, model, vocab, dimension, pooling_strategy, retrieval_corpus):
        self.model = model
        self.vocab = vocab
        self.dimension = dimension
        self.pooling_strategy = pooling_strategy
        self.doc_embeddings = self._embed_corpus(retrieval_corpus, 'articles')

    def __repr__(self):
        return f"{self.__class__.__name__}".lower()

    @log_step
    def search_all(self, queries, top_k, dist_metric):
        query_embeddings = self._embed_corpus(queries, 'queries')
        print("Computing scores...")
        scores = pairwise_distances(X=list(query_embeddings.values()), 
                                    Y=list(self.doc_embeddings.values()),
                                    metric=dist_metric)
        results = np.argsort(scores, axis=1)[:, :top_k] + 1  #+1 because doc ID starts at 1.
        print("Done.")
        return results

    def get_doc_embeddings(self):
        return self.doc_embeddings

    def get_vocab(self):
        return self.vocab

    def _embed_corpus(self, corpus, desc):
        embeddings = dict()
        for i, doc in enumerate(tqdm(corpus, desc='Embedding ' + desc)):
            embeddings[i+1] = self._embed(doc)
        return embeddings

    def _embed(self, text):
        word_embeddings = self._get_word_embeddings(text)
        if not word_embeddings:
            return np.zeros(self.dimension)
        return self._perform_pooling(word_embeddings)

    def _perform_pooling(self, embeddings):
        if self.pooling_strategy == 'mean':
            return np.mean(embeddings, axis=0)
        elif self.pooling_strategy == 'max':
            return np.max(embeddings, axis=0)
        elif self.pooling_strategy == 'sum':
            return np.sum(embeddings, axis=0)
        else:
            sys.exit("Pooling strategy not known, please use 'mean' or 'max'.")

    def _get_word_embeddings(self, text):
        return [self.model[w] for w in text.split()]


class Word2vecRetriever(DenseRetriever):
    def __init__(self, model_path_or_name, pooling_strategy, retrieval_corpus):
        assert pooling_strategy in ['mean', 'max', 'sum'], f'Unknown pooling strategy: {pooling_strategy}'
        #model = KeyedVectors.load_word2vec_format(model_path_or_name, binary=True, unicode_errors="ignore")
        model = KeyedVectors.load_word2vec_format(model_path_or_name, unicode_errors="ignore")
        super().__init__(model=model,
                         #vocab=set(model.vocab.keys()),
                         vocab=set(model.key_to_index.keys()),
                         dimension=model.vector_size,
                         pooling_strategy=pooling_strategy,
                         retrieval_corpus=retrieval_corpus)

    def _get_word_embeddings(self, text):
        return [self.model[w] for w in text.split() if w in self.vocab]


class FasttextRetriever(DenseRetriever):
    def __init__(self, model_path_or_name, pooling_strategy, retrieval_corpus):
        model = fasttext.load_model(model_path_or_name)
        super().__init__(model=model,
                         vocab=set(model.words),
                         dimension=model.get_dimension(),
                         pooling_strategy=pooling_strategy,
                         retrieval_corpus=retrieval_corpus)


class BERTRetriever(DenseRetriever):
    def __init__(self, model_path_or_name, pooling_strategy, retrieval_corpus):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path_or_name)
        self.model = AutoModel.from_pretrained(model_path_or_name, output_hidden_states=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        super().__init__(model=self.model,
                         vocab=self.tokenizer.get_vocab(),
                         dimension=self.model.config.hidden_size,
                         pooling_strategy=pooling_strategy,
                         retrieval_corpus=retrieval_corpus)

    def _embed(self, text, chunk_size=500, overlap=50, pooling_layer=-2):
        """
        Remarks:
        - Why not using the hidden state of [CLS]? Because a pre-trained model is not fine-tuned on any downstream
        tasks yet, the hidden state of [CLS] is not a good sentence representation.
        - Why not averaging the embeddings from the last hidden layer? The last layer is too closed to the target 
        functions (i.e. masked language model and next sentence prediction) during pre-training, therefore may 
        be biased to those targets.
        - Which hidden layer should be use then? As a rule of thumb, use second-to-last layer. Intuitively, the last 
        hidden layer is close to the training output, so it may be biased to the training targets and lead to a bad         
        representation if you don't fine-tune the model later on. On the other hand, the first hidden layer is close 
        to the word embedding and will probably preserve the very original word information (very little self-attention 
        involved). That said, anything in between ([1, 11]) is then a trade-off.
        - Source: https://bert-as-service.readthedocs.io/en/latest/section/faq.html
        """
        # Tokenize input text.
        tokenized = self.tokenizer(text, truncation=False)
        input_ids = self._split_inputs(tokenized['input_ids'], chunk_size, overlap, pad_value=self.tokenizer.pad_token_id)
        attention_masks = self._split_inputs(tokenized['attention_mask'], chunk_size, overlap, pad_value=0)

        # Move features to device.
        input_ids = input_ids.to(self.device)
        attention_masks = attention_masks.to(self.device)

        # Pass features to model.
        self.model.eval()
        with torch.no_grad():
            output = self.model(input_ids, attention_masks).hidden_states #tuple of 'num_layers' 3D tensors of shape [num_splits, chunk_size, D]
            output = torch.stack(output, dim=0) #4D tensor of shape [num_layers, num_splits, chunk_size, D]

        # Extract the embeddings from given layer.
        token_embeddings = output[pooling_layer] #3D tensor of shape [num_splits, chunk_size, D]

        # Pool embeddings to get a document representation.
        doc_embedding = self._pool(token_embeddings, attention_masks)
        return doc_embedding.detach().cpu().numpy()

    def _split_inputs(self, inputs, chunk_size, overlap, pad_value):
        inputs = [inputs[i:i+chunk_size] for i in range(0, len(inputs), chunk_size-overlap)] #split tokenized sequence into chunks of 'chunk_size' with an 'overlap'.
        inputs = inputs[:-1] if len(inputs[-1]) <= overlap and len(inputs) > 1 else inputs #make sure that the last chunk is not simply the repeated overlap of the previous chunk.
        inputs[-1] += [pad_value] * (chunk_size - len(inputs[-1])) #pad the last chunk to the defined 'chunk_size' (All model inputs will therefore have exactly the same length).
        return torch.tensor(inputs)

    def _pool(self, token_embeddings, attention_masks):
        # Flatten tensors.
        token_embeddings = token_embeddings.reshape(1, -1, token_embeddings.shape[2]).squeeze() # [num_splits, chunk_size, D] -> [num_splits x chunk_size,  D]
        attention_masks = attention_masks.flatten() # [num_splits, chunk_size] -> [num_splits x chunk_size]

        if self.pooling_strategy == 'max':
            # Set all values of the [PAD] embeddings to large negative values (so that they are never considered as maximum for a channel).
            attention_masks_expanded = attention_masks.unsqueeze(-1).expand(token_embeddings.size()).float()
            token_embeddings[attention_masks_expanded == 0] = -1e9

            # Compute the maxima along the sequence length dimension -> Tensor[D].
            return torch.max(token_embeddings, dim=0).values
        else:
            # Set all values of the [PAD] embeddings to zeros (so that they are not taken into account in the sum for a channel).
            attention_masks_expanded = attention_masks.unsqueeze(-1).expand(token_embeddings.size()).float()
            token_embeddings[attention_masks_expanded == 0] = 0.0

            # First sum along the 'seq_length' dimension -> Tensor[embedding_dim]).
            sum_embedding = torch.sum(token_embeddings, dim=0)
            if self.pooling_strategy == 'sum':
                return sum_embedding

            # Then, divide all values of the passage vector by the original passage length.
            sum_mask = attention_masks_expanded.sum(dim=0) # -> Tensor[batch_size, embedding_dim] where each value is the length of the corresponding passage.
            sum_mask = torch.clamp(sum_mask, min=1e-9) # Make sure not to have zeros by lower bounding all elements to 1e-9.
            mean_embedding = sum_embedding / sum_mask # Divide each dimension by the sequence length.
            return mean_embedding

class E5ChunkedRetriever(BERTRetriever):
    def _embed(self, text, chunk_size=200, overlap=20, pooling_layer=-1):
        # Tokenize input text.

        tokenized = self.tokenizer(text, truncation=False)
        input_ids = self._split_inputs(tokenized['input_ids'], chunk_size, overlap, pad_value=self.tokenizer.pad_token_id)
        attention_masks = self._split_inputs(tokenized['attention_mask'], chunk_size, overlap, pad_value=0)
        # Move features to device.
        input_ids = input_ids.to(self.device)
        attention_masks = attention_masks.to(self.device)

        # Pass features to model.
        self.model.eval()
        with torch.no_grad():
            output = self.model(input_ids, attention_masks).hidden_states #tuple of 'num_layers' 3D tensors of shape [num_splits, chunk_size, D]
            output = torch.stack(output, dim=0) #4D tensor of shape [num_layers, num_splits, chunk_size, D]

        # Extract the embeddings from given layer.
        token_embeddings = output[pooling_layer] #3D tensor of shape [num_splits, chunk_size, D]
        # Pool embeddings to get a document representation.
        doc_embedding = self._pool(token_embeddings, attention_masks)
        return doc_embedding.detach().cpu().numpy()

class E5InstructChunkedRetriever(E5ChunkedRetriever):
    def search_all(self, queries, top_k, dist_metric):
        task = 'Given a web search query, retrieve relevant passages that answer the query.\n'
        queries = [task + query for query in queries]
        query_embeddings = self._embed_corpus(queries, 'queries')
        print("Computing scores...")
        scores = pairwise_distances(X=list(query_embeddings.values()),
                                    Y=list(self.doc_embeddings.values()),
                                    metric=dist_metric)
        results = np.argsort(scores, axis=1)[:, :top_k] + 1  #+1 because doc ID starts at 1.
        print("Done.")
        return results

class EmbRetriever:
    def __init__(self, query_embeddings, doc_embeddings):
        self.query_embeddings = query_embeddings
        self.doc_embeddings = doc_embeddings

    def search_all(self, queries, top_k, dist_metric):
        print("Computing scores...")
        scores = pairwise_distances(X=self.query_embeddings,
                                    Y=self.doc_embeddings,
                                    metric=dist_metric)
        results = np.argsort(scores, axis=1)[:, :top_k] + 1  #+1 because doc ID starts at 1.
        print("Done.")
        return results

class DPRRetriever:
    def __init__(self, model_path_or_name: str, retrieval_corpus: List[str], language_code: str = "fr_FR"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(model_path_or_name).to(self.device)
        self.model[0].auto_model.set_default_language(language_code)
        self.retrieval_corpus = retrieval_corpus
        self.corpus_embeddings = self.encode(self.retrieval_corpus)
        print()

    def encode(self, texts: List[str]) -> np.ndarray:
        print('Start encode for articles')
        art_embeddings = []
        for i, text in enumerate(texts):
            print(f'Encoding art {i + 1}/{len(texts)}', end='\r')
            embedding = self.model.encode([text], convert_to_numpy=True, normalize_embeddings=True)
            art_embeddings.append(embedding[0])
        return np.array(art_embeddings)

    def encode_queries(self, texts: List[str]) -> np.ndarray:
        print('Start encode for queries')
        query_embeddings = []
        for i, text in enumerate(texts):
            print(f'Encoding query {i + 1}/{len(texts)}', end='\r')
            embedding = self.model.encode([text], convert_to_numpy=True, normalize_embeddings=True)
            query_embeddings.append(embedding[0])
        return np.array(query_embeddings)

    def search_all(self, queries: List[str], top_k: int = 500) -> List[List[int]]:
        results = []
        query_embeddings = self.encode_queries(queries)

        for query_embedding in query_embeddings:
            scores = np.dot(self.corpus_embeddings, query_embedding)
            top_k_indices = np.argsort(scores)[-top_k:][::-1] + 1
            results.append(top_k_indices.tolist())
        return results

class LaBSERetriever:
    def __init__(self, model_path_or_name: str = "sentence-transformers/LaBSE", retrieval_corpus: List[str] = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(model_path_or_name).to(self.device)
        self.retrieval_corpus = retrieval_corpus
        self.corpus_embeddings = self.encode(self.retrieval_corpus)

    def encode(self, texts: List[str]) -> np.ndarray:
        print('Start encode for articles')
        art_embeddings = []
        for i, text in enumerate(texts):
            print(f'Encoding art {i+1}/{len(texts)}', end='\r')
            embedding = self.model.encode([text], convert_to_numpy=True, normalize_embeddings=True)
            art_embeddings.append(embedding[0])
        return np.array(art_embeddings)

    def encode_queries(self, texts: List[str]) -> np.ndarray:
        print('Start encode for queries')
        query_embeddings = []
        for i, text in enumerate(texts):
            print(f'Encoding query {i+1}/{len(texts)}', end='\r')
            embedding = self.model.encode([text], convert_to_numpy=True, normalize_embeddings=True)
            query_embeddings.append(embedding[0])
        return np.array(query_embeddings)

    def search_all(self, queries: List[str], top_k: int = 500) -> List[List[int]]:
        results = []
        query_embeddings = self.encode_queries(queries)

        for query_embedding in query_embeddings:
            scores = np.dot(self.corpus_embeddings, query_embedding)
            top_k_indices = np.argsort(scores)[-top_k:][::-1] + 1
            results.append(top_k_indices.tolist())
        return results

class JinaRetriever:
    def __init__(self, model_path_or_name: str, retrieval_corpus: List[str]):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_path_or_name
        self.model = AutoModel.from_pretrained(self.model_name, trust_remote_code=True).to(self.device)
        self.model.eval()
        self.corpus_embeddings = self.encode(retrieval_corpus, 'arts')

    def encode(self, texts, task_name):
        embs = []
        task = {'query': 'retrieval.query', "arts": 'retrieval.passage'}[task_name]
        i = 0
        for text in texts:
            e = self.model.encode([text], task=task, max_length=8192)
            embs.append(e[0])

            i += 1
            print('text: ', i, '/', len(texts), end='\r')
        print()
        return embs

    def search_all(self, queries, top_k, dist_metric):
        print("Computing scores...")
        query_embeddings = self.encode(queries, 'query')
        scores = pairwise_distances(X=query_embeddings,
                                    Y=self.corpus_embeddings,
                                    metric=dist_metric)
        results = np.argsort(scores, axis=1)[:, :top_k] + 1  #+1 because doc ID starts at 1.
        print("Done.")
        return results

class GTERetriever:
    def __init__(self, model_path_or_name: str, retrieval_corpus: List[str]):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_path_or_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name, trust_remote_code=True).to(self.device)
        self.model.eval()
        self.corpus_embeddings = self.encode(retrieval_corpus)


    def encode(self, texts):
        embs = []
        i = 0
        with torch.no_grad():
            for text in texts:
                inputs = self.tokenizer([text], max_length=8192, padding=True, truncation=True, return_tensors='pt').to(self.device)
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state[:, 0][:768]
                embeddings = F.normalize(embeddings, p=2, dim=1)
                embs.extend(embeddings.cpu())
                i += 1
                print('text: ', i, '/', len(texts), end='\r')
        print()
        return embs

    def search_all(self, queries, top_k, dist_metric):
        print("Computing scores...")
        query_embeddings = self.encode(queries)
        scores = pairwise_distances(X=query_embeddings,
                                    Y=self.corpus_embeddings,
                                    metric=dist_metric)
        results = np.argsort(scores, axis=1)[:, :top_k] + 1  #+1 because doc ID starts at 1.
        print("Done.")
        return results