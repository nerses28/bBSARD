import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import pairwise_distances
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from typing import List

def average_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

class E5Retriever:
    def __init__(self, model_path_or_name: str, retrieval_corpus: List[str], batch_size: int = 16):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_path_or_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()
        self.batch_size = batch_size
        self.retrieval_corpus = [f"passage: {doc}" for doc in retrieval_corpus]
        self.corpus_embeddings = self.encode(self.retrieval_corpus, batch_size=self.batch_size)
        print()

    def encode(self, texts: List[str], batch_size: int = 16) -> np.ndarray:
        embeddings = []
        total_batches = (len(texts) + batch_size - 1) // batch_size
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            with torch.no_grad():
                batch_dict = self.tokenizer(batch_texts, max_length=512, padding=True, truncation=True, return_tensors="pt").to(self.device)
                outputs = self.model(**batch_dict)
                batch_embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
                batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1).cpu().numpy()
                embeddings.append(batch_embeddings)
            current_batch = i // batch_size + 1
            print(f"batch {current_batch}/{total_batches}", end='\r')
        return np.vstack(embeddings)

    def search_all(self, queries: List[str], top_k: int = 500) -> List[List[int]]:
        results = []
        for query in queries:
            query_text = f"query: {query}"
            query_embedding = self.encode([query_text])[0]
            scores = np.dot(self.corpus_embeddings, query_embedding)
            top_k_indices = np.argsort(scores)[-top_k:][::-1]
            results.append(top_k_indices.tolist())
        print()
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

class E5LARGERetriever:
    def __init__(self, model_path_or_name: str = "intfloat/e5-mistral-7b-instruct", retrieval_corpus: List[str] = None,
                 batch_size: int = 16, max_seq_length: int = 4096):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(model_path_or_name).to(self.device)
        self.model.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.retrieval_corpus = retrieval_corpus
        self.corpus_embeddings = self.encode(self.retrieval_corpus, batch_size=self.batch_size)

    def encode(self, texts: List[str], batch_size: int = 16, prompt_name: str = None) -> np.ndarray:
        print('Start encode')
        embeddings = self.model.encode(texts, batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=True,
                                       prompt_name=prompt_name)
        return embeddings

    def search_all(self, queries: List[str], top_k: int = 500) -> List[List[int]]:
        query_embeddings = self.encode(queries, batch_size=self.batch_size, prompt_name="web_search_query")

        results = []
        for query_embedding in query_embeddings:
            scores = np.dot(self.corpus_embeddings, query_embedding)
            top_k_indices = np.argsort(scores)[-top_k:][::-1]
            results.append(top_k_indices.tolist())
        return results



class E5Chunked2Retriever:
    def __init__(self, model_path_or_name: str, retrieval_corpus: List[str], batch_size: int = 16,
                 chunk_size: int = 500, overlap: int = 50):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_path_or_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.retrieval_corpus = retrieval_corpus
        self.corpus_embeddings = self.encode(self.retrieval_corpus, batch_size=self.batch_size)
        print()

    def _split_inputs(self, inputs, pad_value):
        chunks = [inputs[i:i + self.chunk_size] for i in range(0, len(inputs), self.chunk_size - self.overlap)]
        if len(chunks) > 1 and len(chunks[-1]) <= self.overlap:
            chunks = chunks[:-1]
        chunks[-1] += [pad_value] * (self.chunk_size - len(chunks[-1]))
        return torch.tensor(chunks, device=self.device)

    def encode(self, texts: List[str], batch_size: int = 16) -> np.ndarray:
        embeddings = []
        total_batches = (len(texts) + batch_size - 1) // batch_size
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = []
            with torch.no_grad():
                for text in batch_texts:
                    tokenized = self.tokenizer(text, truncation=False)
                    input_ids = self._split_inputs(tokenized['input_ids'], pad_value=self.tokenizer.pad_token_id)
                    attention_mask = self._split_inputs(tokenized['attention_mask'], pad_value=0)

                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    chunk_embeddings = average_pool(outputs.last_hidden_state, attention_mask)
                    print(chunk_embeddings.shape)
                    pooled_embedding = F.normalize(chunk_embeddings.mean(dim=0, keepdim=True), p=2, dim=1)
                    batch_embeddings.append(pooled_embedding.cpu().numpy())

                embeddings.extend(batch_embeddings)
            current_batch = i // batch_size + 1
            print(f"batch {current_batch}/{total_batches}", end='\r')
        return np.vstack(embeddings)

    def search_all(self, queries: List[str], top_k: int = 500) -> List[List[int]]:
        results = []
        for query in queries:
            query_embedding = self.encode([query])[0]
            scores = np.dot(self.corpus_embeddings, query_embedding)
            top_k_indices = np.argsort(scores)[-top_k:][::-1]
            results.append(top_k_indices.tolist())
        print()
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