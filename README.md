### Setup

First, you should install a virtual environment:

```bash
python3 -m venv .venv/bbsard
source .venv/bbsard/bin/activate
```

Then, you can install all dependencies:

```bash
pip install -r requirements.txt
```

## Experiments

### Lexical Models

In order to evaluate the TF-IDF and BM25 models, run:

```bash
python scripts/experiments/run_zeroshot_evaluation.py \
    --articles_path </path/to/articles.csv> \
    --test_questions_path </path/to/questions_test.csv> \
    --retriever {tfidf, bm25} \
    --lang {fr, du} \ 
    --lem \ 
    --output_dir </path/to/output>
```

### Dense Models

#### Zero-Shot Evaluation

```bash
python scripts/experiments/run_zeroshot_evaluation.py \
    --articles_path </path/to/articles.csv> \
    --test_questions_path </path/to/questions_test.csv> \
    --retriever {"tfidf","bm25","word2vec","fasttext","e5-small","e5-base","e5-large",
    "e5-large-instruct","dpr-xm","labse","e5-mistral","mdpr","mcontriever","voyage","openai","jina","gte"} \
    --lang {fr, du} \ 
    --lem \ # [Only for word2vec and fastText] Lemmatize both articles and questions as pre-processing.
    --output_dir </path/to/output> \
    --test_questions_embeddings_path </path/to/test_questions_embeddings> \ # [Only for voyage and openai retriever]
    --articles_embeddings_path </path/to/articles_embeddings> # [Only for voyage and openai retriever]
```
The OpenAI and Voyage retrievers require preprocessed embeddings of articles and questions.
#### List of retrievers and their corresponding models.

| Retriever            | Model                                   |
|----------------------|-----------------------------------------|
| tfidf                | -                                       |
| bm25                 | -                                       |
| word2vec             | -                                       |
| fasttext             | -                                       |
| e5-small             | intfloat/multilingual-e5-small          |
| e5-base              | intfloat/multilingual-e5-base           |
| e5-large             | intfloat/multilingual-e5-large          |
| e5-large-instruct    | intfloat/multilingual-e5-large-instruct |
| e5-mistral           | intfloat/e5-mistral-7b-instruct         |
| dpr-xm               | antoinelouis/dpr-xm                     |
| labse                | sentence-transformers/LaBSE             |
| mdpr                 | castorini/mdpr-tied-pft-msmarco         |
| mcontriever          | facebook/mcontriever-msmarco            |
| jina                 | jinaai/jina-embeddings-v3               |
| gte                  | Alibaba-NLP/gte-multilingual-base       |
| voyage               | -                                       |
| openai               | -                                       |

#### Training

In order to train a bi-encoder model, update the model and training hyperparameters in *scripts/experiments/train_biencoder.py*. Then, run:

```bash
python scripts/experiments/train_biencoder.py
```

To evaluate a trained bi-encoder model, update the checkpoint path in  *scripts/experiments/test_biencoder.py* and run:

```bash
python scripts/experiments/test_biencoder.py
```
