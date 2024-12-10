import os
import json
import argparse
from os.path import abspath, join

import pandas as pd

from utils.eval import Evaluator
from utils.data import TextPreprocessor
from models.lexical_models import TFIDFRetriever, BM25Retriever
from models.zeroshot_dense_models import Word2vecRetriever, FasttextRetriever, BERTRetriever, E5ChunkedRetriever, E5InstructChunkedRetriever, EmbRetriever, LaBSERetriever, JinaRetriever, GTERetriever, DPRRetriever, FlagModel, RetLongCtxtFP16



def add_instructions(model_name: str, query: str):
	task_description = 'Given a question, retrieve passages that answer the question.'
	if model_name == "bge-gemma2":
		return f'<instruct>{task_description}\n<query>{query}'
	elif model_name in ["e5-large-instruct", "e5-mistral"]:
		return f'Instruct: {task_description}\nQuery: {query}'


def main(args):
	print("Loading questions and articles...")
	dfA = pd.read_csv(args.articles_path)
	dfQ_test = pd.read_csv(args.test_questions_path)
	ground_truths = dfQ_test['article_ids'].apply(lambda x: list(map(int, x.split(',')))).tolist()

	if args.retriever in ['tfidf', 'bm25', 'word2vec', 'fasttext']:
		print("Preprocessing articles and questions (lemmatizing={})...".format(args.lem))
		spacy_model = "nl_core_news_sm"
		if args.lang == 'fr':
			spacy_model = "fr_core_news_md"
		cleaner = TextPreprocessor(spacy_model=spacy_model)
		articles = cleaner.preprocess(dfA['article'], lemmatize=args.lem)
		questions = cleaner.preprocess(dfQ_test['question'], lemmatize=args.lem)
	else:
		articles = dfA['article'].tolist()
		questions = dfQ_test['question'].tolist()

	print("Initializing the {} retriever model...".format(args.retriever))
	if args.retriever == 'tfidf':
		retriever = TFIDFRetriever(retrieval_corpus=articles)
	elif args.retriever == 'bm25':
		retriever = BM25Retriever(retrieval_corpus=articles, k1=1.0, b=0.6)
	elif args.retriever == 'word2vec':
		if args.lang == 'fr':
			best_checkpoint = abspath(join(__file__, "../embeddings/word2vec/lemmatized/word2vec_frWac_lem_skipgram_d500.bin"))
		else:
			best_checkpoint = abspath(join(__file__, "../embeddings/word2vec/cow-big.txt"))
		retriever = Word2vecRetriever(model_path_or_name=best_checkpoint, pooling_strategy='mean', retrieval_corpus=articles)
	elif args.retriever == 'fasttext':
		if args.lang == 'fr':
			best_checkpoint = abspath(join(__file__, "../embeddings/fasttext/fasttext_frCc_cbow_d300.bin"))
		else:
			best_checkpoint = abspath(join(__file__, "../embeddings/fasttext/cc.nl.300.bin"))
		retriever = FasttextRetriever(model_path_or_name=best_checkpoint, pooling_strategy='mean', retrieval_corpus=articles)
	elif args.retriever == 'bert':
		if args.lang == 'fr':
			retriever = BERTRetriever(model_path_or_name='camembert-base', pooling_strategy='mean', retrieval_corpus=articles)
		else:
			retriever = BERTRetriever(model_path_or_name='DTAI-KULeuven/robbert-2022-dutch-base', pooling_strategy='mean', retrieval_corpus=articles)
	elif args.retriever.startswith('e5'):
		model_name = {'e5-small': 'intfloat/multilingual-e5-small', 'e5-base': 'intfloat/multilingual-e5-base',
					  "e5-large": 'intfloat/multilingual-e5-large', "e5-mistral": "intfloat/e5-mistral-7b-instruct",
					  "e5-large-instruct": "intfloat/multilingual-e5-large-instruct"}[args.retriever]
		if args.retriever == "e5-mistral":
			retriever = RetLongCtxtFP16(model_path_or_name=model_name, pooling_strategy='mean', retrieval_corpus=articles)
		else:
			retriever = E5ChunkedRetriever(model_path_or_name=model_name, pooling_strategy='mean', retrieval_corpus=articles)
	elif args.retriever == 'dpr-xm':
		language_code = 'fr_XX' if args.lang == 'fr' else 'nl_XX'
		retriever = DPRRetriever(model_path_or_name='antoinelouis/dpr-xm', retrieval_corpus=articles, language_code=language_code)
	elif args.retriever == 'mdpr':
		retriever = E5ChunkedRetriever(model_path_or_name='castorini/mdpr-tied-pft-msmarco', pooling_strategy='mean', retrieval_corpus=articles)
	elif args.retriever == 'labse':
		retriever = LaBSERetriever(model_path_or_name='sentence-transformers/LaBSE', retrieval_corpus=articles)
	elif args.retriever == 'mcontriever':
		retriever = E5ChunkedRetriever(model_path_or_name='facebook/mcontriever-msmarco', pooling_strategy='mean', retrieval_corpus=articles)
	elif args.retriever == 'jina':
		retriever = JinaRetriever(model_path_or_name='jinaai/jina-embeddings-v3', retrieval_corpus=articles)
	elif args.retriever == 'gte':
		retriever = GTERetriever(model_path_or_name='Alibaba-NLP/gte-multilingual-base', retrieval_corpus=articles)
	elif args.retriever == 'bge-gemma2':
		retriever = RetLongCtxtFP16(model_path_or_name='BAAI/bge-multilingual-gemma2', retrieval_corpus=articles)
	elif args.retriever == 'bge-m3':
		retriever = FlagModel(model_path_or_name='BAAI/bge-m3', retrieval_corpus=articles)
	elif args.retriever in ['voyage', 'openai']:
		with open(args.articles_embeddings_path, 'r') as fl:
			emb_art = json.load(fl)
		with open(args.test_questions_embeddings_path, 'r') as fl:
			emb_q = json.load(fl)
		retriever = EmbRetriever(query_embeddings=emb_q, doc_embeddings=emb_art)


	# adding prompts for instruction models
	if args.retriever in ['bge-gemma2', 'e5-mistral', 'e5-large-instruct']:
		questions = list(map(lambda x: add_instructions(query=x), questions))


	print("Running model on test questions...")
	if args.retriever in ['tfidf', 'bm25', 'labse', 'dpr-xm']:
		retrieved_docs = retriever.search_all(questions, top_k=500)
	else:
		retrieved_docs = retriever.search_all(questions, top_k=500, dist_metric='cosine')

	print("Computing the retrieval scores...")
	evaluator = Evaluator()
	scores = evaluator.compute_all_metrics(retrieved_docs, ground_truths)
	
	print("Saving the scores to {} ...".format(args.output_dir))
	os.makedirs(args.output_dir, exist_ok=True)
	with open(join(args.output_dir, f'{args.retriever}_test_results.json'), 'w') as f:
		json.dump(scores, f, indent=2)

	if args.retriever not in ["tfidf", "bm25", 'labse', 'dpr-xm']:
		retrieved_docs = retrieved_docs.tolist()
	with open(join(args.output_dir, f'{args.retriever}_test_retrieved_docs.json'), 'w') as f:
		json.dump(retrieved_docs, f, indent=2)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--articles_path", 
						type=str, 
						default=abspath(join(__file__, "../../../data/bbsard/nl/corpus.csv")),  # for French : /nl/ --> /fr/
						help="Path of the data file containing the law articles."
	)
	parser.add_argument("--test_questions_path", 
						type=str, 
						default=abspath(join(__file__, "../../../data/bbsard/nl/test.csv")),    # for French : /nl/ --> /fr/
						help="Path of the data file containing the test questions."
	)
	parser.add_argument("--lem",
						action='store_true',
						default=False,
						help="Lemmatize the questions and articles for retrieval."
	)
	parser.add_argument("--lang",
						type=str,
						help="Model/data Language",
						choices= ['fr', 'nl']
	)
	parser.add_argument("--retriever", 
						type=str,
						choices=["tfidf","bm25","word2vec","fasttext","bert","e5-small","e5-base","e5-large",
								 "e5-large-instruct","dpr-xm","labse","e5-mistral","mdpr","mcontriever", "voyage",
								 "openai", "jina", "gte", "bge-gemma2", "bge-m3"],
						required=True,
						help="The type of model to use for retrieval"
	)
	parser.add_argument("--output_dir",
						type=str, 
						default=abspath(join(__file__, "../output/zeroshot/test-run/")),
						help="Path of the output directory."
	)
	parser.add_argument("--articles_embeddings_path",
						type=str,
						help="Path to the file containing the articles embeddings (for embedding models like OpenAI and Voyage)"
						)
	parser.add_argument("--test_questions_embeddings_path",
						type=str,
						help="Path to the file containing the question embeddings (for embedding models like OpenAI and Voyage)"
						)
	args, _ = parser.parse_known_args()
	main(args)
