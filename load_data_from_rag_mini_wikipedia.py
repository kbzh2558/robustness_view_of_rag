from datasets import load_dataset

ds = load_dataset("rag-datasets/rag-mini-wikipedia", "question-answer")
dt = load_dataset("rag-datasets/rag-mini-wikipedia", "text-corpus")

ds = ds["test"].to_pandas()
dt = dt['passages'].to_pandas()

dt.to_csv('rag-mini-wikipedia_document.csv', index=False)
ds.to_csv('rag-mini-wikipedia_q_and_a.csv', index=False)