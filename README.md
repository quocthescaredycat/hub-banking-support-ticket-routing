# Hub Banking — Intent Routing and Semantic Search

A BANKING77 research project progressing toward a support-agent workbench: classify an English request into one of 77 intents, suggest a support queue, retrieve similar requests, and record human review.

**Current status:** EDA, TF-IDF/MiniLM baselines, and explicit PyTorch transformer training are implemented. The Colab notebook contains a completed DistilBERT run. RoBERTa/LoRA experiments and the application are the next phase of work.

**Research stack:** Python, PyTorch, Hugging Face, scikit-learn, Sentence Transformers; PEFT is planned for the LoRA comparison.

**Planned application stack:** React/TypeScript, FastAPI, PostgreSQL for ticket/review records, Qdrant for vector retrieval, and Docker Compose for a local demo. Public hosting has not been selected.

- [Current specification, architecture, and delivery gates](SPEC.md)
- [Earlier research plan and references](PLAN.md)
- [Current in-progress CV entry](CV_PROJECT.tex)
- [EDA and light preprocessing](preprocess.ipynb)
- [PyTorch fine-tuning and test evaluation](training_transformer.ipynb)
- [Model architecture classes](model_lib.py)

The native development split uses 9,002 training and 1,001 validation requests; the official 3,080-example test set remains separate. Recorded notebook results and locally verified artifacts are distinguished in the specification. This project uses public benchmark data and proposed application queues.
