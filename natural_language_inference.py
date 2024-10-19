from transformers import pipeline

nli_pipeline = pipeline(model="llm-book/bert-base-japanese-v3-jnli")
text = "二人の男性がジェット機を見ています"
entailment_text = "ジェット機を見ている人が二人います"

# textとentailment_textの論理関係を予測
print(nli_pipeline({"text": text, "text_pair": entailment_text}))