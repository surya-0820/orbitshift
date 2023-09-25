from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_repo = ""
tokenizer = AutoTokenizer.from_pretrained(model_repo)
model = AutoModelForSeq2SeqLM.from_pretrained(model_repo)


model.save_pretrained('./mymodel')
tokenizer.save_pretrained('./mymodel')