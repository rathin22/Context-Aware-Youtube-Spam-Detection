from transformers_with_context import model_with_context
from transformers_without_context import model_without_context
from transformers import BertTokenizer, BertModel, DistilBertTokenizer, DistilBertModel
from sentencetransformers import sentence_transf


device = "cuda"
# Initialize the tokenizer and model
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)

distil_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
distil_model = DistilBertModel.from_pretrained('distilbert-base-uncased').to(device)

print("BERT:\n")
print("WITHOUT CONTEXT:")
model_without_context(bert_model, bert_tokenizer)
print()
print("WITH CONTEXT:")
model_with_context(bert_model, bert_tokenizer)

print("DistilBERT:\n")
print("WITHOUT CONTEXT:")
model_without_context(distil_model, distil_tokenizer)
print()
print("WITH CONTEXT:")
model_with_context(distil_model, distil_tokenizer)

print("all-MiniLM-L12-v2")
print("WITHOUT CONTEXT:")
sentence_transf(includes_context=0)
print()
print("WITH CONTEXT:")
sentence_transf(includes_context=1)