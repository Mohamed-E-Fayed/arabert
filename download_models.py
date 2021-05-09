from transformers import AutoTokenizer, AutoModel
from arabert.preprocess import ArabertPreprocessor

# Mini:   asafaya/bert-mini-arabic
# Medium: asafaya/bert-medium-arabic
# Base:   asafaya/bert-base-arabic
# Large:  asafaya/bert-large-arabic
model="aubmindlab/bert-base-arabertv2"
tokenizer = AutoTokenizer.from_pretrained(model)
model = AutoModel.from_pretrained(model)

tokenizer.save_pretrained('bert-base-arabertv2')
model.save_pretrained('bert-base-arabertv2')


#tokenizer_c = AutoTokenizer.from_pretrained("bert-base-chinese")
#model_c = AutoModel.from_pretrained("bert-base-chinese")
#
#tokenizer_c.save_pretrained('bert-base-chinese')
#model_c.save_pretrained('bert-base-chinese')
#
