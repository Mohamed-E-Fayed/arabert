from transformers import AutoTokenizer, AutoModel

# Mini:   asafaya/bert-mini-arabic
# Medium: asafaya/bert-medium-arabic
# Base:   asafaya/bert-base-arabic
# Large:  asafaya/bert-large-arabic
#models=["aubmindlab/bert-base-arabertv2", "aubmindlab/bert-large-arabertv2"]
models=["aubmindlab/aragpt2-mega"]
for model_name in models:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    tokenizer.save_pretrained(model_name)
    model.save_pretrained(model_name)


#tokenizer_c = AutoTokenizer.from_pretrained("bert-base-chinese")
#model_c = AutoModel.from_pretrained("bert-base-chinese")
#
#tokenizer_c.save_pretrained('bert-base-chinese')
#model_c.save_pretrained('bert-base-chinese')
#
