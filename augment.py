from transformers import AutoModel, AutoTokenizer
from nlpaug.augmenter.word import ContextualWordEmbsAug

from arabert.preprocess import ArabertPreprocessor


if __name__=='__main__':
    model_name="aubmindlab/bert-base-arabertv2"
    arabert_preprocessor=ArabertPreprocessor(model_name=model_name)
    tokenizer=AutoTokenizer.from_pretrained(model_name)
    model=AutoModel.from_pretrained(model_name)
    augmenter=ContextualWordEmbsAug(model_path=model_name,
            action='substitute',
            aug_min=1,
            aug_max=10,
            device='cpu',
    )
    
    data="لقد قرأت كتابين مملين"
    augmented_data=augmenter.augment(data)
    print("data:\n", data)
    print("augmented data:\n", augmented_data)

