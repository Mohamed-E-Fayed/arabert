#
# Note: we should add model.cuda or whatever to run the experiment on GPU instead of CPU
#

from transformers import AutoModel, AutoTokenizer
from nlpaug.augmenter.word import ContextualWordEmbsAug
from nlpaug.augmenter.sentence import ContextualWordEmbsForSentenceAug

from preprocess import ArabertPreprocessor


if __name__=='__main__':
    model_name="aubmindlab/aragpt2-mega"
    #model_name="aubmindlab/bert-large-arabertv02"
    #model_name="asafaya/bert-large-arabic"
    #use_segmentation=True
    use_segmentation=False
    arabert_preprocessor=ArabertPreprocessor(model_name=model_name)
    #tokenizer=AutoTokenizer.from_pretrained(model_name)
    #model=AutoModel.from_pretrained(model_name)
    if 'aragpt2' in model_name.lower():
        augmenter=ContextualWordEmbsForSentenceAug(model_path=model_name,
                device='cpu',
        )

        data=["كتاب", "فاكهة", "سيارة", "كرة القدم", "شائعات", "هذا"]
    else:
        augmenter=ContextualWordEmbsAug(model_path=model_name,
                action='substitute',
                aug_min=1,
                aug_max=3,
                device='cpu',
        )
    
        data=["قرأت كتابين مملين",
                "هذا الكتاب",
                "الكتاب هذا",
                "هي تحب الفاكهة و الخضراوات",
                "لم ألعب الكرة يومًا!",
                "ذهبت إلى الأقصر و أسوان في أجازة",
        ]
    prep_data=arabert_preprocessor.preprocess(data) if use_segmentation else data
    augmented_data=augmenter.augment(prep_data)
    print("data:\n", data)
    print("preprocessed data:\n", prep_data)
    print("augmented data:")
    print(arabert_preprocessor.unpreprocess(augmented_data)) if use_segmentation else print(augmented_data)

