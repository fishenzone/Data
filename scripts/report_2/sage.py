import numpy as np
from augmentex import CharAug, WordAug

class TextAugmentationPipeline:
    def __init__(self, char_aug, word_aug):
        self.char_aug = char_aug
        self.word_aug = word_aug
        
        self.char_actions = char_aug.actions_list
        self.word_actions = word_aug.actions_list
        self.word_actions = ['stopword']

    def augment_text(self, text, augmentations, seed=None):
        np.random.seed(seed)
        aug_type = np.random.choice(list(augmentations.keys()), p=list(augmentations.values()))
        
        if aug_type in ('char', 'both'):
            char_action = np.random.choice(self.char_actions)
            text = self.char_aug.augment(text=text, action=char_action)
            print(char_action)
        
        if aug_type in ('word', 'both'):
            word_action = np.random.choice(self.word_actions)
            text = self.word_aug.augment(text=text, action=word_action)
            print(word_action)

        return text

seed = None

char_aug = CharAug(
    unit_prob=0.3,
    min_aug=1,
    max_aug=5,
    mult_num=3,
    lang="eng",
    platform="pc",
    random_seed=seed,
)

word_aug = WordAug(
    unit_prob=0.4,
    min_aug=1,
    max_aug=5,
    lang="rus",
    platform="pc",
    random_seed=seed,
)


aug_pipeline = TextAugmentationPipeline(char_aug, word_aug)
augmentations = {
    'char': 0.1,  
    'word': 0.8,  
    'both': 0.1   
}

text_eng = "I want to lie down peacefully!"
text_rus = "Я устал очень сильно"

augmented_text_eng = aug_pipeline.augment_text(text_eng, augmentations, seed=seed)
print("Augmented English Text:", augmented_text_eng)

augmented_text_rus = aug_pipeline.augment_text(text_rus, augmentations, seed=seed)
print("Augmented Russian Text:", augmented_text_rus)
