import numpy as np
from augmentex import CharAug, WordAug
from collections import Counter
import re

class TextAugmentationPipeline:
    def __init__(self, augmenters):
        # Expect 'augmenters' to be a dict with language codes as keys and tuples of (CharAug, WordAug) as values
        self.augmenters = augmenters

    def detect_language(self, text):
        # Simple heuristic based on character frequency to guess the language
        eng_chars = set('abcdefghijklmnopqrstuvwxyz')
        rus_chars = set('абвгдежзийклмнопрстуфхцчшщъыьэюя')
        
        chars = Counter(re.sub(r'\W+', '', text.lower()))
        eng_count = sum(chars[c] for c in eng_chars)
        rus_count = sum(chars[c] for c in rus_chars)
        
        return 'eng' if eng_count > rus_count else 'rus'
    
    def augment_text(self, text, augmentations, seed=None):
        np.random.seed(seed)
        lang = self.detect_language(text)
        
        # Select appropriate augmenters based on detected language
        char_aug, word_aug = self.augmenters.get(lang, (None, None))
        
        if not char_aug or not word_aug:
            raise ValueError(f"No augmenters configured for language '{lang}'")
        
        aug_type = np.random.choice(list(augmentations.keys()), p=list(augmentations.values()))
        
        if aug_type in ('char', 'both'):
            char_action = np.random.choice(char_aug.actions_list)
            text = char_aug.augment(text=text, action=char_action)
            print(f'Char action: {char_action}')
        
        if aug_type in ('word', 'both'):
            word_action = np.random.choice(word_aug.actions_list)
            text = word_aug.augment(text=text, action=word_action)
            print(f'Word action: {word_action}')

        return text

# Example instantiation and usage
seed = None

# Initialize augmenters for both languages
augmenters = {
    'eng': (
        CharAug(unit_prob=0.3, min_aug=1, max_aug=5, mult_num=3, lang="eng", platform="pc", random_seed=seed),
        WordAug(unit_prob=0.4, min_aug=1, max_aug=5, lang="eng", platform="pc", random_seed=seed)
    ),
    'rus': (
        CharAug(unit_prob=0.3, min_aug=1, max_aug=5, mult_num=3, lang="rus", platform="pc", random_seed=seed),
        WordAug(unit_prob=0.4, min_aug=1, max_aug=5, lang="rus", platform="pc", random_seed=seed)
    )
}

aug_pipeline = TextAugmentationPipeline(augmenters)
augmentations = {'char': 0.1, 'word': 0.8, 'both': 0.1}

text_eng = "I want to lie down peacefully!"
text_rus = "Я устал очень сильно"

augmented_text_eng = aug_pipeline.augment_text(text_eng, augmentations, seed=seed)
print("Augmented English Text:", augmented_text_eng)

augmented_text_rus = aug_pipeline.augment_text(text_rus, augmentations, seed=seed)
print("Augmented Russian Text:", augmented_text_rus)
