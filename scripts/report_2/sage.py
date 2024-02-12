import numpy as np

import sage
from sage.spelling_corruption import CharAugConfig, CharAugCorruptor, WordAugConfig, WordAugCorruptor

class TextAugmentationPipeline:
    def __init__(self, char_aug_config, word_aug_config):
        self.char_corruptor = CharAugCorruptor.from_config(char_aug_config)
        self.word_corruptor = WordAugCorruptor.from_config(word_aug_config)

    def _apply_augmentation(self, text, corruptor, seed, action=None):
        """
        Apply the specified augmentation action using the given corruptor.
        If no action is specified, a random action is chosen.
        """
        if action:
            return corruptor.corrupt(text, action=action, seed=seed)
        return corruptor.corrupt(text, seed=seed)

    def augment_text(self, text, augmentations, seed=0):
        """
        Augment the text based on specified augmentations and their probabilities.

        :param text: The input text to augment.
        :param augmentations: A dictionary specifying augmentation types ('char', 'word', 'both') 
        and their probabilities.
        :return: Augmented text.
        """
        # Choose augmentation type based on probabilities
        aug_type = np.random.choice(list(augmentations.keys()), p=list(augmentations.values()))

        char_actions = ['shift', 'orfo', 'typo', 'delete', 'multiply', 'swap', 'insert']
        word_actions = ['replace', 'delete', 'swap', 'stopword', 'reverse']
        char_probs = [0.14] * 7
        word_probs = [0.2] * 5
        
        if aug_type in ('char', 'both'):
            char_action = np.random.choice(char_actions, p=char_probs)
            text = self._apply_augmentation(text, self.char_corruptor, seed, char_action)
            
        if aug_type in ('word', 'both'):
            word_action = np.random.choice(word_actions, p=word_probs)
            text = self._apply_augmentation(text, self.word_corruptor, seed, word_action)
        return text

# Configuration for character-level augmentations
char_aug_config = CharAugConfig(unit_prob=0.3, min_aug=1, max_aug=5, mult_num=3)

# Configuration for word-level augmentations
word_aug_config = WordAugConfig(unit_prob=0.4, min_aug=1, max_aug=5)

# Initialize the augmentation pipeline
aug_pipeline = TextAugmentationPipeline(char_aug_config, word_aug_config)

# Define augmentation probabilities
augmentations = {
    'char': 0.3,  # Probability to apply only character-level augmentations
    'word': 0.3,  # Probability to apply only word-level augmentations
    'both': 0.4   # Probability to apply both character and word-level augmentations
}

# Example usage
text_eng = "I want to lie down peacefully!"
text_rus = "Я устал очень сильно"

# Augment English text
augmented_text_eng = aug_pipeline.augment_text(text_eng, augmentations)
print("Augmented English Text:", augmented_text_eng)

# Augment Russian text
augmented_text_rus = aug_pipeline.augment_text(text_rus, augmentations)
print("Augmented Russian Text:", augmented_text_rus)
