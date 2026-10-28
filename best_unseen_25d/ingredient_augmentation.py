import json
import random
from pathlib import Path


class IngredientAugmentor:
    """Apply paper-style ingredient text augmentation for training."""

    def __init__(
        self,
        enabled=False,
        synonyms_json=None,
        synonym_replace_prob=0.5,
        ingredient_sample_prob=0.5
    ):
        self.enabled = enabled
        self.synonym_replace_prob = float(synonym_replace_prob)
        self.ingredient_sample_prob = float(ingredient_sample_prob)
        self.synonym_map = {}

        if self.enabled and synonyms_json:
            synonyms_path = Path(synonyms_json)
            if synonyms_path.exists():
                with open(synonyms_path, 'r') as f:
                    raw_map = json.load(f)

                # Normalize keys for robust lookup
                for key, values in raw_map.items():
                    if not isinstance(values, list):
                        continue
                    normalized_key = key.strip().lower()
                    normalized_values = []
                    for value in values:
                        if not isinstance(value, str):
                            continue
                        synonym = value.strip()
                        if synonym:
                            normalized_values.append(synonym)
                    if normalized_key and normalized_values:
                        self.synonym_map[normalized_key] = normalized_values

    def _replace_with_synonym(self, ingredient):
        key = ingredient.strip().lower()
        candidates = self.synonym_map.get(key, [])
        if not candidates:
            return ingredient
        return random.choice(candidates)

    def augment(self, ingredient_names):
        """
        1) Per-ingredient synonym replacement with 50% probability.
        2) List-level subset sampling with 50% probability.
        """
        if not self.enabled:
            return ingredient_names

        if not ingredient_names:
            return ingredient_names

        augmented = []
        for ingredient in ingredient_names:
            if random.random() < self.synonym_replace_prob:
                augmented.append(self._replace_with_synonym(ingredient))
            else:
                augmented.append(ingredient)

        # Keep at least one ingredient if sampling is triggered
        if len(augmented) > 1 and random.random() < self.ingredient_sample_prob:
            subset_size = random.randint(1, len(augmented))
            augmented = random.sample(augmented, subset_size)

        return augmented
