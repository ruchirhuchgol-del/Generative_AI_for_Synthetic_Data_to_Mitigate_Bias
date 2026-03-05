"""
Text Augmenter
==============

Data augmentation utilities for text data including:
- Synonym replacement
- Random insertion
- Random deletion
- Random swap
- Back-translation (if API available)
"""

from typing import Any, Dict, List, Optional, Set, Union
import logging
import random

import numpy as np

logger = logging.getLogger(__name__)

# Default synonym dictionary for common words
DEFAULT_SYNONYMS = {
    "good": ["great", "excellent", "fine", "wonderful", "positive"],
    "bad": ["poor", "terrible", "awful", "negative", "unfavorable"],
    "big": ["large", "huge", "enormous", "substantial", "significant"],
    "small": ["little", "tiny", "minor", "compact", "modest"],
    "fast": ["quick", "rapid", "swift", "speedy", "prompt"],
    "slow": ["gradual", "unhurried", "leisurely", "measured"],
    "important": ["significant", "crucial", "essential", "vital", "key"],
    "new": ["novel", "fresh", "recent", "modern", "innovative"],
    "old": ["aged", "previous", "former", "traditional", "established"],
    "different": ["various", "diverse", "distinct", "varied", "unique"],
}


class TextAugmenter:
    """
    Data augmenter for text data.
    
    Provides multiple augmentation strategies:
    - Synonym replacement: Replace words with synonyms
    - Random insertion: Insert random words
    - Random deletion: Delete random words
    - Random swap: Swap adjacent words
    - Back-translation: Translate and back (requires API)
    
    Can be configured to avoid sensitive terms.
    
    Attributes:
        augmentation_ratio: Ratio of augmented to original samples
        synonym_dict: Dictionary of word synonyms
    """
    
    def __init__(
        self,
        augmentation_ratio: float = 0.5,
        synonym_dict: Optional[Dict[str, List[str]]] = None,
        sensitive_terms: Optional[Set[str]] = None,
        synonym_replace_prob: float = 0.1,
        insert_prob: float = 0.1,
        delete_prob: float = 0.1,
        swap_prob: float = 0.1,
        max_augmentations: int = 2,
        preserve_sensitive: bool = True,
        seed: Optional[int] = None
    ):
        """
        Initialize the text augmenter.
        
        Args:
            augmentation_ratio: Ratio of augmented to original samples
            synonym_dict: Dictionary mapping words to synonyms
            sensitive_terms: Terms to avoid augmenting
            synonym_replace_prob: Probability of synonym replacement
            insert_prob: Probability of random insertion
            delete_prob: Probability of random deletion
            swap_prob: Probability of random swap
            max_augmentations: Maximum number of augmentations per sample
            preserve_sensitive: Whether to preserve sensitive terms
            seed: Random seed
        """
        self.augmentation_ratio = augmentation_ratio
        self.synonym_dict = synonym_dict or DEFAULT_SYNONYMS
        self.sensitive_terms = sensitive_terms or set()
        self.synonym_replace_prob = synonym_replace_prob
        self.insert_prob = insert_prob
        self.delete_prob = delete_prob
        self.swap_prob = swap_prob
        self.max_augmentations = max_augmentations
        self.preserve_sensitive = preserve_sensitive
        self.seed = seed
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            
    def augment(
        self,
        texts: Union[str, List[str]],
        method: str = "synonym",
        n_samples: Optional[int] = None
    ) -> List[str]:
        """
        Augment text data.
        
        Args:
            texts: Single text or list of texts
            method: Augmentation method ('synonym', 'insert', 'delete', 'swap', 'combined')
            n_samples: Number of augmented samples (None for ratio-based)
            
        Returns:
            List of augmented texts
        """
        if isinstance(texts, str):
            texts = [texts]
            
        if n_samples is None:
            n_samples = int(len(texts) * self.augmentation_ratio)
            
        augmented = []
        
        for _ in range(n_samples):
            # Sample a random text
            text = random.choice(texts)
            
            # Apply augmentation
            if method == "synonym":
                aug_text = self._synonym_replace(text)
            elif method == "insert":
                aug_text = self._random_insert(text)
            elif method == "delete":
                aug_text = self._random_delete(text)
            elif method == "swap":
                aug_text = self._random_swap(text)
            elif method == "combined":
                aug_text = self._combined_augment(text)
            else:
                raise ValueError(f"Unknown augmentation method: {method}")
                
            augmented.append(aug_text)
            
        return augmented
    
    def _synonym_replace(self, text: str) -> str:
        """Replace words with synonyms."""
        words = text.split()
        new_words = words.copy()
        
        for i, word in enumerate(words):
            # Skip sensitive terms
            if self.preserve_sensitive and word.lower() in self.sensitive_terms:
                continue
                
            # Check for synonym
            word_lower = word.lower()
            if word_lower in self.synonym_dict:
                if random.random() < self.synonym_replace_prob:
                    synonym = random.choice(self.synonym_dict[word_lower])
                    # Preserve capitalization
                    if word[0].isupper():
                        synonym = synonym.capitalize()
                    new_words[i] = synonym
                    
        return " ".join(new_words)
    
    def _random_insert(self, text: str) -> str:
        """Insert random words from text."""
        words = text.split()
        
        if not words:
            return text
            
        n_insertions = max(1, int(len(words) * self.insert_prob))
        
        for _ in range(n_insertions):
            # Pick random word from text
            insert_word = random.choice(words)
            
            # Skip sensitive terms
            if self.preserve_sensitive and insert_word.lower() in self.sensitive_terms:
                continue
                
            # Insert at random position
            insert_pos = random.randint(0, len(words))
            words.insert(insert_pos, insert_word)
            
        return " ".join(words)
    
    def _random_delete(self, text: str) -> str:
        """Delete random words."""
        words = text.split()
        
        if len(words) <= 3:
            return text
            
        new_words = []
        
        for word in words:
            # Skip sensitive terms
            if self.preserve_sensitive and word.lower() in self.sensitive_terms:
                new_words.append(word)
                continue
                
            # Keep word with probability
            if random.random() > self.delete_prob:
                new_words.append(word)
                
        # Ensure at least one word
        if not new_words:
            new_words = [random.choice(words)]
            
        return " ".join(new_words)
    
    def _random_swap(self, text: str) -> str:
        """Swap adjacent words."""
        words = text.split()
        
        if len(words) < 2:
            return text
            
        n_swaps = max(1, int(len(words) * self.swap_prob))
        
        for _ in range(n_swaps):
            # Pick random adjacent pair
            idx = random.randint(0, len(words) - 2)
            
            # Skip if sensitive
            if self.preserve_sensitive:
                if words[idx].lower() in self.sensitive_terms or \
                   words[idx + 1].lower() in self.sensitive_terms:
                    continue
                    
            # Swap
            words[idx], words[idx + 1] = words[idx + 1], words[idx]
            
        return " ".join(words)
    
    def _combined_augment(self, text: str) -> str:
        """Apply multiple augmentations."""
        # Apply up to max_augmentations methods
        methods = [
            self._synonym_replace,
            self._random_insert,
            self._random_delete,
            self._random_swap
        ]
        
        selected = random.sample(
            methods, 
            min(self.max_augmentations, len(methods))
        )
        
        result = text
        for method in selected:
            result = method(result)
            
        return result
    
    def back_translate(
        self,
        text: str,
        src_lang: str = "en",
        pivot_lang: str = "fr"
    ) -> str:
        """
        Back-translate text through a pivot language.
        
        Note: Requires translation API to be configured.
        
        Args:
            text: Input text
            src_lang: Source language
            pivot_lang: Pivot language for translation
            
        Returns:
            Back-translated text
        """
        try:
            # Try using googletrans if available
            from googletrans import Translator
            
            translator = Translator()
            
            # Translate to pivot language
            translated = translator.translate(text, src=src_lang, dest=pivot_lang)
            
            # Translate back
            back_translated = translator.translate(
                translated.text, src=pivot_lang, dest=src_lang
            )
            
            return back_translated.text
            
        except ImportError:
            logger.warning("googletrans not installed. Back-translation disabled.")
            return text
        except Exception as e:
            logger.warning(f"Back-translation failed: {e}")
            return text
    
    def augment_batch(
        self,
        texts: List[str],
        labels: Optional[List[Any]] = None,
        method: str = "synonym",
        n_samples: Optional[int] = None
    ) -> Union[List[str], Tuple[List[str], List[Any]]]:
        """
        Augment a batch of texts, optionally with labels.
        
        Args:
            texts: List of texts
            labels: Optional labels
            method: Augmentation method
            n_samples: Number of samples
            
        Returns:
            Augmented texts (and labels if provided)
        """
        if n_samples is None:
            n_samples = int(len(texts) * self.augmentation_ratio)
            
        augmented_texts = self.augment(texts, method, n_samples)
        
        if labels is not None:
            # For text augmentation, labels stay the same
            augmented_labels = random.choices(labels, k=n_samples)
            return augmented_texts, augmented_labels
            
        return augmented_texts
    
    def add_sensitive_terms(self, terms: List[str]) -> None:
        """Add terms to the sensitive terms set."""
        self.sensitive_terms.update(t.lower() for t in terms)
        
    def add_synonyms(self, word: str, synonyms: List[str]) -> None:
        """Add synonyms for a word."""
        if word.lower() not in self.synonym_dict:
            self.synonym_dict[word.lower()] = []
        self.synonym_dict[word.lower()].extend(synonyms)


class EasyDataAugmentation:
    """
    Easy Data Augmentation (EDA) implementation.
    
    Implements the EDA techniques from:
    "EDA: Easy Data Augmentation Techniques for Boosting Performance on Text Classification Tasks"
    (Wei and Zou, 2019)
    """
    
    def __init__(
        self,
        alpha: float = 0.1,
        n_aug: int = 4,
        synonym_dict: Optional[Dict[str, List[str]]] = None
    ):
        """
        Initialize EDA augmenter.
        
        Args:
            alpha: Percentage of words to modify
            n_aug: Number of augmented sentences per original
            synonym_dict: Dictionary of synonyms
        """
        self.alpha = alpha
        self.n_aug = n_aug
        
        self.augmenter = TextAugmenter(
            synonym_dict=synonym_dict,
            synonym_replace_prob=alpha,
            insert_prob=alpha,
            delete_prob=alpha,
            swap_prob=alpha
        )
        
    def augment(self, text: str) -> List[str]:
        """
        Generate n_aug augmented versions of text.
        
        Args:
            text: Input text
            
        Returns:
            List of augmented texts
        """
        augmented = []
        
        # One of each method
        augmented.append(self.augmenter._synonym_replace(text))
        augmented.append(self.augmenter._random_insert(text))
        augmented.append(self.augmenter._random_delete(text))
        augmented.append(self.augmenter._random_swap(text))
        
        return augmented[:self.n_aug]
