"""
Text Preprocessor
=================

Preprocessing utilities for text data including:
- Text cleaning and normalization
- Tokenization (word, character, BPE)
- Stopword removal
- Lemmatization and stemming
- Sensitive term detection and anonymization
"""

from typing import Any, Callable, Dict, List, Optional, Set, Union
import logging
import re
import string
from pathlib import Path
import json

import numpy as np

logger = logging.getLogger(__name__)

# Default patterns
DEFAULT_PUNCTUATION = string.punctuation
DEFAULT_STOPWORDS_EN = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "was", "are", "were", "been",
    "be", "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "must", "shall", "can", "need", "dare", "ought",
    "used", "it", "its", "this", "that", "these", "those", "i", "you", "he",
    "she", "we", "they", "what", "which", "who", "whom", "when", "where",
    "why", "how", "all", "each", "every", "both", "few", "more", "most",
    "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so",
    "than", "too", "very", "just", "also"
}

# Sensitive term patterns for fairness
SENSITIVE_PATTERNS = {
    "gender": {
        "male_terms": ["he", "him", "his", "man", "men", "boy", "boys", "father", "son", "brother", "husband", "uncle", "nephew", "mr", "sir", "gentleman"],
        "female_terms": ["she", "her", "hers", "woman", "women", "girl", "girls", "mother", "daughter", "sister", "wife", "aunt", "niece", "ms", "mrs", "madam", "lady"],
        "stereotypical_male": ["aggressive", "ambitious", "analytical", "assertive", "confident", "decisive", "dominant", "independent", "logical", "strong"],
        "stereotypical_female": ["nurturing", "caring", "emotional", "sensitive", "compassionate", "gentle", "supportive", "collaborative", "empathetic", "warm"]
    },
    "race": {
        "indicators": ["race", "ethnicity", "origin", "ethnic", "racial"]
    }
}


class TextPreprocessor:
    """
    Comprehensive preprocessor for text data.
    
    Handles:
    - Text cleaning (lowercase, punctuation, whitespace)
    - Tokenization with multiple strategies
    - Stopword removal
    - Lemmatization and stemming
    - Sensitive term detection and anonymization
    - Custom preprocessing pipelines
    
    Attributes:
        lowercase: Whether to convert to lowercase
        remove_punctuation: Whether to remove punctuation
        remove_stopwords: Whether to remove stopwords
        lemmatize: Whether to apply lemmatization
        stem: Whether to apply stemming
        tokenize_method: Tokenization method
    """
    
    def __init__(
        self,
        lowercase: bool = True,
        remove_punctuation: bool = False,
        remove_stopwords: bool = False,
        remove_numbers: bool = False,
        lemmatize: bool = False,
        stem: bool = False,
        tokenize_method: str = "word",
        max_length: Optional[int] = None,
        min_length: int = 1,
        stopwords: Optional[Set[str]] = None,
        custom_patterns: Optional[Dict[str, str]] = None,
        anonymize_sensitive: bool = False,
        sensitive_categories: Optional[List[str]] = None
    ):
        """
        Initialize the text preprocessor.
        
        Args:
            lowercase: Convert text to lowercase
            remove_punctuation: Remove punctuation characters
            remove_stopwords: Remove common stopwords
            remove_numbers: Remove numeric characters
            lemmatize: Apply lemmatization (requires NLTK)
            stem: Apply stemming (requires NLTK)
            tokenize_method: Tokenization method ('word', 'char', 'sentence')
            max_length: Maximum sequence length
            min_length: Minimum sequence length
            stopwords: Custom stopwords set
            custom_patterns: Custom regex patterns {name: pattern}
            anonymize_sensitive: Whether to anonymize sensitive terms
            sensitive_categories: Categories to anonymize
        """
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_stopwords = remove_stopwords
        self.remove_numbers = remove_numbers
        self.lemmatize = lemmatize
        self.stem = stem
        self.tokenize_method = tokenize_method
        self.max_length = max_length
        self.min_length = min_length
        self.anonymize_sensitive = anonymize_sensitive
        self.sensitive_categories = sensitive_categories or ["gender", "race"]
        
        self.stopwords = stopwords if stopwords is not None else DEFAULT_STOPWORDS_EN
        self.custom_patterns = custom_patterns or {}
        
        # Initialize NLP tools if needed
        self._lemmatizer = None
        self._stemmer = None
        self._tokenizer = None
        
        if lemmatize:
            self._init_lemmatizer()
        if stem:
            self._init_stemmer()
            
        # Build sensitive term dictionary
        self._sensitive_terms = self._build_sensitive_terms()
        
        self._is_fitted = False
        self._statistics: Dict[str, Any] = {}
        
    def _init_lemmatizer(self) -> None:
        """Initialize lemmatizer."""
        try:
            from nltk.stem import WordNetLemmatizer
            self._lemmatizer = WordNetLemmatizer()
        except ImportError:
            logger.warning("NLTK not installed. Lemmatization disabled.")
            self.lemmatize = False
            
    def _init_stemmer(self) -> None:
        """Initialize stemmer."""
        try:
            from nltk.stem import PorterStemmer
            self._stemmer = PorterStemmer()
        except ImportError:
            logger.warning("NLTK not installed. Stemming disabled.")
            self.stem = False
    
    def _build_sensitive_terms(self) -> Dict[str, Set[str]]:
        """Build dictionary of sensitive terms by category."""
        terms = {}
        for category in self.sensitive_categories:
            if category in SENSITIVE_PATTERNS:
                category_terms = set()
                for term_list in SENSITIVE_PATTERNS[category].values():
                    category_terms.update(term.lower() for term in term_list)
                terms[category] = category_terms
        return terms
    
    def fit(self, texts: List[str]) -> "TextPreprocessor":
        """
        Fit the preprocessor on training texts.
        
        Args:
            texts: List of training texts
            
        Returns:
            Fitted preprocessor
        """
        # Compute statistics
        self._statistics = {
            "n_texts": len(texts),
            "avg_length": np.mean([len(t.split()) for t in texts]),
            "max_length": max(len(t.split()) for t in texts) if texts else 0,
            "min_length": min(len(t.split()) for t in texts) if texts else 0,
            "vocabulary_size": len(set(
                word for text in texts for word in text.lower().split()
            ))
        }
        
        self._is_fitted = True
        logger.info(f"TextPreprocessor fitted on {len(texts)} texts")
        
        return self
    
    def transform(self, texts: Union[str, List[str]]) -> Union[str, List[str]]:
        """
        Transform texts using fitted preprocessor.
        
        Args:
            texts: Single text or list of texts
            
        Returns:
            Preprocessed text(s)
        """
        if isinstance(texts, str):
            return self._preprocess_text(texts)
        return [self._preprocess_text(text) for text in texts]
    
    def fit_transform(self, texts: List[str]) -> List[str]:
        """
        Fit and transform in one step.
        
        Args:
            texts: List of training texts
            
        Returns:
            Preprocessed texts
        """
        return self.fit(texts).transform(texts)
    
    def _preprocess_text(self, text: str) -> str:
        """
        Apply all preprocessing steps to a single text.
        
        Args:
            text: Input text
            
        Returns:
            Preprocessed text
        """
        # Apply custom patterns first
        for pattern, replacement in self.custom_patterns.items():
            text = re.sub(pattern, replacement, text)
        
        # Lowercase
        if self.lowercase:
            text = text.lower()
        
        # Remove numbers
        if self.remove_numbers:
            text = re.sub(r'\d+', '', text)
        
        # Remove punctuation
        if self.remove_punctuation:
            text = text.translate(str.maketrans('', '', DEFAULT_PUNCTUATION))
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenize
        tokens = self._tokenize(text)
        
        # Remove stopwords
        if self.remove_stopwords:
            tokens = [t for t in tokens if t.lower() not in self.stopwords]
        
        # Lemmatize
        if self.lemmatize and self._lemmatizer:
            tokens = [self._lemmatizer.lemmatize(t) for t in tokens]
        
        # Stem
        if self.stem and self._stemmer:
            tokens = [self._stemmer.stem(t) for t in tokens]
        
        # Filter by length
        tokens = [t for t in tokens if len(t) >= self.min_length]
        
        # Truncate
        if self.max_length is not None:
            tokens = tokens[:self.max_length]
        
        # Anonymize sensitive terms
        if self.anonymize_sensitive:
            tokens = self._anonymize_tokens(tokens)
        
        return ' '.join(tokens)
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text using specified method.
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        if self.tokenize_method == "word":
            return text.split()
        elif self.tokenize_method == "char":
            return list(text.replace(' ', '_'))
        elif self.tokenize_method == "sentence":
            return re.split(r'[.!?]+', text)
        else:
            return text.split()
    
    def _anonymize_tokens(self, tokens: List[str]) -> List[str]:
        """
        Anonymize sensitive terms in tokens.
        
        Args:
            tokens: List of tokens
            
        Returns:
            Anonymized tokens
        """
        result = []
        for token in tokens:
            replaced = False
            for category, terms in self._sensitive_terms.items():
                if token.lower() in terms:
                    result.append(f"<{category.upper()}>")
                    replaced = True
                    break
            if not replaced:
                result.append(token)
        return result
    
    def detect_sensitive_terms(self, text: str) -> Dict[str, List[str]]:
        """
        Detect sensitive terms in text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary mapping categories to found terms
        """
        if self.lowercase:
            text = text.lower()
            
        detected = {}
        tokens = set(text.split())
        
        for category, terms in self._sensitive_terms.items():
            found = tokens.intersection(terms)
            if found:
                detected[category] = list(found)
                
        return detected
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get computed statistics."""
        return self._statistics.copy()
    
    def save(self, path: Union[str, Path]) -> None:
        """Save preprocessor configuration."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        config = {
            "lowercase": self.lowercase,
            "remove_punctuation": self.remove_punctuation,
            "remove_stopwords": self.remove_stopwords,
            "remove_numbers": self.remove_numbers,
            "lemmatize": self.lemmatize,
            "stem": self.stem,
            "tokenize_method": self.tokenize_method,
            "max_length": self.max_length,
            "min_length": self.min_length,
            "anonymize_sensitive": self.anonymize_sensitive,
            "sensitive_categories": self.sensitive_categories,
            "statistics": self._statistics
        }
        
        with open(path, "w") as f:
            json.dump(config, f, indent=2)
            
        logger.info(f"TextPreprocessor saved to {path}")
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> "TextPreprocessor":
        """Load preprocessor configuration."""
        with open(path) as f:
            config = json.load(f)
            
        preprocessor = cls(
            lowercase=config["lowercase"],
            remove_punctuation=config["remove_punctuation"],
            remove_stopwords=config["remove_stopwords"],
            remove_numbers=config["remove_numbers"],
            lemmatize=config["lemmatize"],
            stem=config["stem"],
            tokenize_method=config["tokenize_method"],
            max_length=config["max_length"],
            min_length=config["min_length"],
            anonymize_sensitive=config["anonymize_sensitive"],
            sensitive_categories=config["sensitive_categories"]
        )
        preprocessor._statistics = config.get("statistics", {})
        preprocessor._is_fitted = True
        
        return preprocessor


class TextCleaner:
    """
    Simple text cleaner for basic preprocessing tasks.
    
    Provides static methods for common cleaning operations.
    """
    
    @staticmethod
    def remove_urls(text: str) -> str:
        """Remove URLs from text."""
        return re.sub(r'http\S+|www\S+', '', text)
    
    @staticmethod
    def remove_emails(text: str) -> str:
        """Remove email addresses from text."""
        return re.sub(r'\S+@\S+', '', text)
    
    @staticmethod
    def remove_mentions(text: str) -> str:
        """Remove @mentions from text."""
        return re.sub(r'@\w+', '', text)
    
    @staticmethod
    def remove_hashtags(text: str) -> str:
        """Remove hashtags from text."""
        return re.sub(r'#\w+', '', text)
    
    @staticmethod
    def remove_html_tags(text: str) -> str:
        """Remove HTML tags from text."""
        return re.sub(r'<[^>]+>', '', text)
    
    @staticmethod
    def remove_emojis(text: str) -> str:
        """Remove emojis from text."""
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"
            "\U0001F300-\U0001F5FF"
            "\U0001F680-\U0001F6FF"
            "\U0001F1E0-\U0001F1FF"
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "]+",
            flags=re.UNICODE
        )
        return emoji_pattern.sub('', text)
    
    @staticmethod
    def normalize_unicode(text: str) -> str:
        """Normalize unicode characters."""
        import unicodedata
        return unicodedata.normalize('NFKC', text)
    
    @staticmethod
    def expand_contractions(text: str) -> str:
        """Expand common contractions."""
        contractions = {
            "won't": "will not",
            "can't": "cannot",
            "n't": " not",
            "'re": " are",
            "'s": " is",
            "'d": " would",
            "'ll": " will",
            "'ve": " have",
            "'m": " am"
        }
        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)
        return text
