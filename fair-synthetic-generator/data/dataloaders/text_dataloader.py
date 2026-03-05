"""
Text Dataloader
===============

Data loader for text data with tokenization and preprocessing.
Supports various tokenization strategies and fairness-aware text processing.
"""

from typing import Any, Dict, List, Optional, Union, Callable
from pathlib import Path
import logging
import json
import re

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from .base_dataloader import BaseDataset, BaseDataLoader, collate_fn_variable_length

logger = logging.getLogger(__name__)


class Tokenizer:
    """
    Simple tokenizer supporting multiple strategies.
    
    Supports:
    - Word-level tokenization
    - Character-level tokenization  
    - BPE-style tokenization (simplified)
    """
    
    SPECIAL_TOKENS = {
        "pad": "<PAD>",
        "unk": "<UNK>",
        "bos": "<BOS>",
        "eos": "<EOS>",
        "mask": "<MASK>"
    }
    
    def __init__(
        self,
        vocab_size: int = 30000,
        tokenization: str = "word",
        lowercase: bool = True,
        min_frequency: int = 2,
        special_tokens: Optional[Dict[str, str]] = None
    ):
        """
        Initialize the tokenizer.
        
        Args:
            vocab_size: Maximum vocabulary size
            tokenization: Tokenization strategy
            lowercase: Whether to lowercase text
            min_frequency: Minimum frequency for token inclusion
            special_tokens: Custom special tokens
        """
        self.vocab_size = vocab_size
        self.tokenization = tokenization
        self.lowercase = lowercase
        self.min_frequency = min_frequency
        
        self.special_tokens = {**self.SPECIAL_TOKENS, **(special_tokens or {})}
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self._fitted = False
        
        # Initialize special tokens
        for i, (name, token) in enumerate(self.special_tokens.items()):
            self.token_to_id[token] = i
            self.id_to_token[i] = token
            
    def fit(self, texts: List[str]) -> "Tokenizer":
        """
        Build vocabulary from texts.
        
        Args:
            texts: List of text samples
            
        Returns:
            Self for chaining
        """
        # Count tokens
        token_counts: Dict[str, int] = {}
        
        for text in texts:
            tokens = self._tokenize(text)
            for token in tokens:
                token_counts[token] = token_counts.get(token, 0) + 1
                
        # Sort by frequency and take top
        sorted_tokens = sorted(
            token_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Add tokens to vocabulary
        next_id = len(self.special_tokens)
        for token, count in sorted_tokens:
            if count >= self.min_frequency and next_id < self.vocab_size:
                self.token_to_id[token] = next_id
                self.id_to_token[next_id] = token
                next_id += 1
                
        self._fitted = True
        logger.info(f"Vocabulary built with {len(self.token_to_id)} tokens")
        
        return self
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize a single text."""
        if self.lowercase:
            text = text.lower()
            
        if self.tokenization == "word":
            # Simple word tokenization
            tokens = re.findall(r'\b\w+\b', text)
        elif self.tokenization == "char":
            tokens = list(text)
        elif self.tokenization == "bpe":
            # Simplified BPE-style (word pieces)
            tokens = re.findall(r'\b\w+\b', text)
            tokens = [t[:4] for t in tokens]  # Truncate for demo
        else:
            tokens = text.split()
            
        return tokens
    
    def encode(
        self,
        text: str,
        max_length: Optional[int] = None,
        add_special_tokens: bool = True
    ) -> List[int]:
        """
        Encode text to token IDs.
        
        Args:
            text: Input text
            max_length: Maximum sequence length
            add_special_tokens: Add BOS/EOS tokens
            
        Returns:
            List of token IDs
        """
        tokens = self._tokenize(text)
        
        ids = []
        if add_special_tokens:
            ids.append(self.token_to_id[self.special_tokens["bos"]])
            
        for token in tokens:
            ids.append(self.token_to_id.get(
                token,
                self.token_to_id[self.special_tokens["unk"]]
            ))
            
        if add_special_tokens:
            ids.append(self.token_to_id[self.special_tokens["eos"]])
            
        if max_length is not None:
            ids = ids[:max_length]
            
        return ids
    
    def decode(
        self,
        ids: List[int],
        skip_special_tokens: bool = True
    ) -> str:
        """
        Decode token IDs to text.
        
        Args:
            ids: List of token IDs
            skip_special_tokens: Skip special tokens in output
            
        Returns:
            Decoded text
        """
        tokens = []
        special_ids = {self.token_to_id[t] for t in self.special_tokens.values()}
        
        for id_ in ids:
            if skip_special_tokens and id_ in special_ids:
                continue
            tokens.append(self.id_to_token.get(id_, self.special_tokens["unk"]))
            
        return " ".join(tokens)
    
    @property
    def vocab_size_actual(self) -> int:
        """Get actual vocabulary size."""
        return len(self.token_to_id)
    
    @property
    def pad_token_id(self) -> int:
        """Get padding token ID."""
        return self.token_to_id[self.special_tokens["pad"]]
    
    @property
    def unk_token_id(self) -> int:
        """Get unknown token ID."""
        return self.token_to_id[self.special_tokens["unk"]]
    
    def save(self, path: Path) -> None:
        """Save tokenizer to file."""
        state = {
            "token_to_id": self.token_to_id,
            "id_to_token": {int(k): v for k, v in self.id_to_token.items()},
            "config": {
                "vocab_size": self.vocab_size,
                "tokenization": self.tokenization,
                "lowercase": self.lowercase,
                "min_frequency": self.min_frequency,
                "special_tokens": self.special_tokens
            }
        }
        with open(path, "w") as f:
            json.dump(state, f)
            
    @classmethod
    def load(cls, path: Path) -> "Tokenizer":
        """Load tokenizer from file."""
        with open(path) as f:
            state = json.load(f)
            
        tokenizer = cls(**state["config"])
        tokenizer.token_to_id = state["token_to_id"]
        tokenizer.id_to_token = {int(k): v for k, v in state["id_to_token"].items()}
        tokenizer._fitted = True
        
        return tokenizer


class TextDataset(BaseDataset):
    """
    Dataset for text data with tokenization.
    
    Features:
    - Multiple tokenization strategies
    - Vocabulary management
    - Text preprocessing
    - Sensitive term detection
    """
    
    def __init__(
        self,
        data_path: Optional[Union[str, Path]] = None,
        texts: Optional[List[str]] = None,
        labels: Optional[List[Any]] = None,
        schema_path: Optional[Union[str, Path]] = None,
        tokenizer: Optional[Tokenizer] = None,
        max_length: int = 256,
        text_column: str = "text",
        label_column: Optional[str] = None,
        sensitive_attributes: Optional[List[str]] = None,
        preprocess_fn: Optional[Callable] = None,
        transform: Optional[Any] = None
    ):
        """
        Initialize the text dataset.
        
        Args:
            data_path: Path to data file (CSV, JSON, TXT)
            texts: Direct list of texts (alternative to data_path)
            labels: Optional labels for each text
            schema_path: Path to schema file
            tokenizer: Tokenizer instance (will be created if None)
            max_length: Maximum sequence length
            text_column: Name of text column in data file
            label_column: Name of label column
            sensitive_attributes: List of sensitive attribute names
            preprocess_fn: Optional preprocessing function
            transform: Optional transform
        """
        super().__init__(data_path, transform, sensitive_attributes)
        
        self.texts: List[str] = texts or []
        self.labels: List[Any] = labels or []
        self.schema_path = Path(schema_path) if schema_path else None
        self.tokenizer = tokenizer or Tokenizer()
        self.max_length = max_length
        self.text_column = text_column
        self.label_column = label_column
        self.preprocess_fn = preprocess_fn
        
        self._encoded_texts: List[List[int]] = []
        self._sensitive_terms: Dict[str, List[str]] = {}
        
        # Load schema if provided
        if self.schema_path:
            self._load_schema()
            
        # Load data if path provided
        if self.data_path and not self.texts:
            self.load_data()
            
    def _load_schema(self) -> None:
        """Load schema from JSON file."""
        if not self.schema_path.exists():
            logger.warning(f"Schema file not found: {self.schema_path}")
            return
            
        with open(self.schema_path) as f:
            schema = json.load(f)
            
        self.max_length = schema.get("fields", [{}])[0].get("max_length", 256)
        
        # Load sensitive terms
        if "protected_content" in schema:
            self._sensitive_terms = schema["protected_content"].get(
                "protected_word_lists", {}
            )
            
        self._metadata["schema"] = schema
        
    def load_data(self) -> None:
        """Load data from file."""
        if not self.data_path:
            raise ValueError("No data path specified")
            
        suffix = self.data_path.suffix.lower()
        
        if suffix == ".csv":
            import pandas as pd
            df = pd.read_csv(self.data_path)
            self.texts = df[self.text_column].tolist()
            if self.label_column and self.label_column in df.columns:
                self.labels = df[self.label_column].tolist()
                
        elif suffix == ".json":
            with open(self.data_path) as f:
                data = json.load(f)
            if isinstance(data, list):
                self.texts = [item.get(self.text_column, "") for item in data]
                if self.label_column:
                    self.labels = [item.get(self.label_column) for item in data]
            else:
                self.texts = data.get(self.text_column, [])
                self.labels = data.get(self.label_column, [])
                
        elif suffix == ".txt":
            with open(self.data_path) as f:
                self.texts = [line.strip() for line in f if line.strip()]
        else:
            raise ValueError(f"Unsupported file format: {suffix}")
            
        # Preprocess texts
        if self.preprocess_fn:
            self.texts = [self.preprocess_fn(t) for t in self.texts]
            
        # Fit tokenizer if not fitted
        if not self.tokenizer._fitted:
            self.tokenizer.fit(self.texts)
            
        # Encode texts
        self._encode_texts()
        
        logger.info(f"Loaded {len(self.texts)} text samples")
        
    def _encode_texts(self) -> None:
        """Encode all texts."""
        self._encoded_texts = [
            self.tokenizer.encode(text, self.max_length)
            for text in self.texts
        ]
        
    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self._encoded_texts) if self._encoded_texts else len(self.texts)
        
    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        Get a single sample.
        
        Args:
            index: Sample index
            
        Returns:
            Dictionary with input_ids, attention_mask, and optional labels
        """
        # Get encoded text
        if self._encoded_texts:
            input_ids = self._encoded_texts[index]
        else:
            input_ids = self.tokenizer.encode(self.texts[index], self.max_length)
            
        # Create attention mask
        attention_mask = [1] * len(input_ids)
        
        # Pad to max length
        padding_length = self.max_length - len(input_ids)
        if padding_length > 0:
            input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_length
            attention_mask = attention_mask + [0] * padding_length
            
        sample = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "text": self.texts[index],
            "index": index
        }
        
        # Add label if present
        if self.labels:
            label = self.labels[index]
            if isinstance(label, str):
                # Encode string label
                label = hash(label) % 10000  # Simple hash for demo
            sample["label"] = torch.tensor(label, dtype=torch.long)
            
        # Check for sensitive terms
        if self._sensitive_terms:
            sample["has_sensitive"] = self._check_sensitive(self.texts[index])
            
        # Apply transform
        if self.transform:
            sample = self.transform(sample)
            
        return sample
    
    def _check_sensitive(self, text: str) -> torch.Tensor:
        """Check for sensitive terms in text."""
        text_lower = text.lower()
        detected = []
        
        for category, terms in self._sensitive_terms.items():
            found = any(term.lower() in text_lower for term in terms)
            detected.append(float(found))
            
        return torch.tensor(detected, dtype=torch.float32)
    
    def get_vocabulary_size(self) -> int:
        """Get vocabulary size."""
        return self.tokenizer.vocab_size_actual
    
    def decode(self, ids: List[int]) -> str:
        """Decode token IDs to text."""
        return self.tokenizer.decode(ids)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        stats = super().get_statistics()
        
        if self.texts:
            lengths = [len(t.split()) for t in self.texts]
            stats.update({
                "num_texts": len(self.texts),
                "avg_length": np.mean(lengths),
                "max_length": max(lengths),
                "min_length": min(lengths),
                "vocabulary_size": self.get_vocabulary_size(),
            })
            
        return stats


class TextDataLoader(BaseDataLoader):
    """
    Data loader for text data.
    """
    
    def __init__(
        self,
        dataset: TextDataset,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 4,
        pin_memory: bool = True,
        drop_last: bool = False,
        sampler_strategy: Optional[str] = None
    ):
        """
        Initialize the text data loader.
        
        Args:
            dataset: Text dataset
            batch_size: Batch size
            shuffle: Whether to shuffle
            num_workers: Number of workers
            pin_memory: Pin memory for GPU
            drop_last: Drop last batch
            sampler_strategy: Fairness sampling strategy
        """
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            sampler_strategy=sampler_strategy,
            group_labels=None  # Text data doesn't use group sampling by default
        )
        
    def decode_batch(self, batch: Dict[str, Any]) -> List[str]:
        """
        Decode a batch of token IDs to texts.
        
        Args:
            batch: Batch dictionary
            
        Returns:
            List of decoded texts
        """
        input_ids = batch["input_ids"]
        return [
            self.dataset.decode(ids.tolist())
            for ids in input_ids
        ]
