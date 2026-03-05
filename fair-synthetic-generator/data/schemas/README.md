# Data Schemas

This directory contains JSON Schema definitions for the multimodal data used in the Fair Synthetic Data Generator.

## Schema Files

| File | Description |
|------|-------------|
| `tabular_schema.json` | Schema for tabular data (numerical and categorical features) |
| `text_schema.json` | Schema for text data (tokenization, vocabulary, preprocessing) |
| `image_schema.json` | Schema for image data (dimensions, channels, augmentation) |

## Schema Features

### Tabular Schema (`tabular_schema.json`)
- **Numerical Features**: Define type, range, distribution, and statistics
- **Categorical Features**: Define values, probabilities, ordinal ordering
- **Derived Features**: Bin transformations, computed features
- **Target Variable**: Classification or regression targets
- **Protected Attributes**: Sensitive attributes with privileged/unprivileged values
- **Correlations**: Known correlations to preserve
- **Constraints**: Custom data constraints

### Text Schema (`text_schema.json`)
- **Fields**: Multiple text fields with individual settings
- **Vocabulary**: Tokenization method, size, special tokens
- **Preprocessing**: Lowercase, punctuation, stopwords handling
- **Anonymization**: Pattern-based sensitive content removal
- **Fairness Constraints**: Bias marker removal, neutralization
- **Augmentation**: Synonym replacement, back-translation

### Image Schema (`image_schema.json`)
- **Dimensions**: Width, height, channels, format
- **Fields**: Multiple image fields with content types
- **Fairness Constraints**: Attribute anonymization, balanced distribution
- **Normalization**: Mean/std for preprocessing
- **Augmentation**: Geometric and color transforms
- **Privacy**: Face/text detection, EXIF removal

## Usage

### Loading a Schema

```python
import json

# Load schema
with open("data/schemas/tabular_schema.json") as f:
    schema = json.load(f)

# Access feature definitions
numerical_features = schema["numerical_features"]
categorical_features = schema["categorical_features"]
protected_attributes = schema["protected_attributes"]
```

### Validating Data Against Schema

```python
import jsonschema

# Load schema
with open("data/schemas/tabular_schema.json") as f:
    schema = json.load(f)

# Validate a data record
data_record = {
    "age": 35,
    "income": 75000,
    "gender": "female",
    "education": "bachelor"
}

# Validate (using definitions from schema)
# Note: You may need to adapt the validation for your specific use case
```

### Creating a New Schema

```python
import json

# Define schema
schema = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "schema_name": "my_dataset",
    "version": "1.0.0",
    "description": "Custom dataset schema",
    "numerical_features": [
        {
            "name": "feature1",
            "type": "continuous",
            "min": 0,
            "max": 100,
            "distribution": "normal",
            "mean": 50,
            "std": 15
        }
    ],
    "categorical_features": [
        {
            "name": "category1",
            "type": "nominal",
            "values": ["A", "B", "C"]
        }
    ],
    "protected_attributes": [
        {
            "name": "sensitive_attr",
            "privileged_value": "A",
            "unprivileged_values": ["B", "C"]
        }
    ]
}

# Save schema
with open("data/schemas/my_schema.json", "w") as f:
    json.dump(schema, f, indent=2)
```

## Schema Versioning

Schemas should be versioned using semantic versioning (MAJOR.MINOR.PATCH):

- **MAJOR**: Breaking changes (removing features, changing types)
- **MINOR**: New features (adding new fields, new options)
- **PATCH**: Bug fixes, clarifications

## Contributing

When adding new schemas or modifying existing ones:

1. Validate JSON syntax
2. Test with sample data
3. Update documentation
4. Follow the existing schema structure
