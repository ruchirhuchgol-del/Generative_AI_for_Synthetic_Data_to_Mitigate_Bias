#!/usr/bin/env python
"""
Generate Synthetic Schema
=========================

Script for generating synthetic data schemas for different modalities.
Creates JSON schema definitions that can be used for data validation,
synthetic data generation, and fairness analysis.

Usage:
    python generate_synthetic_schema.py [OPTIONS]

Options:
    --modality TYPE       Modality type: tabular, image, text, multimodal
    --output PATH         Output path for schema file
    --sensitive-attrs     Comma-separated list of sensitive attributes
    --num-features N      Number of features for tabular schema
    --image-size SIZE     Image dimensions (e.g., 64x64 or 256x256x3)
    --text-type TYPE      Text type: sequence, document, structured
    --seed N              Random seed for reproducibility
    --template NAME       Use predefined template (adult, compas, credit)
    -h, --help            Show this help message

Examples:
    python generate_synthetic_schema.py --modality tabular --num-features 20
    python generate_synthetic_schema.py --template adult --output schemas/adult.json
    python generate_synthetic_schema.py --modality multimodal --sensitive-attrs gender,race
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np


# Predefined templates
TEMPLATES = {
    "adult": {
        "name": "adult_census",
        "description": "Adult Census Income dataset schema",
        "tabular": {
            "features": [
                {"name": "age", "type": "numerical", "range": [17, 90], "description": "Age of individual"},
                {"name": "workclass", "type": "categorical", "categories": ["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov", "State-gov", "Without-pay", "Never-worked"]},
                {"name": "fnlwgt", "type": "numerical", "range": [12285, 1484705], "description": "Final weight"},
                {"name": "education", "type": "categorical", "categories": ["Bachelors", "Some-college", "11th", "HS-grad", "Prof-school", "Assoc-acdm", "Assoc-voc", "9th", "7th-8th", "12th", "Masters", "1st-4th", "10th", "Doctorate", "5th-6th", "Preschool"]},
                {"name": "education_num", "type": "numerical", "range": [1, 16], "description": "Education years"},
                {"name": "marital_status", "type": "categorical", "categories": ["Married-civ-spouse", "Divorced", "Never-married", "Separated", "Widowed", "Married-spouse-absent", "Married-AF-spouse"]},
                {"name": "occupation", "type": "categorical", "categories": ["Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial", "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical", "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv", "Armed-Forces"]},
                {"name": "relationship", "type": "categorical", "categories": ["Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried"]},
                {"name": "race", "type": "categorical", "categories": ["White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"], "sensitive": True},
                {"name": "sex", "type": "categorical", "categories": ["Male", "Female"], "sensitive": True},
                {"name": "capital_gain", "type": "numerical", "range": [0, 99999]},
                {"name": "capital_loss", "type": "numerical", "range": [0, 4356]},
                {"name": "hours_per_week", "type": "numerical", "range": [1, 99]},
                {"name": "native_country", "type": "categorical", "categories": ["United-States", "Cambodia", "England", "Puerto-Rico", "Canada", "Germany", "Outlying-US(Guam-USVI-etc)", "India", "Japan", "Greece", "South", "China", "Cuba", "Iran", "Honduras", "Philippines", "Italy", "Poland", "Jamaica", "Vietnam", "Mexico", "Portugal", "Ireland", "France", "Dominican-Republic", "Laos", "Ecuador", "Taiwan", "Haiti", "Columbia", "Hungary", "Guatemala", "Nicaragua", "Scotland", "Thailand", "Yugoslavia", "El-Salvador", "Trinadad&Tobago", "Peru", "Hong", "Holand-Netherlands"]},
            ],
            "target": {"name": "income", "type": "binary", "categories": ["<=50K", ">50K"]},
            "sensitive_attributes": ["race", "sex"]
        }
    },
    "compas": {
        "name": "compas_recidivism",
        "description": "COMPAS Recidivism dataset schema",
        "tabular": {
            "features": [
                {"name": "age", "type": "numerical", "range": [18, 96]},
                {"name": "c_charge_degree", "type": "categorical", "categories": ["F", "M"]},
                {"name": "race", "type": "categorical", "categories": ["African-American", "Caucasian", "Hispanic", "Asian", "Native American", "Other"], "sensitive": True},
                {"name": "sex", "type": "categorical", "categories": ["Male", "Female"], "sensitive": True},
                {"name": "age_cat", "type": "categorical", "categories": ["Less than 25", "25 - 45", "Greater than 45"]},
                {"name": "juv_fel_count", "type": "numerical", "range": [0, 20]},
                {"name": "juv_misd_count", "type": "numerical", "range": [0, 13]},
                {"name": "juv_other_count", "type": "numerical", "range": [0, 17]},
                {"name": "priors_count", "type": "numerical", "range": [0, 38]},
                {"name": "days_b_screening_arrest", "type": "numerical", "range": [-414, 1027]},
                {"name": "c_days_from_compas", "type": "numerical", "range": [0, 979]},
                {"name": "is_recid", "type": "binary"},
                {"name": "r_charge_degree", "type": "categorical", "categories": ["(F1)", "(F2)", "(F3)", "(F7)", "(M1)", "(M2)", "(MO3)", "nan"]},
                {"name": "is_violent_recid", "type": "binary"},
            ],
            "target": {"name": "two_year_recid", "type": "binary"},
            "sensitive_attributes": ["race", "sex"]
        }
    },
    "credit": {
        "name": "credit_default",
        "description": "Credit Card Default dataset schema",
        "tabular": {
            "features": [
                {"name": "LIMIT_BAL", "type": "numerical", "range": [10000, 1000000], "description": "Credit limit"},
                {"name": "SEX", "type": "categorical", "categories": ["Male", "Female"], "sensitive": True},
                {"name": "EDUCATION", "type": "categorical", "categories": ["Graduate", "University", "High School", "Others"]},
                {"name": "MARRIAGE", "type": "categorical", "categories": ["Married", "Single", "Others"]},
                {"name": "AGE", "type": "numerical", "range": [21, 79]},
                *[{"name": f"PAY_{i}", "type": "numerical", "range": [-2, 8]} for i in [0, 2, 3, 4, 5, 6]],
                *[{"name": f"BILL_AMT{i}", "type": "numerical", "range": [-165580, 964511]} for i in [1, 2, 3, 4, 5, 6]],
                *[{"name": f"PAY_AMT{i}", "type": "numerical", "range": [0, 873552]} for i in [1, 2, 3, 4, 5, 6]],
            ],
            "target": {"name": "default_payment", "type": "binary"},
            "sensitive_attributes": ["SEX"]
        }
    },
    "healthcare": {
        "name": "healthcare_costs",
        "description": "Healthcare costs dataset schema",
        "tabular": {
            "features": [
                {"name": "age", "type": "numerical", "range": [18, 64]},
                {"name": "sex", "type": "categorical", "categories": ["male", "female"], "sensitive": True},
                {"name": "bmi", "type": "numerical", "range": [15, 53], "description": "Body Mass Index"},
                {"name": "children", "type": "numerical", "range": [0, 5], "description": "Number of children"},
                {"name": "smoker", "type": "categorical", "categories": ["yes", "no"]},
                {"name": "region", "type": "categorical", "categories": ["northeast", "northwest", "southeast", "southwest"]},
            ],
            "target": {"name": "charges", "type": "numerical", "range": [1121, 63770], "description": "Medical charges"},
            "sensitive_attributes": ["sex", "age"]
        }
    }
}


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate Synthetic Data Schema",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--modality",
        type=str,
        choices=["tabular", "image", "text", "multimodal"],
        default="tabular",
        help="Modality type for schema"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for schema file"
    )
    
    parser.add_argument(
        "--sensitive-attrs",
        type=str,
        default=None,
        help="Comma-separated list of sensitive attributes"
    )
    
    parser.add_argument(
        "--num-features",
        type=int,
        default=10,
        help="Number of features for tabular schema"
    )
    
    parser.add_argument(
        "--num-sensitive",
        type=int,
        default=2,
        help="Number of sensitive attributes to include"
    )
    
    parser.add_argument(
        "--image-size",
        type=str,
        default="64x64x3",
        help="Image dimensions (e.g., 64x64 or 256x256x3)"
    )
    
    parser.add_argument(
        "--text-type",
        type=str,
        choices=["sequence", "document", "structured"],
        default="document",
        help="Text type for text schema"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    parser.add_argument(
        "--template",
        type=str,
        choices=list(TEMPLATES.keys()),
        default=None,
        help="Use predefined template"
    )
    
    parser.add_argument(
        "--name",
        type=str,
        default="custom_schema",
        help="Schema name"
    )
    
    parser.add_argument(
        "--description",
        type=str,
        default="Custom generated schema",
        help="Schema description"
    )
    
    return parser.parse_args()


def generate_tabular_schema(
    num_features: int,
    num_sensitive: int,
    sensitive_attrs: Optional[List[str]] = None,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Generate a tabular data schema.
    
    Args:
        num_features: Total number of features
        num_sensitive: Number of sensitive attributes
        sensitive_attrs: Optional list of sensitive attribute names
        seed: Random seed
        
    Returns:
        Dictionary containing the schema definition
    """
    np.random.seed(seed)
    
    # Feature types distribution
    feature_types = ["numerical", "categorical", "binary"]
    
    features = []
    sensitive_list = []
    
    # Generate sensitive attributes first
    if sensitive_attrs:
        for i, attr_name in enumerate(sensitive_attrs[:num_sensitive]):
            feature = {
                "name": attr_name,
                "type": "categorical",
                "categories": [f"group_{j}" for j in range(np.random.randint(2, 5))],
                "sensitive": True,
                "description": f"Sensitive attribute: {attr_name}"
            }
            features.append(feature)
            sensitive_list.append(attr_name)
    else:
        # Generate default sensitive attributes
        sensitive_names = ["gender", "race", "age_group", "disability_status", "nationality"]
        for i in range(min(num_sensitive, len(sensitive_names))):
            feature = {
                "name": sensitive_names[i],
                "type": "categorical",
                "categories": [f"group_{j}" for j in range(np.random.randint(2, 4))],
                "sensitive": True,
                "description": f"Sensitive attribute: {sensitive_names[i]}"
            }
            features.append(feature)
            sensitive_list.append(sensitive_names[i])
    
    # Generate remaining features
    remaining_features = num_features - len(features)
    
    for i in range(remaining_features):
        ftype = np.random.choice(feature_types, p=[0.5, 0.35, 0.15])
        
        if ftype == "numerical":
            min_val = float(np.random.randint(-100, 0))
            max_val = float(np.random.randint(1, 1000))
            feature = {
                "name": f"feature_{i}",
                "type": "numerical",
                "range": [min_val, max_val],
                "distribution": np.random.choice(["normal", "uniform", "lognormal"]),
                "description": f"Numerical feature {i}"
            }
        elif ftype == "categorical":
            n_categories = np.random.randint(3, 10)
            feature = {
                "name": f"feature_{i}",
                "type": "categorical",
                "categories": [f"cat_{j}" for j in range(n_categories)],
                "description": f"Categorical feature {i}"
            }
        else:  # binary
            feature = {
                "name": f"feature_{i}",
                "type": "binary",
                "description": f"Binary feature {i}"
            }
        
        features.append(feature)
    
    # Add target variable
    target = {
        "name": "target",
        "type": np.random.choice(["binary", "categorical", "numerical"]),
        "description": "Target variable for prediction"
    }
    
    if target["type"] == "categorical":
        target["categories"] = [f"class_{j}" for j in range(np.random.randint(3, 6))]
    
    return {
        "features": features,
        "target": target,
        "sensitive_attributes": sensitive_list
    }


def generate_image_schema(
    image_size: str,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Generate an image data schema.
    
    Args:
        image_size: Image dimensions string (e.g., "64x64x3")
        seed: Random seed
        
    Returns:
        Dictionary containing the image schema definition
    """
    # Parse image size
    dims = [int(d) for d in image_size.lower().split('x')]
    
    if len(dims) == 2:
        height, width = dims
        channels = 3
    elif len(dims) == 3:
        height, width, channels = dims
    else:
        raise ValueError(f"Invalid image size format: {image_size}")
    
    return {
        "format": "image",
        "dimensions": {
            "height": height,
            "width": width,
            "channels": channels
        },
        "color_mode": "rgb" if channels == 3 else "grayscale",
        "normalization": {
            "method": "min_max",
            "range": [-1, 1]
        },
        "augmentation": {
            "enabled": True,
            "methods": ["random_flip", "random_crop", "color_jitter"]
        },
        "sensitive_attributes": [
            {
                "name": "protected_visual_features",
                "type": "region_based",
                "description": "Visual features that could reveal sensitive attributes"
            }
        ]
    }


def generate_text_schema(
    text_type: str,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Generate a text data schema.
    
    Args:
        text_type: Type of text data
        seed: Random seed
        
    Returns:
        Dictionary containing the text schema definition
    """
    base_schema = {
        "format": "text",
        "encoding": "utf-8",
        "language": "en",
    }
    
    if text_type == "sequence":
        base_schema.update({
            "max_length": 512,
            "min_length": 10,
            "tokenization": {
                "method": "wordpiece",
                "vocab_size": 30522,
                "special_tokens": ["[CLS]", "[SEP]", "[PAD]", "[UNK]", "[MASK]"]
            },
            "structure": "sequence"
        })
    elif text_type == "document":
        base_schema.update({
            "max_length": 4096,
            "min_length": 100,
            "structure": "document",
            "paragraphs": {
                "min_paragraphs": 1,
                "max_paragraphs": 20
            },
            "tokenization": {
                "method": "sentencepiece",
                "vocab_size": 32000
            }
        })
    else:  # structured
        base_schema.update({
            "structure": "structured",
            "fields": [
                {"name": "title", "max_length": 100, "required": True},
                {"name": "content", "max_length": 5000, "required": True},
                {"name": "metadata", "type": "dict", "required": False}
            ]
        })
    
    base_schema["sensitive_attributes"] = [
        {
            "name": "demographic_markers",
            "type": "pattern_based",
            "description": "Text patterns that could reveal demographic information"
        }
    ]
    
    return base_schema


def generate_multimodal_schema(
    tabular_features: int = 10,
    image_size: str = "64x64x3",
    text_type: str = "document",
    sensitive_attrs: Optional[List[str]] = None,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Generate a multimodal data schema.
    
    Args:
        tabular_features: Number of tabular features
        image_size: Image dimensions
        text_type: Type of text data
        sensitive_attrs: List of sensitive attributes
        seed: Random seed
        
    Returns:
        Dictionary containing the multimodal schema definition
    """
    np.random.seed(seed)
    
    return {
        "format": "multimodal",
        "modalities": {
            "tabular": generate_tabular_schema(
                num_features=tabular_features,
                num_sensitive=0,  # Will be handled at multimodal level
                seed=seed
            ),
            "image": generate_image_schema(image_size, seed),
            "text": generate_text_schema(text_type, seed)
        },
        "cross_modal_constraints": {
            "alignment_required": True,
            "consistency_check": True,
            "feature_mapping": {
                "tabular_to_image": "conditional_generation",
                "tabular_to_text": "description_generation"
            }
        },
        "sensitive_attributes": sensitive_attrs or ["gender", "race"],
        "fusion_strategy": "late_fusion"
    }


def main():
    """Main function."""
    args = parse_args()
    
    # Set seed
    np.random.seed(args.seed)
    
    # Parse sensitive attributes
    sensitive_attrs = None
    if args.sensitive_attrs:
        sensitive_attrs = [s.strip() for s in args.sensitive_attrs.split(",")]
    
    # Generate schema
    if args.template:
        print(f"Using template: {args.template}")
        schema = TEMPLATES[args.template].copy()
        schema["generated_at"] = datetime.now().isoformat()
        schema["version"] = "1.0.0"
    else:
        print(f"Generating {args.modality} schema...")
        
        if args.modality == "tabular":
            tabular_schema = generate_tabular_schema(
                num_features=args.num_features,
                num_sensitive=args.num_sensitive,
                sensitive_attrs=sensitive_attrs,
                seed=args.seed
            )
            schema = {
                "name": args.name,
                "description": args.description,
                "version": "1.0.0",
                "generated_at": datetime.now().isoformat(),
                "modality": "tabular",
                "tabular": tabular_schema
            }
            
        elif args.modality == "image":
            image_schema = generate_image_schema(
                image_size=args.image_size,
                seed=args.seed
            )
            schema = {
                "name": args.name,
                "description": args.description,
                "version": "1.0.0",
                "generated_at": datetime.now().isoformat(),
                "modality": "image",
                "image": image_schema
            }
            
        elif args.modality == "text":
            text_schema = generate_text_schema(
                text_type=args.text_type,
                seed=args.seed
            )
            schema = {
                "name": args.name,
                "description": args.description,
                "version": "1.0.0",
                "generated_at": datetime.now().isoformat(),
                "modality": "text",
                "text": text_schema
            }
            
        elif args.modality == "multimodal":
            multimodal_schema = generate_multimodal_schema(
                tabular_features=args.num_features,
                image_size=args.image_size,
                text_type=args.text_type,
                sensitive_attrs=sensitive_attrs,
                seed=args.seed
            )
            schema = {
                "name": args.name,
                "description": args.description,
                "version": "1.0.0",
                "generated_at": datetime.now().isoformat(),
                "modality": "multimodal",
                **multimodal_schema
            }
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        script_dir = Path(__file__).parent
        project_root = script_dir.parent.parent
        output_dir = project_root / "data" / "schemas"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"{schema.get('name', args.name)}_schema.json"
        output_path = output_dir / filename
    
    # Save schema
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(schema, f, indent=2)
    
    print(f"\nSchema generated successfully!")
    print(f"  Name: {schema.get('name', args.name)}")
    print(f"  Modality: {args.modality if not args.template else 'from template'}")
    print(f"  Output: {output_path}")
    
    # Print summary
    if args.modality == "tabular" or args.template:
        tabular = schema.get("tabular", {})
        features = tabular.get("features", [])
        sensitive = tabular.get("sensitive_attributes", [])
        print(f"\n  Features: {len(features)}")
        print(f"  Sensitive attributes: {len(sensitive)} ({', '.join(sensitive)})")
    
    return schema


if __name__ == "__main__":
    main()
