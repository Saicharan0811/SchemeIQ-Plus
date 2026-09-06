"""Foundation, manifest, and environment verification tests for SchemeIQ+."""

from pathlib import Path
import pandas as pd
import pytest


def test_directory_structure():
    """Verify that all core directories and files exist."""
    base_dir = Path(__file__).resolve().parent.parent

    expected_paths = [
        base_dir / "data",
        base_dir / "data" / "documents",
        base_dir / "data" / "documents" / "source_manifest.csv",
        base_dir / "data" / "verification_report.md",
        base_dir / "src" / "__init__.py",
        base_dir / "src" / "ingestion" / "__init__.py",
        base_dir / "src" / "retrieval" / "__init__.py",
        base_dir / "src" / "eligibility" / "__init__.py",
        base_dir / "src" / "recommendation" / "__init__.py",
        base_dir / "tests" / "__init__.py",
        base_dir / ".env",
        base_dir / ".gitignore",
        base_dir / "requirements.txt",
        base_dir / "README.md",
    ]

    for p in expected_paths:
        assert p.exists(), f"Expected path does not exist: {p}"


def test_schemes_dataset_presence():
    """Verify that a verified schemes dataset exists in the data directory."""
    base_dir = Path(__file__).resolve().parent.parent
    schemes_csv = base_dir / "data" / "schemes.csv"
    starter_csv = base_dir / "data" / "SchemeIQ_Plus_Telangana_Starter_Dataset(1).csv"

    dataset_path = schemes_csv if schemes_csv.exists() else starter_csv
    assert dataset_path.exists(), "No valid schemes dataset found in data/"

    df = pd.read_csv(dataset_path)
    assert len(df) in (0, 14), f"Unexpected dataset row count: {len(df)}"
    assert "scheme_id" in df.columns, "scheme_id column missing from dataset"


def test_source_manifest():
    """Verify that source_manifest.csv is present, covers 14 schemes, and has 0 missing fields."""
    manifest_path = Path(__file__).resolve().parent.parent / "data" / "documents" / "source_manifest.csv"
    assert manifest_path.exists(), "data/documents/source_manifest.csv is missing"

    df = pd.read_csv(manifest_path)
    expected_columns = [
        "scheme_id",
        "scheme_name",
        "document_title",
        "document_type",
        "official_url",
        "source_authority",
        "publication_or_update_date",
    ]
    assert list(df.columns) == expected_columns, f"Unexpected manifest columns: {list(df.columns)}"
    assert df["scheme_id"].nunique() == 14, f"Expected 14 unique schemes, found {df['scheme_id'].nunique()}"
    assert df.isnull().sum().sum() == 0, "Found unexpected null values in source manifest"


def test_package_imports():
    """Verify that all pipeline packages import without error."""
    import src
    import src.ingestion
    import src.retrieval
    import src.eligibility
    import src.recommendation

    assert src.__version__ == "0.1.0"
