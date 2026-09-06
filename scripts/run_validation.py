import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.corpus_validator import validate_corpus, write_validation_reports

def main():
    raw_dir = Path("data/documents/raw")
    stats = validate_corpus(raw_dir)
    write_validation_reports(stats, raw_dir / "corpus_validation_report.json", raw_dir / "corpus_validation_report.md")
    print(f"Validated {stats['total_files_inspected']} files.")
    print(f"Valid: {stats['valid_documents']}")
    print(f"Missing: {stats['missing_coverage_scheme_ids']}")

if __name__ == "__main__":
    main()
