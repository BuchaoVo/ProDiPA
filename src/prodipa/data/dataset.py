"""Fail-closed split-aware record filtering."""

REQUIRED_PROVENANCE = ("seq_sha256", "accessions", "uniprot_ids", "source_release", "dataset_version", "split_version")


def training_records(records):
    selected = []
    for index, record in enumerate(records):
        missing = [field for field in REQUIRED_PROVENANCE if not record.get(field)]
        if missing:
            raise ValueError(f"record {index} missing provenance: {', '.join(missing)}")
        if record.get("split") != "train":
            raise ValueError(f"record {index} is {record.get('split')!r}; training accepts Train only")
        selected.append(record)
    return selected

