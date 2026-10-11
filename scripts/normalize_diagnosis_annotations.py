"""Normalize Diagnosis annotation Excel files into machine-readable JSONL.

One-time conversion. Reads only sample_id, primary_label, secondary_label and
the review-type column; gold answers/labels, EM/accuracy references, C1 text
and evidence columns are never read. Output rows follow the canonical C1
sample order. The source Excel files are opened read-only and never modified.

NQ labels come from NQ_C1_3216_initial_labels.xlsx (restricted to the 500 C1
sample IDs). NQ_C1_500_initial_labels.xlsx is a historical reference only.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.diagnosis.annotations import AnnotationRecord
from src.diagnosis.dev_dataset import load_jsonl
from src.diagnosis.labels import DiagnosisLabel
from src.diagnosis.schema import DatasetName, normalize_sample_id

ANNOTATION_DIR = ROOT / "data" / "annotations"
OUTPUT_DIR = ANNOTATION_DIR / "normalized"
C1_DIR = ROOT / "results" / "preliminary"

PRIMARY_LABEL_MAP: dict[str, DiagnosisLabel] = {
    "Correct / No Failure": DiagnosisLabel.CORRECT,
    "Evidence Insufficiency": DiagnosisLabel.EVIDENCE_INSUFFICIENCY,
    "Evidence Accessibility Failure": DiagnosisLabel.EVIDENCE_ACCESSIBILITY,
    "Evidence Utilisation Failure": DiagnosisLabel.EVIDENCE_UTILISATION,
}

SECONDARY_LABEL_MAP: dict[str, str] = {
    "7B Oracle Recovered": "oracle_recovered_7b",
    "Likely Small-Model Capability Limitation": "likely_small_model_capability_limitation",
    "Unresolved / General Hard Case": "unresolved_hard_case",
}

# Keys are the verbatim review types used in the source workbooks.
REVIEW_REASON_MAP: dict[str, str] = {
    "AI flagged: review question scope, reference answer or source-text integrity; "
    "not a confirmed dataset defect.": "ai_flagged",
    "标准答案与问法/事实明显冲突": "gold_conflict",
    "问法、时间或答案范围可能有多解": "ambiguous_scope",
    "gold 与提供的证据冲突": "gold_conflict",
    "gold 与神谕证据冲突": "gold_conflict",
    "可访问性/利用失败边界待复核": "label_boundary",
    "输出标签与解释矛盾": "output_inconsistent",
    "检索上下文自身冲突": "context_conflict",
    "声明用词异常": "claim_wording",
    "声明拼写及诊断边界": "claim_wording",
    "声明拼写与 gold 冲突": "claim_wording",
    "声明实体名与 oracle 不符": "claim_wording",
    "神谕证据对所问关系不充分": "oracle_insufficient",
    "神谕证据没有回答所问角色": "oracle_insufficient",
    "神谕证据缺少声明时间限定": "oracle_insufficient",
    "参考证据未充分反驳声明": "oracle_insufficient",
    "模糊声明与参考证据不足": "ambiguous_scope",
    "作品指代需要复核": "ambiguous_scope",
    "同名人物指代不明确": "ambiguous_scope",
    "国籍与法律公民身份范围": "ambiguous_scope",
    "颜色范围读法": "ambiguous_scope",
    "军种训练时间范围": "ambiguous_scope",
    "once 的限定解释": "ambiguous_scope",
    "任职年数精度": "ambiguous_scope",
    "作者与故事主题关系": "ambiguous_scope",
}


@dataclass(frozen=True)
class AnnotationSource:
    dataset: DatasetName
    workbook: Path
    expected_rows: int
    main_sheet: str
    review_sheet: str
    review_column: str
    c1_path: Path
    output: Path


SOURCES: tuple[AnnotationSource, ...] = (
    AnnotationSource(
        dataset=DatasetName.NQ,
        workbook=ANNOTATION_DIR / "NQ_C1_3216_initial_labels.xlsx",
        expected_rows=3216,
        main_sheet="NQ C1 Annotation",
        review_sheet="Quality Review",
        review_column="review_status",
        c1_path=C1_DIR / "nq_c1_7b.jsonl",
        output=OUTPUT_DIR / "nq_500_labels.jsonl",
    ),
    AnnotationSource(
        dataset=DatasetName.FEVER,
        workbook=ANNOTATION_DIR / "FEVER_C1_500_initial_labels.xlsx",
        expected_rows=500,
        main_sheet="FEVER C1 Annotation",
        review_sheet="数据质量提示",
        review_column="提示类型",
        c1_path=C1_DIR / "fever_c1_7b.jsonl",
        output=OUTPUT_DIR / "fever_500_labels.jsonl",
    ),
)


def excel_sample_id(value: object) -> str:
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return normalize_sample_id(value)


def map_primary_label(value: object) -> DiagnosisLabel:
    text = value.strip() if isinstance(value, str) else value
    if text not in PRIMARY_LABEL_MAP:
        raise ValueError(f"Unmapped primary label {value!r}")
    return PRIMARY_LABEL_MAP[text]


def map_secondary_label(value: object) -> str | None:
    text = value.strip() if isinstance(value, str) else value
    if text in (None, ""):
        return None
    if text not in SECONDARY_LABEL_MAP:
        raise ValueError(f"Unmapped secondary label {value!r}")
    return SECONDARY_LABEL_MAP[text]


def map_review_reason(value: object) -> str:
    text = value.strip() if isinstance(value, str) else value
    if text not in REVIEW_REASON_MAP:
        raise ValueError(f"Unmapped review type {value!r}")
    return REVIEW_REASON_MAP[text]


def read_columns(workbook, sheet: str, columns: tuple[str, ...]) -> list[dict]:
    """Return only the requested columns of a sheet; other columns are never read into memory."""
    rows = workbook[sheet].iter_rows(values_only=True)
    header = [str(cell).strip() if cell is not None else "" for cell in next(rows)]
    missing = [column for column in columns if column not in header]
    if missing:
        raise ValueError(f"Sheet {sheet!r} is missing columns {missing}")
    positions = [header.index(column) for column in columns]
    return [
        {column: row[position] for column, position in zip(columns, positions)}
        for row in rows
        if any(cell not in (None, "") for cell in row)
    ]


def index_by_sample_id(rows: list[dict], sheet: str) -> dict[str, dict]:
    indexed: dict[str, dict] = {}
    for row in rows:
        sample_id = excel_sample_id(row["sample_id"])
        if sample_id in indexed:
            raise ValueError(f"Duplicate sample_id {sample_id!r} in sheet {sheet!r}")
        indexed[sample_id] = row
    return indexed


def build_annotations(source: AnnotationSource) -> list[AnnotationRecord]:
    from openpyxl import load_workbook

    workbook = load_workbook(source.workbook, read_only=True, data_only=True)
    try:
        main = index_by_sample_id(
            read_columns(workbook, source.main_sheet, ("sample_id", "primary_label", "secondary_label")),
            source.main_sheet,
        )
        reviews = index_by_sample_id(
            read_columns(workbook, source.review_sheet, ("sample_id", source.review_column)),
            source.review_sheet,
        )
    finally:
        workbook.close()

    if len(main) != source.expected_rows:
        raise ValueError(f"{source.workbook.name}: expected {source.expected_rows} rows, got {len(main)}")
    unknown_reviews = set(reviews) - set(main)
    if unknown_reviews:
        raise ValueError(f"{source.workbook.name}: review rows without main rows: {sorted(unknown_reviews)[:5]}")

    c1_ids = [normalize_sample_id(record["sample_id"]) for record in load_jsonl(source.c1_path)]
    missing = [sample_id for sample_id in c1_ids if sample_id not in main]
    if missing:
        raise ValueError(f"{source.workbook.name}: {len(missing)} C1 sample IDs have no label, e.g. {missing[:5]}")

    annotations = []
    for sample_id in c1_ids:
        row = main[sample_id]
        review = reviews.get(sample_id)
        annotations.append(
            AnnotationRecord(
                sample_id=sample_id,
                dataset=source.dataset,
                primary_label=map_primary_label(row["primary_label"]),
                secondary_label=map_secondary_label(row["secondary_label"]),
                needs_manual_review=review is not None,
                review_reason=map_review_reason(review[source.review_column]) if review else None,
                annotation_source=source.workbook.name,
            )
        )
    return annotations


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    tmp.replace(path)


def main() -> int:
    for source in SOURCES:
        annotations = build_annotations(source)
        write_jsonl(source.output, [annotation.to_dict() for annotation in annotations])
        labels = Counter(annotation.primary_label.value for annotation in annotations)
        review = sum(annotation.needs_manual_review for annotation in annotations)
        print(f"{source.dataset.value}: {len(annotations)} annotations -> {source.output.relative_to(ROOT)}")
        print(f"  primary_label: {dict(labels)}")
        print(f"  needs_manual_review: {review}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
