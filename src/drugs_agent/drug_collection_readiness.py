"""Confirm the drug collection exists and holds data before the agent uses it."""

from pymilvus import FunctionType

from src.adapters.vector_store.exceptions import VectorDBCollectionError
from src.adapters.vector_store.milvus_vector_store import MilvusProvider

from .milvus_drug_search import DRUG_NAME_SPARSE_FIELD


async def verify_drug_collection_ready_for_search(milvus_provider: MilvusProvider) -> int:
    """Return the drug record count, raising if the collection cannot support search."""
    await milvus_provider.connect()
    await milvus_provider.check_collection()
    await _require_drug_name_bm25_function(milvus_provider)
    drug_record_count = await milvus_provider.count_entities()
    if drug_record_count == 0:
        raise VectorDBCollectionError(
            f"Collection '{milvus_provider.collection_name}' is empty. "
            "Run drug_data_ingestion_pipeline/drug_ingestion_pipeline.py first."
        )
    return drug_record_count


async def _require_drug_name_bm25_function(milvus_provider: MilvusProvider) -> None:
    drug_collection_schema = await milvus_provider.describe_collection()
    has_drug_name_bm25_function = any(
        collection_function["type"] == FunctionType.BM25
        and DRUG_NAME_SPARSE_FIELD in collection_function["output_field_names"]
        for collection_function in drug_collection_schema["functions"]
    )
    if not has_drug_name_bm25_function:
        raise VectorDBCollectionError(
            f"Collection '{milvus_provider.collection_name}' has no BM25 field '{DRUG_NAME_SPARSE_FIELD}', "
            "so it was built before hybrid search. Re-run the ingestion pipeline "
            "with drop_existing: true."
        )
