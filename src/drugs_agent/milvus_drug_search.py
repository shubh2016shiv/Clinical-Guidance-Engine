"""Search Milvus drug records with dense retrieval and drug-name BM25."""

import asyncio

from pymilvus import AnnSearchRequest, RRFRanker

from src.adapters.vector_store.base import SearchResult
from src.adapters.vector_store.milvus_vector_store import MilvusProvider

DRUG_EMBEDDING_FIELD = "embedding"
DRUG_NAME_SPARSE_FIELD = "drug_name_sparse"
# The ingested vectors are normalized and indexed with inner product.
DRUG_EMBEDDING_METRIC = "IP"
# Give both rankings enough candidates before reciprocal rank fusion.
HYBRID_SEARCH_CANDIDATE_MULTIPLIER = 3
MIN_CANDIDATES_PER_SEARCH = 30


def build_drug_embedding_query_text(drug_name: str | None, drug_class: str | None) -> str:
    """Format search text like the drug records used to build the index."""
    labeled_query_parts = []
    if drug_name and drug_name.strip():
        labeled_query_parts.append(f"Drug: {drug_name.strip()}")
    if drug_class and drug_class.strip():
        labeled_query_parts.append(f"Class: {drug_class.strip()}")
    if not labeled_query_parts:
        raise ValueError("Give a drug_name, a drug_class, or both")
    return " | ".join(labeled_query_parts)


class DrugSearcher:
    """Find drugs in Milvus using dense search and, for names, BM25 with RRF.

    RRF scores come from ranks rather than vector similarity. They should not be
    compared with dense-search scores or used as similarity thresholds.
    """

    def __init__(self, milvus_provider: MilvusProvider, *, rrf_rank_constant: int = 60) -> None:
        self.milvus_provider = milvus_provider
        self.rrf_rank_constant = rrf_rank_constant

    async def search_drug_records(
        self,
        drug_name: str | None = None,
        drug_class: str | None = None,
        max_drug_results: int | None = None,
    ) -> list[SearchResult]:
        drug_embedding_query_text = build_drug_embedding_query_text(drug_name, drug_class)
        max_drug_results = max_drug_results or self.milvus_provider.config.search.top_k
        if not (drug_name and drug_name.strip()):
            return await self.milvus_provider.search_by_text(
                drug_embedding_query_text, limit=max_drug_results
            )

        drug_query_embedding = await self.milvus_provider.embeddings.create_text_embedding(
            drug_embedding_query_text
        )
        milvus_client = await self.milvus_provider.get_client()
        return await asyncio.to_thread(
            self._run_dense_and_bm25_drug_search,
            milvus_client,
            drug_query_embedding,
            drug_name.strip(),
            max_drug_results,
        )

    def _run_dense_and_bm25_drug_search(
        self,
        milvus_client,
        drug_query_embedding: list[float],
        drug_name: str,
        max_drug_results: int,
    ) -> list[SearchResult]:
        milvus_search_config = self.milvus_provider.config.search
        candidate_count_per_search = max(
            max_drug_results * HYBRID_SEARCH_CANDIDATE_MULTIPLIER,
            MIN_CANDIDATES_PER_SEARCH,
        )
        milvus_search_requests = [
            AnnSearchRequest(
                data=[drug_query_embedding],
                anns_field=DRUG_EMBEDDING_FIELD,
                param={
                    "metric_type": DRUG_EMBEDDING_METRIC,
                    "params": milvus_search_config.search_params,
                },
                limit=candidate_count_per_search,
            ),
            AnnSearchRequest(
                data=[drug_name],
                anns_field=DRUG_NAME_SPARSE_FIELD,
                param={"metric_type": "BM25"},
                limit=candidate_count_per_search,
            ),
        ]
        milvus_hybrid_search_results = milvus_client.hybrid_search(
            collection_name=self.milvus_provider.collection_name,
            reqs=milvus_search_requests,
            ranker=RRFRanker(self.rrf_rank_constant),
            limit=max_drug_results,
            output_fields=milvus_search_config.output_fields,
        )
        return [
            SearchResult(
                id=milvus_hit["id"],
                score=float(milvus_hit["distance"]),
                fields={
                    output_field_name: milvus_hit["entity"].get(output_field_name)
                    for output_field_name in milvus_search_config.output_fields
                },
                metadata={"collection": self.milvus_provider.collection_name, "ranker": "rrf"},
            )
            for milvus_hit in milvus_hybrid_search_results[0]
        ]
