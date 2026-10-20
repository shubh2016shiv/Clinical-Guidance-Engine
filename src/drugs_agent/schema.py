"""The structured answer the drug agent returns."""

from pydantic import BaseModel, ConfigDict


class DrugMatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    drug_name: str
    drug_class: str | None
    drug_sub_class: str | None
    therapeutic_category: str | None
    route_of_administration: str | None
    formulation: str | None
    dosage_strengths: list[str]


class DrugAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    found: bool
    matches: list[DrugMatch]
