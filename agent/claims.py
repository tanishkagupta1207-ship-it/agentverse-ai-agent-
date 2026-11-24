from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, Field, model_validator


class CodeEvidence(BaseModel):
    type: Literal["code"]
    path: str
    start: int
    end: int


class CommitEvidence(BaseModel):
    type: Literal["commit"]
    sha: str


class IssueEvidence(BaseModel):
    type: Literal["issue", "pr"]
    number: int


Evidence = Annotated[
    Union[CodeEvidence, CommitEvidence, IssueEvidence],
    Field(discriminator="type"),
]

Label = Literal["verified", "history_backed", "assumption"]


class Claim(BaseModel):
    id: str
    text: str
    label: Label
    evidence: list[Evidence] = Field(default_factory=list)
    note: Optional[str] = None  # e.g. "partially supported" after verifier downgrade

    @model_validator(mode="after")
    def check_label_matches_evidence(self):
        kinds = {e.type for e in self.evidence}
        if self.label != "assumption" and not self.evidence:
            raise ValueError("a claim with no evidence can only be 'assumption'")
        if self.label == "verified" and "code" not in kinds:
            raise ValueError("'verified' needs at least one code evidence")
        if self.label == "history_backed" and not kinds & {"commit", "issue", "pr"}:
            raise ValueError("'history_backed' needs a commit, issue or pr evidence")
        return self


class AnswerResult(BaseModel):
    claims: list[Claim] = Field(default_factory=list)
    not_found: bool = False  # explicit abstention
    note: Optional[str] = None


if __name__ == "__main__":
    ok = AnswerResult.model_validate({
        "claims": [
            {"id": "c1", "text": "Auth runs before routing.", "label": "verified",
             "evidence": [{"type": "code", "path": "app/auth.py", "start": 42, "end": 58}]},
            {"id": "c2", "text": "Retry added after timeout bugs.", "label": "history_backed",
             "evidence": [{"type": "commit", "sha": "a1b2c3d"}, {"type": "pr", "number": 87}]},
            {"id": "c3", "text": "Probably for rate limiting.", "label": "assumption"},
        ]
    })
    print("valid:", len(ok.claims), "claims")
    try:
        Claim(id="x", text="no proof", label="verified")
    except ValueError as e:
        print("rejected as expected")