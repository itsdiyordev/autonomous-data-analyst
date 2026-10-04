from typing import Literal

from pydantic import BaseModel, Field, field_validator


class RegisterInput(BaseModel):
    email: str = Field(min_length=5, max_length=255)
    name: str = Field(min_length=2, max_length=100)
    password: str = Field(min_length=8, max_length=128)

    @field_validator("email")
    @classmethod
    def valid_email(cls, value):
        value = value.strip().lower()
        if "@" not in value or "." not in value.rsplit("@", 1)[1]:
            raise ValueError("Enter a valid email address")
        return value


class LoginInput(BaseModel):
    email: str
    password: str


class RunInput(BaseModel):
    dataset_id: str
    objective: str = Field(min_length=8, max_length=2000)
    target: str | None = Field(default=None, min_length=1, max_length=255)
    task: Literal["auto", "classification", "regression", "clustering"] = "auto"
    test_size: float = Field(default=0.2, ge=0.1, le=0.3)
    budget_seconds: int = Field(default=180, ge=30, le=600)
    split_strategy: Literal["random", "chronological", "group"] = "random"
    split_column: str | None = None
    cv_folds: int = Field(default=3, ge=2, le=5)


class PredictInput(BaseModel):
    records: list[dict] = Field(min_length=1, max_length=1000)


class ChatInput(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
