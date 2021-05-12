""" Dataclasses for Request/Response """
from dataclasses import dataclass, field, MISSING
from typing import List, Union
from pydantic import BaseModel


@dataclass()
class AppRequest:
    features: List[str] = field(default=list)
    data: List[List[Union[int, float]]] = field(default=list)


class AppResponse(BaseModel):
    predict: List[int]
