# pylint: disable=no-name-in-module
""" Dataclasses for Request/Response """

from dataclasses import dataclass, field
from typing import List, Union
from pydantic import BaseModel


@dataclass()
class AppRequest:
    """ Dataclass for http-request """
    features: List[str] = field(default=list)
    data: List[List[Union[int, float]]] = field(default=list)


class AppResponse(BaseModel):
    """ Dataclass (pydantic) for http-response """
    predict: List[int]
