from dataclasses import dataclass
from random import Random
import string
import re

from .perturbation import Perturbation
from .perturbation_description import PerturbationDescription


class MissPuntPerturbation(Perturbation):
    """
    A simple perturbation that replaces existing spaces with 0-max_spaces spaces (thus potentially merging words).
    """

    @dataclass(frozen=True)
    class Description(PerturbationDescription):
        prob: float = 0.0

    name: str = "miss punctuation"

    def __init__(self, prob=0.1):
        self.prob = prob

    @property
    def description(self) -> PerturbationDescription:
        return MissPuntPerturbation.Description(name=self.name, robustness=True, prob=self.prob)

    def perturb(self, text: str, rng: Random) -> str:
        if rng.random() < self.prob:
            return text.translate(str.maketrans('', '', string.punctuation))
        else:
            return text