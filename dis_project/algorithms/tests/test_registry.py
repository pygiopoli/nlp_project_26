from unittest import TestCase

from algorithms.base import AlgorithmSpec, WSDResult
from algorithms.registry import ALGORITHMS, build_registry, get_algorithm


def dummy_run(sentence: str, target: str) -> WSDResult:
    return WSDResult(algorithm="dummy", target=target, ranking=())


def make_spec(key: str) -> AlgorithmSpec:
    return AlgorithmSpec(key=key, label=key.title(), description="", run=dummy_run)


class BuildRegistryTests(TestCase):
    def test_maps_keys_to_specs_in_order(self) -> None:
        specs = [make_spec("b"), make_spec("a")]
        registry = build_registry(specs)
        self.assertEqual(list(registry), ["b", "a"])
        self.assertIs(registry["a"], specs[1])

    def test_rejects_duplicate_keys(self) -> None:
        with self.assertRaises(ValueError):
            build_registry([make_spec("a"), make_spec("a")])


class RegisteredAlgorithmsTests(TestCase):
    def test_expected_keys_in_order(self) -> None:
        self.assertEqual(list(ALGORITHMS), ["lesk", "lesk_pos", "extended_lesk", "extended_lesk_pos"])


class GetAlgorithmTests(TestCase):
    def test_unknown_key_raises_key_error(self) -> None:
        with self.assertRaises(KeyError):
            get_algorithm("does-not-exist")
