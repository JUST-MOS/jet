"""
Tests for jet.spec: parameter and data-vector specifications.
"""

from __future__ import annotations

import unittest

import numpy as np

from jet.spec import DataVectorSpec, Param, ParameterSpec


class TestParam(unittest.TestCase):
    """Construction and serialisation of a single parameter."""

    def setUp(self) -> None:
        self.param = Param("ns", bounds=(0.92, 1.00), block="cosmo", doc="spectral index")

    def tearDown(self) -> None:
        del self.param

    def test_attributes(self) -> None:
        self.assertEqual(self.param.name, "ns")
        self.assertEqual(self.param.bounds, (0.92, 1.00))
        self.assertEqual(self.param.block, "cosmo")
        self.assertEqual(self.param.doc, "spectral index")

    def test_width(self) -> None:
        self.assertAlmostEqual(self.param.width, 0.08)

    def test_unbounded_width_is_none(self) -> None:
        self.assertIsNone(Param("free").width)

    def test_rejects_empty_name(self) -> None:
        with self.assertRaises(ValueError):
            Param("")

    def test_rejects_inverted_bounds(self) -> None:
        with self.assertRaises(ValueError):
            Param("bad", bounds=(1.0, 0.0))

    def test_rejects_degenerate_bounds(self) -> None:
        with self.assertRaises(ValueError):
            Param("bad", bounds=(0.5, 0.5))

    def test_rejects_malformed_bounds(self) -> None:
        with self.assertRaises(ValueError):
            Param("bad", bounds=(1.0,))

    def test_dict_roundtrip(self) -> None:
        self.assertEqual(Param.from_dict(self.param.to_dict()), self.param)

    def test_dict_roundtrip_unbounded(self) -> None:
        param = Param("free", block="hod")
        self.assertEqual(Param.from_dict(param.to_dict()), param)
        self.assertIsNone(Param.from_dict(param.to_dict()).bounds)

    def test_equality_is_by_content(self) -> None:
        self.assertEqual(
            self.param, Param("ns", bounds=(0.92, 1.00), block="cosmo", doc="spectral index")
        )
        self.assertNotEqual(self.param, Param("ns", bounds=(0.90, 1.00), block="cosmo"))


class TestParameterSpec(unittest.TestCase):
    """Column ordering, lookup, slicing, bounds checking and hashing."""

    def setUp(self) -> None:
        self.spec = ParameterSpec(
            [
                Param("Omegab", bounds=(0.04, 0.06), block="cosmo"),
                Param("ns", bounds=(0.92, 1.00), block="cosmo"),
                Param("logMcut", bounds=(12.0, 13.8), block="hod"),
            ]
        )

    def tearDown(self) -> None:
        del self.spec

    def test_dim_and_names(self) -> None:
        self.assertEqual(self.spec.dim, 3)
        self.assertEqual(self.spec.names, ("Omegab", "ns", "logMcut"))

    def test_container_protocol(self) -> None:
        self.assertEqual(len(self.spec), 3)
        self.assertEqual(self.spec[0].name, "Omegab")
        self.assertIn("ns", self.spec)
        self.assertNotIn("sigma8", self.spec)
        self.assertEqual([p.name for p in self.spec], ["Omegab", "ns", "logMcut"])

    def test_index(self) -> None:
        self.assertEqual(self.spec.index("logMcut"), 2)
        self.assertEqual(self.spec.indices(["logMcut", "Omegab"]), [2, 0])

    def test_index_unknown_lists_known_names(self) -> None:
        with self.assertRaises(KeyError) as caught:
            self.spec.index("sigma8")
        self.assertIn("Omegab", str(caught.exception))

    def test_rejects_empty(self) -> None:
        with self.assertRaises(ValueError):
            ParameterSpec([])

    def test_rejects_duplicate_names(self) -> None:
        with self.assertRaises(ValueError) as caught:
            ParameterSpec([Param("ns"), Param("ns")])
        self.assertIn("duplicate", str(caught.exception))

    def test_rejects_non_param(self) -> None:
        with self.assertRaises(TypeError):
            ParameterSpec(["ns"])  # type: ignore[list-item]

    def test_blocks(self) -> None:
        self.assertEqual(self.spec.blocks, ("cosmo", "cosmo", "hod"))

    def test_subset_by_block_preserves_order(self) -> None:
        cosmo = self.spec.subset(block="cosmo")
        self.assertEqual(cosmo.names, ("Omegab", "ns"))

    def test_subset_by_names_preserves_spec_order(self) -> None:
        # Requested order is reversed; the spec's own order must win, so that
        # slicing a value array with `indices` stays consistent.
        subset = self.spec.subset(names=["logMcut", "Omegab"])
        self.assertEqual(subset.names, ("Omegab", "logMcut"))
        self.assertEqual(self.spec.indices(subset.names), [0, 2])

    def test_subset_requires_exactly_one_selector(self) -> None:
        with self.assertRaises(ValueError):
            self.spec.subset()
        with self.assertRaises(ValueError):
            self.spec.subset(names=["ns"], block="cosmo")

    def test_subset_unknown_name(self) -> None:
        with self.assertRaises(KeyError):
            self.spec.subset(names=["nope"])

    def test_subset_empty_selection(self) -> None:
        with self.assertRaises(ValueError):
            self.spec.subset(block="nothing")

    def test_has_bounds(self) -> None:
        self.assertTrue(self.spec.has_bounds)
        self.assertFalse(ParameterSpec([Param("free")]).has_bounds)

    def test_bounds_array(self) -> None:
        np.testing.assert_allclose(
            self.spec.bounds_array(), [[0.04, 0.06], [0.92, 1.00], [12.0, 13.8]]
        )

    def test_bounds_array_requires_all_bounds(self) -> None:
        with self.assertRaises(ValueError):
            ParameterSpec([Param("free")]).bounds_array()

    def test_check_within_bounds_accepts_interior(self) -> None:
        X = np.array([[0.05, 0.95, 13.0]])
        self.assertEqual(self.spec.check_within_bounds(X), [])

    def test_check_within_bounds_reports_each_violation(self) -> None:
        X = np.array([[0.05, 0.95, 13.0], [0.07, 0.95, 13.0]])
        violations = self.spec.check_within_bounds(X)
        self.assertEqual(len(violations), 1)
        name, row, value = violations[0]
        self.assertEqual((name, row), ("Omegab", 1))
        self.assertAlmostEqual(value, 0.07)

    def test_check_within_bounds_slack(self) -> None:
        # 0.061 sits 0.001 beyond the (0.04, 0.06) bound, which is 5 per cent
        # of that parameter's 0.02-wide range.
        X = np.array([[0.061, 0.95, 13.0]])
        self.assertEqual(len(self.spec.check_within_bounds(X, slack=0.0)), 1)
        self.assertEqual(len(self.spec.check_within_bounds(X, slack=0.04)), 1)
        self.assertEqual(self.spec.check_within_bounds(X, slack=0.1), [])

    def test_check_within_bounds_rejects_wrong_width(self) -> None:
        with self.assertRaises(ValueError):
            self.spec.check_within_bounds(np.zeros((2, 5)))

    def test_check_within_bounds_without_bounds_is_empty(self) -> None:
        spec = ParameterSpec([Param("free")])
        self.assertEqual(spec.check_within_bounds(np.zeros((2, 1))), [])

    def test_dict_roundtrip(self) -> None:
        self.assertEqual(ParameterSpec.from_dict(self.spec.to_dict()), self.spec)

    def test_hash_is_content_addressed(self) -> None:
        same = ParameterSpec(
            [
                Param("Omegab", bounds=(0.04, 0.06), block="cosmo"),
                Param("ns", bounds=(0.92, 1.00), block="cosmo"),
                Param("logMcut", bounds=(12.0, 13.8), block="hod"),
            ]
        )
        self.assertEqual(self.spec.hash(), same.hash())

    def test_hash_changes_with_order(self) -> None:
        swapped = ParameterSpec(list(reversed(list(self.spec))))
        self.assertNotEqual(self.spec.hash(), swapped.hash())

    def test_hash_changes_with_bounds(self) -> None:
        changed = ParameterSpec(
            [
                Param("Omegab", bounds=(0.03, 0.06), block="cosmo"),
                Param("ns", bounds=(0.92, 1.00), block="cosmo"),
                Param("logMcut", bounds=(12.0, 13.8), block="hod"),
            ]
        )
        self.assertNotEqual(self.spec.hash(), changed.hash())

    def test_hash_survives_dict_roundtrip(self) -> None:
        self.assertEqual(self.spec.hash(), ParameterSpec.from_dict(self.spec.to_dict()).hash())


class TestDataVectorSpec(unittest.TestCase):
    """Data-vector description: width, log flag, binning metadata."""

    def setUp(self) -> None:
        self.spec = DataVectorSpec("wp", n_bins=15, log=True)

    def tearDown(self) -> None:
        del self.spec

    def test_dim_and_label(self) -> None:
        self.assertEqual(self.spec.dim, 15)
        self.assertEqual(self.spec.name, "wp")
        self.assertEqual(self.spec.axis_label, "log10(wp)")

    def test_axis_label_without_log(self) -> None:
        self.assertEqual(DataVectorSpec("ng", n_bins=1).axis_label, "ng")

    def test_rejects_bad_n_bins(self) -> None:
        with self.assertRaises(ValueError):
            DataVectorSpec("wp", n_bins=0)
        with self.assertRaises(ValueError):
            DataVectorSpec("wp", n_bins=2.5)  # type: ignore[arg-type]

    def test_rejects_empty_name(self) -> None:
        with self.assertRaises(ValueError):
            DataVectorSpec("", n_bins=3)

    def test_bin_edges_must_have_n_plus_one_entries(self) -> None:
        DataVectorSpec("wp", n_bins=3, bin_edges=[0.0, 1.0, 2.0, 3.0])
        with self.assertRaises(ValueError):
            DataVectorSpec("wp", n_bins=3, bin_edges=[0.0, 1.0, 2.0])

    def test_labels_must_match_n_bins(self) -> None:
        DataVectorSpec("wp", n_bins=2, labels=["a", "b"])
        with self.assertRaises(ValueError):
            DataVectorSpec("wp", n_bins=3, labels=["a", "b"])

    def test_dict_roundtrip_preserves_edges(self) -> None:
        spec = DataVectorSpec(
            "wp", n_bins=3, log=True, bin_edges=[0.1, 0.5, 1.0, 5.0], labels=["a", "b", "c"]
        )
        restored = DataVectorSpec.from_dict(spec.to_dict())
        self.assertEqual(restored, spec)
        np.testing.assert_allclose(restored.bin_edges, [0.1, 0.5, 1.0, 5.0])
        self.assertEqual(restored.labels, ("a", "b", "c"))

    def test_hash_distinguishes_log_flag(self) -> None:
        self.assertNotEqual(
            DataVectorSpec("wp", n_bins=15, log=True).hash(),
            DataVectorSpec("wp", n_bins=15, log=False).hash(),
        )

    def test_hash_distinguishes_width(self) -> None:
        self.assertNotEqual(
            DataVectorSpec("wp", n_bins=15).hash(),
            DataVectorSpec("wp", n_bins=10).hash(),
        )


if __name__ == "__main__":
    unittest.main()
