"""The `PauliString` class and `PauliSum`'s arithmetic operators.

Covers the five constructors, the `p()` shorthand, printing and equality, the
commute predicates, and the three product methods — plus `+`/`-`/`*` and their
in-place forms on `PauliSum`, which merge matching strings rather than
overwriting them.
"""

import pytest

from paulistrings import PauliString, PauliSum, p


# ---- constructors ----


def test_identity_is_all_i():
    string = PauliString.identity(3)
    assert string.label == "III"
    assert string.num_qubits == 3
    assert string.weight == 0


def test_single_site_constructors():
    assert PauliString.x(0, 3).label == "XII"
    assert PauliString.y(1, 3).label == "IYI"
    assert PauliString.z(2, 3).label == "IIZ"
    assert PauliString.x(0, 3).weight == 1


def test_num_qubits_is_positional_or_keyword():
    assert PauliString.x(0, num_qubits=3) == PauliString.x(0, 3)
    assert PauliString.identity(num_qubits=2).label == "II"


def test_from_label_round_trips():
    for label in ("IXYZ", "YYYY", "I", ""):
        assert PauliString.from_label(label).label == label
        assert PauliString.from_label(label).num_qubits == len(label)


def test_p_is_from_label():
    assert p("XYZ") == PauliString.from_label("XYZ")
    assert p("XYZ").weight == 3


def test_from_label_rejects_a_character_outside_ixyz():
    with pytest.raises(ValueError, match="Pauli character"):
        p("XQZ")


@pytest.mark.parametrize("ctor", [PauliString.x, PauliString.y, PauliString.z])
def test_qubit_index_out_of_range_is_a_value_error(ctor):
    with pytest.raises(ValueError, match="out of range"):
        ctor(3, 3)
    with pytest.raises(ValueError, match="out of range"):
        ctor(0, 0)


def test_crosses_a_word_boundary():
    # Qubit 64 lives in the second word, i.e. the W=2 dispatch band.
    string = PauliString.x(64, 100)
    assert string.label == "I" * 64 + "X" + "I" * 35
    assert string.weight == 1


# ---- printing, equality, hashing ----


def test_str_and_repr_are_both_the_label():
    string = p("XYZ")
    assert str(string) == "XYZ"
    assert repr(string) == "XYZ"


def test_equality_is_label_and_num_qubits():
    assert p("XII") == PauliString.x(0, 3)
    assert p("XII") != p("XI")
    assert p("XII") != p("YII")
    assert p("XII") != "XII"


def test_hashable_by_value():
    assert len({p("XYZ"), PauliString.from_label("XYZ"), p("ZYX")}) == 2


# ---- single-string operations ----


def test_x_and_z_anticommute_on_the_same_qubit():
    x, z = PauliString.x(0, 1), PauliString.z(0, 1)
    assert x.commutes_with(z) is False
    assert x.anticommutes_with(z) is True


def test_disjoint_support_commutes():
    x0, x1 = PauliString.x(0, 2), PauliString.x(1, 2)
    assert x0.commutes_with(x1) is True
    assert x0.anticommutes_with(x1) is False


def test_anticommutes_with_is_the_exact_complement():
    strings = [p(label) for label in ("II", "XI", "YI", "ZI", "XY", "ZZ", "YZ")]
    for a in strings:
        for b in strings:
            assert a.commutes_with(b) is not a.anticommutes_with(b)


def test_mul_is_x_z_equals_minus_i_y():
    coeff, product = PauliString.x(0, 1).mul(PauliString.z(0, 1))
    assert coeff == -1j
    assert product == p("Y")


def test_mul_composes_disjoint_sites():
    coeff, product = PauliString.x(0, 3).mul(PauliString.y(1, 3))
    assert coeff == 1 + 0j
    assert product == p("XYI")


def test_commutator_and_anticommutator_of_anticommuting_strings():
    x, z = PauliString.x(0, 1), PauliString.z(0, 1)
    # [X, Z] = 2 X Z = -2i Y, and {X, Z} = 0.
    assert x.commutator(z) == (-2j, p("Y"))
    assert x.anticommutator(z) == (0j, p("Y"))


def test_commutator_and_anticommutator_of_commuting_strings():
    x0, x1 = PauliString.x(0, 2), PauliString.x(1, 2)
    # [X⊗I, I⊗X] = 0, and {X⊗I, I⊗X} = 2 X⊗X.
    assert x0.commutator(x1) == (0j, p("XX"))
    assert x0.anticommutator(x1) == (2 + 0j, p("XX"))


def test_brackets_sum_to_twice_the_product():
    a, b = p("XYZ"), p("ZZX")
    coeff, product = a.mul(b)
    comm_coeff, comm_product = a.commutator(b)
    anti_coeff, anti_product = a.anticommutator(b)
    assert comm_coeff + anti_coeff == 2 * coeff
    assert comm_product == product == anti_product


def test_binary_methods_reject_a_qubit_count_mismatch():
    for method in ("commutes_with", "anticommutes_with", "mul", "commutator", "anticommutator"):
        with pytest.raises(ValueError, match="num_qubits mismatch"):
            getattr(p("XY"), method)(p("XYZ"))


# ---- PauliSum arithmetic ----


def test_add_combines_matching_strings_and_keeps_the_rest():
    a = PauliSum.from_strings({"XI": 1.0, "ZI": 0.5}, num_qubits=2)
    b = PauliSum.from_strings({"XI": 2.0, "IZ": 1.0}, num_qubits=2)
    total = a + b
    assert len(total) == 3
    assert total.overlap(PauliSum.from_strings({"XI": 1.0}, num_qubits=2)) == 3 + 0j
    # The operands are untouched.
    assert len(a) == 2 and len(b) == 2


def test_sub_cancels_exactly():
    a = PauliSum.from_strings({"XI": 1.0, "ZI": 0.5}, num_qubits=2)
    assert len(a - a) == 0
    difference = a - PauliSum.from_strings({"ZI": 0.5}, num_qubits=2)
    assert len(difference) == 1
    assert difference.coefficients() == [1 + 0j]


def test_iadd_and_isub_are_in_place():
    total = PauliSum.from_strings({"XI": 1.0}, num_qubits=2)
    total += PauliSum.from_strings({"XI": 1.0, "IZ": 1.0}, num_qubits=2)
    assert len(total) == 2
    total -= PauliSum.from_strings({"IZ": 1.0}, num_qubits=2)
    assert len(total) == 1
    assert total.coefficients() == [2 + 0j]


def test_in_place_ops_accept_the_same_object_on_both_sides():
    total = PauliSum.from_strings({"XI": 1.5}, num_qubits=2)
    total += total
    assert total.coefficients() == [3 + 0j]
    total -= total
    assert len(total) == 0


def test_mul_scales_every_coefficient():
    sum_ = PauliSum.from_strings({"XI": 1.0, "IZ": -2.0}, num_qubits=2)
    assert sorted(c.real for c in (sum_ * 2.0).coefficients()) == [-4.0, 2.0]
    assert sorted(c.real for c in (2 * sum_).coefficients()) == [-4.0, 2.0]
    # Canonical order is lexicographic on (x, z), so "IZ" precedes "XI".
    assert (sum_ * 1j).coefficients() == [-2j, 1j]


def test_imul_is_in_place_and_zero_empties_the_sum():
    sum_ = PauliSum.from_strings({"XI": 1.0}, num_qubits=2)
    sum_ *= 3.0
    assert sum_.coefficients() == [3 + 0j]
    sum_ *= 0.0
    assert len(sum_) == 0


def test_multiplying_two_sums_is_a_type_error():
    sum_ = PauliSum.from_strings({"XI": 1.0}, num_qubits=2)
    with pytest.raises(TypeError, match="full operator product"):
        sum_ * sum_
    with pytest.raises(TypeError, match="complex or real number"):
        sum_ * "two"


def test_arithmetic_rejects_a_qubit_count_mismatch():
    a = PauliSum.from_strings({"XI": 1.0}, num_qubits=2)
    b = PauliSum.from_strings({"XII": 1.0}, num_qubits=3)
    with pytest.raises(ValueError, match="num_qubits mismatch"):
        a + b
    with pytest.raises(ValueError, match="num_qubits mismatch"):
        a - b


def test_arithmetic_on_the_second_width_tier():
    # 70 qubits is the W=2 band: the merge and the scale must dispatch there too.
    wide = "I" * 69 + "Z"
    a = PauliSum.from_strings({wide: 1.0}, num_qubits=70)
    total = a + a
    total *= 0.5
    assert len(total) == 1
    assert total.coefficients() == [1 + 0j]
    assert PauliString.z(69, 70).label == wide


def test_adding_a_non_sum_is_a_type_error():
    a = PauliSum.from_strings({"XI": 1.0}, num_qubits=2)
    with pytest.raises(TypeError):
        a + 1.0


# ---- pretty-printing ----


def test_empty_sum_prints_as_zero():
    assert str(PauliSum(4)) == "0"
    assert repr(PauliSum(4)) == "0"


def test_str_and_repr_show_coefficient_and_label():
    # Canonical order is lexicographic on (x, z), so "IZ" (x=0) precedes "XI" (x=1).
    sum_ = PauliSum.from_strings({"XI": 0.5, "IZ": -1.0}, num_qubits=2)
    assert str(sum_) == "-1*IZ + 0.5*XI"
    assert repr(sum_) == str(sum_)


def test_a_real_coefficient_drops_the_imaginary_part_but_a_complex_one_does_not():
    real = PauliSum.from_strings({"X": 0.25}, num_qubits=1)
    assert str(real) == "0.25*X"
    complex_ = PauliSum.from_strings({"X": 0.25 + 0.5j}, num_qubits=1)
    assert str(complex_) == "(0.25+0.5j)*X"
    negative_imag = PauliSum.from_strings({"X": 0.25 - 0.5j}, num_qubits=1)
    assert str(negative_imag) == "(0.25-0.5j)*X"


def test_a_sum_past_the_preview_count_is_truncated_with_a_count():
    terms = {("I" * i + "X" + "I" * (5 - i)): 1.0 for i in range(6)}
    sum_ = PauliSum.from_strings(terms, num_qubits=6)
    text = str(sum_)
    assert text.endswith("... (2 more terms)")
    assert text.count("*") == 4


def test_preview_is_prefix_order_not_sorted_by_magnitude():
    # A huge first coefficient must still show first, in storage order --
    # sorting by magnitude just to print a preview would cost O(N log N).
    sum_ = PauliSum.from_strings({"II": 1000.0, "XX": 1.0, "IX": 1.0, "XI": 1.0, "YY": 1.0}, num_qubits=2)
    assert str(sum_).startswith("1000*II")


# ---- from_strings: inference and the (labels, coefficients) form ----


def test_num_qubits_is_inferred_from_a_dict():
    sum_ = PauliSum.from_strings({"XII": 1.0, "IXI": 1.0})
    assert sum_.num_qubits == 3
    assert len(sum_) == 2


def test_num_qubits_is_inferred_from_labels_and_coefficients():
    sum_ = PauliSum.from_strings(["XII", "IXI"], [1.0, 2.0])
    assert sum_.num_qubits == 3
    assert sorted(c.real for c in sum_.coefficients()) == [1.0, 2.0]


def test_explicit_num_qubits_still_works_for_both_forms():
    a = PauliSum.from_strings({"XI": 1.0}, num_qubits=2)
    b = PauliSum.from_strings(["XI"], [1.0], num_qubits=2)
    assert a.num_qubits == b.num_qubits == 2


def test_a_repeated_label_in_the_list_form_accumulates():
    sum_ = PauliSum.from_strings(["XI", "XI", "IZ"], [1.0, 2.0, 5.0])
    assert len(sum_) == 2
    assert sorted(c.real for c in sum_.coefficients()) == [3.0, 5.0]


def test_mismatched_label_and_coefficient_counts_is_a_value_error():
    with pytest.raises(ValueError, match="labels but"):
        PauliSum.from_strings(["XI"], [1.0, 2.0])


def test_inferring_num_qubits_from_nothing_is_a_value_error():
    with pytest.raises(ValueError, match="cannot infer num_qubits"):
        PauliSum.from_strings({})
    with pytest.raises(ValueError, match="cannot infer num_qubits"):
        PauliSum.from_strings([], [])


def test_a_dict_with_coefficients_also_given_is_a_type_error():
    with pytest.raises(TypeError, match="not a dict with coefficients"):
        PauliSum.from_strings({"XI": 1.0}, [1.0])
