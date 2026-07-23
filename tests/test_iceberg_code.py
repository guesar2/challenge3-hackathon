"""
test_iceberg_code.py

Tier-0 (free, offline, no qnexus) validation of iceberg_code.py's stabilizer
and logical-operator algebra, against the Iceberg paper's Eqs. (1)-(12) and
Chao & Reichardt's independent [[n, n-2, 2]] derivation.

Important subtlety caught while writing these tests: Eqs. (1), (2), (5),
(6), (9) (single- and two-qubit compiled identities) are UNCONDITIONAL
matrix identities over the full 2^n Hilbert space -- e.g. Xbar_i Xbar_j =
(X_i X_t)(X_j X_t) = X_i X_j exactly, no cancellation needed beyond X_t^2=I.
But the GLOBAL identities (Eqs. 3, 4, 7, 8, and by extension 10-12), like
"the product of all Xbar_j equals X_t X_b", only hold *within the code
space*: e.g. prod_j Xbar_j = (prod_j X_j) X_t^k = prod_j X_j (k even), and
prod_j X_j = X_t X_b only once you use S_X = X_1...X_k X_t X_b = +1, which
is true on the code space, not as an unconditional operator identity on
the full Hilbert space. This is fine for compiling logical circuits (an
encoded computation never leaves the code space), but it means the global
identities are tested here as projected identities (P_code @ LHS @ P_code
== P_code @ RHS @ P_code), not full-Hilbert-space matrix equality.
"""
import numpy as np
import pytest

from iceberg_code import (
    b_index,
    code_space_projector,
    ghz_all_zero_statevector,
    logical_x,
    logical_y,
    logical_z,
    physical_pauli_string,
    stabilizer_sx,
    stabilizer_sz,
    t_index,
    validate_k,
)

K_VALUES = (2, 4, 6)


def commutator(a, b):
    return (a @ b - b @ a).toarray()


def anticommutator(a, b):
    return (a @ b + b @ a).toarray()


def is_zero(m, atol=1e-9):
    return np.allclose(m, 0, atol=atol)


@pytest.mark.parametrize("k", K_VALUES)
def test_validate_k_rejects_odd_and_small(k):
    assert validate_k(k) == k + 2
    with pytest.raises(ValueError):
        validate_k(k + 1)  # odd
    with pytest.raises(ValueError):
        validate_k(0)


@pytest.mark.parametrize("k", K_VALUES)
def test_t_b_indices_distinct_and_in_range(k):
    n = validate_k(k)
    assert t_index(k) == k
    assert b_index(k) == k + 1
    assert t_index(k) != b_index(k)
    assert 0 <= t_index(k) < n
    assert 0 <= b_index(k) < n


@pytest.mark.parametrize("k", K_VALUES)
def test_stabilizers_commute_are_hermitian_involutions(k):
    sx, sz = stabilizer_sx(k), stabilizer_sz(k)
    for s in (sx, sz):
        assert is_zero((s @ s).toarray() - np.eye(s.shape[0]))  # involution
        assert is_zero((s - s.conj().T).toarray())  # Hermitian
    # S_X, S_Z commute because n = k+2 is even (each of the n qubit-wise
    # X_i,Z_i pairs anticommutes, contributing (-1) each; an even number
    # of them cancels overall)
    assert is_zero(commutator(sx, sz))


@pytest.mark.parametrize("k", K_VALUES)
@pytest.mark.parametrize("i", (0,))
def test_logical_x_z_anticommute_on_same_qubit(k, i):
    assert is_zero(anticommutator(logical_x(k, i), logical_z(k, i)))


@pytest.mark.parametrize("k", (4, 6))
def test_logical_ops_commute_across_different_qubits(k):
    for i in range(k):
        for j in range(k):
            if i == j:
                continue
            assert is_zero(commutator(logical_x(k, i), logical_x(k, j)))
            assert is_zero(commutator(logical_z(k, i), logical_z(k, j)))
            assert is_zero(commutator(logical_x(k, i), logical_z(k, j)))


@pytest.mark.parametrize("k", K_VALUES)
def test_logical_ops_commute_with_stabilizers(k):
    sx, sz = stabilizer_sx(k), stabilizer_sz(k)
    for i in range(k):
        for s in (sx, sz):
            assert is_zero(commutator(logical_x(k, i), s))
            assert is_zero(commutator(logical_z(k, i), s))


@pytest.mark.parametrize("k", (4, 6))
def test_eq2_xbar_i_xbar_j_equals_physical_xi_xj(k):
    """Eq. (2): Xbar_i Xbar_j = X_i X_j -- unconditional identity."""
    for i in range(k):
        for j in range(k):
            if i == j:
                continue
            lhs = logical_x(k, i) @ logical_x(k, j)
            rhs = physical_pauli_string(k, {i: 'x', j: 'x'})
            assert is_zero((lhs - rhs).toarray())


@pytest.mark.parametrize("k", (4, 6))
def test_eq6_zbar_i_zbar_j_equals_physical_zi_zj(k):
    """Eq. (6): Zbar_i Zbar_j = Z_i Z_j -- unconditional identity."""
    for i in range(k):
        for j in range(k):
            if i == j:
                continue
            lhs = logical_z(k, i) @ logical_z(k, j)
            rhs = physical_pauli_string(k, {i: 'z', j: 'z'})
            assert is_zero((lhs - rhs).toarray())


@pytest.mark.parametrize("k", (4, 6))
def test_eq9_ybar_i_ybar_j_equals_physical_yi_yj(k):
    """Eq. (9): Ybar_i Ybar_j = Y_i Y_j -- unconditional identity."""
    for i in range(k):
        for j in range(k):
            if i == j:
                continue
            lhs = logical_y(k, i) @ logical_y(k, j)
            rhs = physical_pauli_string(k, {i: 'y', j: 'y'})
            assert is_zero((lhs - rhs).toarray())


@pytest.mark.parametrize("k", (2, 4))
def test_logical_y_is_a_valid_pauli(k):
    """Ybar_i should behave like a genuine logical Pauli: square to
    identity, anticommute with Xbar_i/Zbar_i, commute with everything on
    other logical qubits and with the stabilizers."""
    sx, sz = stabilizer_sx(k), stabilizer_sz(k)
    for i in range(k):
        yi = logical_y(k, i)
        assert is_zero((yi @ yi).toarray() - np.eye(yi.shape[0]))
        assert is_zero(anticommutator(yi, logical_x(k, i)))
        assert is_zero(anticommutator(yi, logical_z(k, i)))
        assert is_zero(commutator(yi, sx))
        assert is_zero(commutator(yi, sz))
        for j in range(k):
            if j == i:
                continue
            assert is_zero(commutator(yi, logical_x(k, j)))
            assert is_zero(commutator(yi, logical_z(k, j)))


@pytest.mark.parametrize("k", (2, 4, 6))
def test_eq4_global_x_product_equals_xt_xb_on_code_space(k):
    """Eq. (4): (ox_j Xbar_j) = X_t X_b -- holds only on the code space
    (see module docstring); verified here via the code-space projector."""
    proj = code_space_projector(k)
    lhs = proj
    for j in range(k):
        lhs = lhs @ logical_x(k, j)
    rhs = proj @ physical_pauli_string(k, {t_index(k): 'x', b_index(k): 'x'})
    assert is_zero((lhs - rhs).toarray())


@pytest.mark.parametrize("k", (2, 4, 6))
def test_eq8_global_z_product_equals_zt_zb_on_code_space(k):
    """Eq. (8): (ox_j Zbar_j) = Z_t Z_b -- code-space identity."""
    proj = code_space_projector(k)
    lhs = proj
    for j in range(k):
        lhs = lhs @ logical_z(k, j)
    rhs = proj @ physical_pauli_string(k, {t_index(k): 'z', b_index(k): 'z'})
    assert is_zero((lhs - rhs).toarray())


@pytest.mark.parametrize("k", (4, 6))
def test_eq3_global_x_except_i_equals_xb_xi_on_code_space(k):
    """Eq. (3): (ox_{j != i} Xbar_j) = X_b X_i -- code-space identity."""
    proj = code_space_projector(k)
    for i in range(k):
        lhs = proj
        for j in range(k):
            if j == i:
                continue
            lhs = lhs @ logical_x(k, j)
        rhs = proj @ physical_pauli_string(k, {b_index(k): 'x', i: 'x'})
        assert is_zero((lhs - rhs).toarray())


@pytest.mark.parametrize("k", (4, 6))
def test_eq7_global_z_except_i_equals_zt_zi_on_code_space(k):
    """Eq. (7): (ox_{j != i} Zbar_j) = Z_t Z_i -- code-space identity."""
    proj = code_space_projector(k)
    for i in range(k):
        lhs = proj
        for j in range(k):
            if j == i:
                continue
            lhs = lhs @ logical_z(k, j)
        rhs = proj @ physical_pauli_string(k, {t_index(k): 'z', i: 'z'})
        assert is_zero((lhs - rhs).toarray())


@pytest.mark.parametrize("k", K_VALUES)
def test_ghz_all_zero_state_is_plus_one_stabilizer_eigenstate(k):
    sv = ghz_all_zero_statevector(k)
    assert np.isclose(np.linalg.norm(sv), 1.0)
    for s in (stabilizer_sx(k), stabilizer_sz(k)):
        out = s @ sv
        assert np.allclose(out, sv, atol=1e-9)


@pytest.mark.parametrize("k", K_VALUES)
def test_ghz_all_zero_state_is_logical_zero_for_every_i(k):
    """Zbar_i |0bar> = +|0bar> for every logical qubit i, by definition of
    the encoded all-zero state."""
    sv = ghz_all_zero_statevector(k)
    for i in range(k):
        out = logical_z(k, i) @ sv
        assert np.allclose(out, sv, atol=1e-9)
