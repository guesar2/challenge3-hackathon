"""
fh_shot_observables.py

Convert Z-basis measurement bitstrings (from fh_local_sampler, or the real H2
backends) into the Fermi-Hubbard observables the challenge asks for:
per-site particle density, per-site double occupancy, average double occupancy,
and staggered magnetization. All of these are Z-diagonal, so a single Z-basis
shot set is sufficient.

Bit convention: bitstring[q] is the measured value of qubit q (0/1), qubit 0
first -- matching fh_local_sampler and fh_lattice. Occupation of orbital q is
exactly its bit (|1> = occupied), so density/doublon/magnetization are simple
per-shot averages. Bootstrap resampling gives shot-noise standard errors.
"""
from __future__ import annotations

import numpy as np

from fh_lattice import HubbardLattice, UP, DOWN


def _occ_array(bitstrings, n_qubits):
    """(n_shots, n_qubits) integer occupation array from bitstrings."""
    arr = np.array([[int(b) for b in s] for s in bitstrings], dtype=float)
    if arr.shape[1] != n_qubits:
        raise ValueError(f"bitstring length {arr.shape[1]} != n_qubits {n_qubits}")
    return arr


def _observables_from_occ(occ, lat: HubbardLattice):
    """Compute the observable dict from an (n_shots, n_qubits) occupation array."""
    up_idx = [lat.qubit(s, UP) for s in lat.sites]
    dn_idx = [lat.qubit(s, DOWN) for s in lat.sites]

    n_up = occ[:, up_idx]           # (shots, n_sites)
    n_dn = occ[:, dn_idx]
    density = n_up + n_dn           # per-site density per shot
    doublon = n_up * n_dn           # per-site double occupancy per shot
    sz = 0.5 * (n_up - n_dn)        # per-site S^z per shot

    signs = np.array([(-1) ** (x + y) for (x, y) in lat.sites])
    m_stag_per_shot = (sz * signs).mean(axis=1)   # (shots,)

    return {
        "density_per_site": density.mean(axis=0),          # (n_sites,)
        "double_per_site": doublon.mean(axis=0),
        "avg_double_occupancy": float(doublon.mean()),
        "staggered_magnetization": float(m_stag_per_shot.mean()),
        "total_particles": float(density.sum(axis=1).mean()),
    }


def bitstrings_to_observables(bitstrings, lat: HubbardLattice):
    occ = _occ_array(bitstrings, lat.n_qubits)
    return _observables_from_occ(occ, lat)


def bootstrap_errors(bitstrings, lat: HubbardLattice, n_boot=500, seed=0):
    """Bootstrap standard errors for the scalar observables (avg double
    occupancy, staggered magnetization, total particles)."""
    occ = _occ_array(bitstrings, lat.n_qubits)
    n_shots = occ.shape[0]
    rng = np.random.default_rng(seed)
    d = np.empty(n_boot); m = np.empty(n_boot); nt = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n_shots, size=n_shots)
        o = _observables_from_occ(occ[idx], lat)
        d[b] = o["avg_double_occupancy"]
        m[b] = o["staggered_magnetization"]
        nt[b] = o["total_particles"]
    return {
        "avg_double_occupancy": d.std(ddof=1),
        "staggered_magnetization": m.std(ddof=1),
        "total_particles": nt.std(ddof=1),
    }