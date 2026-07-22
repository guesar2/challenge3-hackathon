"""
fh_lattice.py

Geometry for the 2D Fermi-Hubbard model and the Jordan-Wigner (JW) mode
ordering used everywhere else in the project.

A spinful site (x, y) carries two fermionic modes -- one per spin sigma in
{up, down} -- so an Lx-by-Ly lattice needs 2*Lx*Ly qubits under a JW mapping
(this is why 4x4 = 32 qubits does NOT fit Quantinuum's 26-qubit H2 exact
emulator; see README / fh_config.py for the honest size ceiling).

Mode ordering (this is the single source of truth for how sites/spins map to
qubit indices):
  - Sites are visited in a boustrophedon ("snake") order: row 0 left-to-right,
    row 1 right-to-left, and so on. Snake order makes spatially-adjacent sites
    adjacent in the linear order, which keeps Jordan-Wigner Z-strings for
    horizontal hops as short as possible (length 0), and vertical hops one row
    long.
  - All spin-up modes come first (in snake order), then all spin-down modes
    (in snake order). Same-spin hopping therefore stays within one contiguous
    block (short/well-defined Z-strings), while the on-site interaction couples
    mode (i, up) to mode (i, down) -- far apart in the ordering, but the
    interaction is diagonal (only Z's), so no Z-string is needed there anyway.

This module is pure geometry: no qiskit/pytket/scipy dependency, so it is
trivially testable and importable from both the classical and quantum sides.
"""
from __future__ import annotations

UP, DOWN = 0, 1


class HubbardLattice:
    """A 2D rectangular Fermi-Hubbard lattice with a fixed JW mode ordering.

    Parameters
    ----------
    Lx, Ly : int
        Lattice dimensions (Lx columns, Ly rows).
    periodic_x, periodic_y : bool
        Whether nearest-neighbour bonds wrap around each direction. Defaults to
        open boundaries (False) -- the smallest, least error-prone choice for a
        finite ED-verifiable benchmark. (Periodic boundaries reduce finite-size
        effects, cf. the paper's mixed-boundary trick, but add wrap-around bonds
        with longer JW strings.)
    """

    def __init__(self, Lx: int, Ly: int, periodic_x: bool = False, periodic_y: bool = False):
        if Lx < 1 or Ly < 1:
            raise ValueError("Lx, Ly must be >= 1")
        self.Lx = Lx
        self.Ly = Ly
        self.periodic_x = periodic_x
        self.periodic_y = periodic_y

        # Snake ordering of sites -> linear position.
        self._snake = []
        for y in range(Ly):
            xs = range(Lx) if (y % 2 == 0) else range(Lx - 1, -1, -1)
            for x in xs:
                self._snake.append((x, y))
        self._pos = {site: i for i, site in enumerate(self._snake)}

    # ---- basic counts ----
    @property
    def n_sites(self) -> int:
        return self.Lx * self.Ly

    @property
    def n_qubits(self) -> int:
        return 2 * self.n_sites

    @property
    def sites(self):
        """Sites in snake order (list of (x, y))."""
        return list(self._snake)

    # ---- mode / qubit indexing ----
    def site_pos(self, site) -> int:
        """Linear (snake) position of a site."""
        return self._pos[site]

    def qubit(self, site, spin: int) -> int:
        """Qubit index for (site, spin). Up block first, then down block."""
        base = self._pos[site]
        return base if spin == UP else self.n_sites + base

    def all_qubits_for_spin(self, spin: int):
        """List of qubit indices carrying the given spin, in snake order."""
        return [self.qubit(s, spin) for s in self._snake]

    # ---- bonds ----
    def bonds(self):
        """Nearest-neighbour spatial bonds as ((x1,y1),(x2,y2)) pairs.

        Horizontal bonds first, then vertical, each optionally wrapping if the
        corresponding periodic flag is set. Wrap-around bonds are only emitted
        when the lattice is longer than 2 in that direction (for L=2 the wrap
        bond would duplicate the open bond).
        """
        out = []
        # horizontal
        for y in range(self.Ly):
            for x in range(self.Lx - 1):
                out.append(((x, y), (x + 1, y)))
            if self.periodic_x and self.Lx > 2:
                out.append(((self.Lx - 1, y), (0, y)))
        # vertical
        for x in range(self.Lx):
            for y in range(self.Ly - 1):
                out.append(((x, y), (x, y + 1)))
            if self.periodic_y and self.Ly > 2:
                out.append(((x, self.Ly - 1), (x, 0)))
        return out

    def spin_bonds(self, spin: int):
        """Hopping bonds as (qubit_a, qubit_b) pairs for one spin species,
        always ordered a < b (JW strings are defined between a and b)."""
        pairs = []
        for s1, s2 in self.bonds():
            a = self.qubit(s1, spin)
            b = self.qubit(s2, spin)
            pairs.append((a, b) if a < b else (b, a))
        return pairs

    # ---- initial-state helpers ----
    def neel_occupation(self):
        """A half-filling Neel-type product state: site (x,y) hosts a single
        up-fermion if (x+y) is even, a single down-fermion if (x+y) is odd.

        Returns a length-n_qubits list of 0/1 occupations indexed by qubit.
        Uniform density (one fermion per site) with staggered magnetization --
        a clean, ED-checkable initial state for quench dynamics, echoing the
        antiferromagnet-then-delocalise setup of arXiv:2511.02125.
        """
        occ = [0] * self.n_qubits
        for (x, y) in self._snake:
            spin = UP if ((x + y) % 2 == 0) else DOWN
            occ[self.qubit((x, y), spin)] = 1
        return occ

    def occupation_to_label(self, occ):
        """Turn a per-qubit occupation list into a bitstring label with qubit 0
        as the LEFT-most character (matches pytket/qiskit little-vs-big handling
        used by the circuit builders in this project)."""
        return "".join(str(int(b)) for b in occ)

    def edge_coloring(self):
        """Greedy edge-colouring of the *spatial* bond graph so that bonds
        sharing a site get different colours. Returns a list of colour groups,
        each a list of ((x1,y1),(x2,y2)) bonds. Hops within one colour group act
        on disjoint sites and so can be placed in the same Trotter sub-layer.

        Uses networkx if available; otherwise falls back to a single group
        (correct, just less parallel).
        """
        try:
            import networkx as nx
        except Exception:
            return [self.bonds()]
        g = nx.Graph()
        g.add_edges_from(self.bonds())
        coloring = nx.coloring.greedy_color(nx.line_graph(g))
        groups = {}
        for edge, color in coloring.items():
            groups.setdefault(color, []).append(edge)
        return list(groups.values())

    def __repr__(self):
        return (f"HubbardLattice({self.Lx}x{self.Ly}, "
                f"periodic=({self.periodic_x},{self.periodic_y}), "
                f"n_sites={self.n_sites}, n_qubits={self.n_qubits})")