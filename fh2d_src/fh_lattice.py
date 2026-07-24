"""
fh_lattice.py

Geometry for the 2D Fermi-Hubbard model and the Jordan-Wigner (JW) mode
ordering used everywhere else in the project.

A spinful site (x, y) carries two fermionic modes -- one per spin sigma in
{up, down} -- so an Lx-by-Ly lattice needs 2*Lx*Ly qubits under a JW mapping.

BOUNDARY CONDITIONS
-------------------
This project uses PERIODIC boundaries in both directions, always. Periodic
boundaries restore translational invariance and cut finite-size effects, which
is what makes a 4- or 12-site cluster a meaningful stand-in for the bulk model.
There is no open-boundary option any more: it was removed so that no figure can
silently mix conventions.

One honest caveat, unchanged from before: a wrap-around bond is only emitted
when the lattice is LONGER than 2 in that direction. For L=2 the wrap bond
would connect the same pair of sites that the ordinary bond already connects,
i.e. it would double the hopping amplitude on that bond rather than add a new
one. So a 2x2 "periodic" cluster is a 4-site ring with 4 bonds -- identical to
the open 2x2 plaquette. The 3x4 cluster is genuinely periodic: 24 bonds.

Mode ordering (single source of truth for how sites/spins map to qubit indices):
  - Sites are visited in a boustrophedon ("snake") order: row 0 left-to-right,
    row 1 right-to-left, and so on, which keeps JW Z-strings short.
  - All spin-up modes come first (in snake order), then all spin-down modes.
    Same-spin hopping therefore stays within one contiguous block, and the
    on-site interaction is diagonal so it needs no Z-string at all. This block
    structure is also what lets fh_sector.py factorise the sector Hamiltonian.

This module is pure geometry: no qiskit/pytket/scipy dependency.
"""
from __future__ import annotations

UP, DOWN = 0, 1


class HubbardLattice:
    """A 2D rectangular Fermi-Hubbard lattice, periodic in both directions,
    with a fixed JW mode ordering.

    Parameters
    ----------
    Lx, Ly : int
        Lattice dimensions (Lx columns, Ly rows).
    """

    def __init__(self, Lx: int, Ly: int):
        if Lx < 1 or Ly < 1:
            raise ValueError("Lx, Ly must be >= 1")
        self.Lx = Lx
        self.Ly = Ly

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

        Horizontal bonds first, then vertical, each wrapping around (periodic).
        Wrap-around bonds are only emitted when the lattice is longer than 2 in
        that direction -- for L=2 the wrap bond would duplicate the ordinary
        bond rather than add a new one (see the module docstring).
        """
        out = []
        # horizontal
        for y in range(self.Ly):
            for x in range(self.Lx - 1):
                out.append(((x, y), (x + 1, y)))
            if self.Lx > 2:
                out.append(((self.Lx - 1, y), (0, y)))
        # vertical
        for x in range(self.Lx):
            for y in range(self.Ly - 1):
                out.append(((x, y), (x, y + 1)))
            if self.Ly > 2:
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
        UNIFORM density (one fermion per site) with staggered magnetization --
        a clean, ED-checkable initial state for quench dynamics, echoing the
        antiferromagnet-then-delocalise setup of arXiv:2511.02125.

        Because the density is uniform and stays uniform under the dynamics,
        this state is the right probe for m_stag but shows NOTHING in a
        per-site density map. Use stripe_occupation() for that.
        """
        occ = [0] * self.n_qubits
        for (x, y) in self._snake:
            spin = UP if ((x + y) % 2 == 0) else DOWN
            occ[self.qubit((x, y), spin)] = 1
        return occ

    def stripe_occupation(self):
        """A half-filling CHARGE-IMBALANCED product state ("doublon stripe").

        Columns are loaded left-to-right until half filling is reached:
          - the leftmost columns are DOUBLY occupied (<n> = 2),
          - one middle column carries a single fermion per site in a staggered
            pattern (<n> = 1) to balance N_up = N_dn,
          - the remaining columns are EMPTY (<n> = 0).

        Total particle number is still N_sites (half filling) and N_up = N_dn,
        so the state lives in the same symmetry sector as the Neel state -- but
        now the density profile is strongly non-uniform, so the per-site density
        heatmap actually shows something: doublons melting and charge spreading
        into the empty region. This is the standard charge-relaxation quench
        (cf. arXiv:2511.02125 / arXiv:2508.12307), and its rate is what the
        on-site U controls.

        Returns a length-n_qubits list of 0/1 occupations indexed by qubit.
        """
        occ = [0] * self.n_qubits
        n_up_target = n_dn_target = self.n_sites // 2
        n_up = n_dn = 0
        for x in range(self.Lx):
            col = [(x, y) for y in range(self.Ly)]
            if n_up + self.Ly <= n_up_target and n_dn + self.Ly <= n_dn_target:
                # full column of doublons
                for s in col:
                    occ[self.qubit(s, UP)] = 1
                    occ[self.qubit(s, DOWN)] = 1
                n_up += self.Ly
                n_dn += self.Ly
            else:
                # partial column: staggered singles, only as many as we still need
                for (x_, y_) in col:
                    if ((x_ + y_) % 2 == 0) and n_up < n_up_target:
                        occ[self.qubit((x_, y_), UP)] = 1
                        n_up += 1
                    elif ((x_ + y_) % 2 == 1) and n_dn < n_dn_target:
                        occ[self.qubit((x_, y_), DOWN)] = 1
                        n_dn += 1
        if n_up != n_up_target or n_dn != n_dn_target:
            raise ValueError(
                f"stripe state could not reach half filling on {self.Lx}x{self.Ly} "
                f"(got N_up={n_up}, N_dn={n_dn}, need {n_up_target} each). "
                f"Use 'neel' or an explicit bitstring for this lattice.")
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
        return (f"HubbardLattice({self.Lx}x{self.Ly}, periodic, "
                f"n_sites={self.n_sites}, n_qubits={self.n_qubits})")
