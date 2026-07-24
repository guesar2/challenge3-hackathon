"""
iceberg_decode.py

Classical post-processing for Iceberg-encoded shot data (produced by
iceberg_tfim_circuit.py's circuits, run via qnexus_backend.py-style
submission): reconstruct the S_Z stabilizer parity from the final
destructive measurement, apply the accept/discard rule (any
syndrome-measurement round's ancilla flags, or the final S_Z check), and
decode the k logical Z_i bits back into the same '0'/'1'-bitstring
convention shot_observables.py expects.
"""
from iceberg_code import b_index, validate_k


def reconstruct_sz(data_bits, k):
    """S_Z parity bit (0 means +1 eigenvalue, i.e. "no error") from the
    n-bit final Z-basis measurement of the code register (data_bits: a
    '0'/'1' string or sequence, length n=k+2, in iceberg_code.py's
    physical-qubit index order)."""
    validate_k(k)
    return sum(int(b) for b in data_bits) % 2


def decode_logical_bits(data_bits, k):
    """Decode the k logical Z_i values from the n-bit final measurement:
    Zbar_i = Z_i Z_b (Eq. 5), so logical bit i = data_bits[i] XOR
    data_bits[b_index(k)]. Returns a k-character '0'/'1' string -- the
    same one-classical-bit-per-logical-qubit convention
    shot_observables.py's bitstrings_to_observables/bitstrings_to_mx
    expect for an N=k qubit system.
    """
    b = b_index(k)
    bit_b = int(data_bits[b])
    return ''.join(str(int(data_bits[i]) ^ bit_b) for i in range(k))


def should_discard(flag_bits, data_bits, k):
    """flag_bits: an iterable of '0'/'1' characters -- every syndrome-
    measurement round's ancilla outcomes (a1 then a2 per round),
    concatenated in round order. Discards if any flag bit is '1', or if
    the reconstructed S_Z parity from the final measurement is nonzero
    (catches any error that slipped past every mid-circuit round but
    still left the state outside the code space by the end -- see
    docs/ICEBERG_QEC_PLAN.md and tests/test_iceberg_fault_tolerance.py's
    eventually_detected_or_correct for why a mid-circuit round can
    legitimately miss an error that this final check still catches).
    """
    if any(b == '1' for b in flag_bits):
        return True
    return reconstruct_sz(data_bits, k) != 0


def decode_shots(raw_shots, k):
    """raw_shots: list of (flags_str, data_str) pairs -- flags_str is the
    per-shot concatenation of every syndrome-measurement round's (a1, a2)
    outcomes in round order, data_str is the n=k+2-character final
    Z-basis measurement string (both physical-qubit-index order matching
    iceberg_code.py).

    Returns (kept_logical_bitstrings, discard_rate): kept_logical_bitstrings
    is a list of k-character '0'/'1' strings, one per accepted shot (ready
    for shot_observables.py), and discard_rate is the fraction of shots
    discarded -- report this alongside any observable computed from
    kept_logical_bitstrings, since it directly reflects the paper's own
    "price to pay" tradeoff (Fig. 2c/3b: more protection means more
    discards, made up for with more circuit repetitions).
    """
    kept = []
    n_discarded = 0
    for flags_str, data_str in raw_shots:
        if should_discard(flags_str, data_str, k):
            n_discarded += 1
            continue
        kept.append(decode_logical_bits(data_str, k))
    discard_rate = n_discarded / len(raw_shots) if raw_shots else 0.0
    return kept, discard_rate
