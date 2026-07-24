"""Tier-0 tests for iceberg_decode.py -- pure Python, no circuits/qnexus."""
from iceberg_decode import decode_logical_bits, decode_shots, reconstruct_sz, should_discard

K = 4  # n = 6: qubits 0,1,2,3 = [k], 4 = t, 5 = b


def test_reconstruct_sz_all_zero_is_even():
    assert reconstruct_sz("000000", K) == 0


def test_reconstruct_sz_odd_weight_is_flagged():
    assert reconstruct_sz("100000", K) == 1
    assert reconstruct_sz("110000", K) == 0  # even weight -> parity 0


def test_decode_logical_bits_all_zero():
    assert decode_logical_bits("000000", K) == "0000"


def test_decode_logical_bits_xors_against_b():
    # b_index(4) == 5; flipping b flips every decoded logical bit.
    assert decode_logical_bits("000001", K) == "1111"
    # flipping only qubit 0 (and not b) flips only logical bit 0.
    assert decode_logical_bits("100000", K) == "1000"


def test_should_discard_on_any_flag_bit():
    assert should_discard("10", "000000", K) is True
    assert should_discard("00", "000000", K) is False


def test_should_discard_on_nonzero_final_sz():
    assert should_discard("00", "100000", K) is True  # odd weight -> S_Z=-1


def test_decode_shots_filters_and_reports_rate():
    raw_shots = [
        ("00", "000000"),  # keep, logical 0000
        ("10", "000000"),  # discard (flag)
        ("00", "100000"),  # discard (S_Z odd)
        ("00", "110000"),  # keep, logical 1100
    ]
    kept, rate = decode_shots(raw_shots, K)
    assert kept == ["0000", "1100"]
    assert rate == 0.5


def test_decode_shots_empty():
    kept, rate = decode_shots([], K)
    assert kept == []
    assert rate == 0.0
