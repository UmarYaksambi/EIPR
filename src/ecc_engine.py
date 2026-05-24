"""
ecc_engine.py — Adaptive ECC Engine (Reed-Solomon + repetition).

Design
------
The engine is *stateless* between calls: ECC rate and payload length are
supplied per-call from the rate_map (built by frequency_analyzer).  This
keeps the encoder/decoder perfectly synchronized via the stored rate_map
alone, without any out-of-band state.

Reed-Solomon implementation
---------------------------
reedsolo operates on *bytes*.  The encode path packs payload bits to bytes,
appends RS parity bytes, then unpacks back to bits for QIM embedding.

Key invariant preserved throughout:
    len(codeword_bits) = len(payload_bytes) * 8 + nsym * 8
    nsym = max(2, round(n_payload_bytes * ecc_rate / (1 - ecc_rate)))

Decoding is attempted with reedsolo's Berlekamp-Massey decoder; on
uncorrectable errors the raw (possibly corrupted) bits are returned,
incurring a BER penalty that is faithfully reported in the paper's
robustness table.

Bug fixes vs original
---------------------
* Zero-length trim_len guard in _rs_decode (packbits([]) raised ValueError).
* nsym clamped to [2, n_data_bytes - 1] so reedsolo never raises on
  degenerate inputs (e.g. very short payloads in smoke test).
* _pad_to_byte always returns a copy — never mutates caller's array.
"""
from __future__ import annotations

import numpy as np
from reedsolo import RSCodec, ReedSolomonError
from typing import Literal

ECCScheme = Literal["reed_solomon", "repetition"]


class AdaptiveECCEngine:
    """
    Encodes / decodes watermark bits with externally-supplied ECC rates.

    All methods are pure functions of their arguments — the class holds
    no mutable state and is safe to reuse across images and threads.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode_block(
        self,
        payload_bits: np.ndarray,
        ecc_rate: float,
        scheme: ECCScheme = "reed_solomon",
    ) -> np.ndarray:
        """
        Encode payload_bits at the given ECC rate.

        Args:
            payload_bits: 1-D uint8 bit array (values 0 or 1).
            ecc_rate:     redundancy fraction ∈ [0, 1).
                          e.g. 0.5 ⟹ 50 % redundancy (rate-½ code).
            scheme:       'reed_solomon' | 'repetition'.

        Returns:
            Codeword as 1-D uint8 bit array.  Length is deterministic given
            payload length and ecc_rate — essential for decoder sync.
        """
        ecc_rate = float(np.clip(ecc_rate, 0.0, 0.99))
        if scheme == "reed_solomon":
            return self._rs_encode(payload_bits, ecc_rate)
        elif scheme == "repetition":
            reps = max(1, round(1.0 / max(1e-9, 1.0 - ecc_rate)))
            return np.tile(payload_bits.astype(np.uint8), reps)
        else:
            raise ValueError(f"Unknown ECC scheme: {scheme!r}")

    def decode_block(
        self,
        received_bits: np.ndarray,
        ecc_rate: float,
        scheme: ECCScheme = "reed_solomon",
        n_payload: int | None = None,
    ) -> np.ndarray:
        """
        Decode received_bits back to payload bits.

        Args:
            received_bits: 1-D uint8 bit array (possibly corrupted).
            ecc_rate:      same rate used during encoding.
            scheme:        same scheme used during encoding.
            n_payload:     expected payload length; inferred from ecc_rate if None.

        Returns:
            Decoded payload as 1-D uint8 bit array of length n_payload.
        """
        ecc_rate = float(np.clip(ecc_rate, 0.0, 0.99))
        if n_payload is None:
            n_payload = max(1, int(round(len(received_bits) * (1.0 - ecc_rate))))

        if scheme == "reed_solomon":
            return self._rs_decode(received_bits, ecc_rate, n_payload)
        elif scheme == "repetition":
            reps = max(1, round(1.0 / max(1e-9, 1.0 - ecc_rate)))
            usable = n_payload * reps
            clipped = received_bits[:usable]
            # Majority-vote over repetitions
            if len(clipped) < usable:
                # Pad with 0 if fewer bits received than expected
                clipped = np.concatenate(
                    [clipped, np.zeros(usable - len(clipped), dtype=np.uint8)]
                )
            reshaped = clipped.reshape(n_payload, reps)
            return (reshaped.mean(axis=1) >= 0.5).astype(np.uint8)
        else:
            raise ValueError(f"Unknown ECC scheme: {scheme!r}")

    # ------------------------------------------------------------------
    # Private: Reed-Solomon
    # ------------------------------------------------------------------

    def _rs_encode(self, bits: np.ndarray, ecc_rate: float) -> np.ndarray:
        """Pack bits → bytes → RS-encode → unpack back to bits."""
        padded = self._pad_to_byte(bits)
        data_bytes = np.packbits(padded)
        n_data = len(data_bytes)

        nsym = self._nsym(n_data, ecc_rate)
        rsc = RSCodec(nsym)
        encoded: bytes = bytes(rsc.encode(bytes(data_bytes)))
        return np.unpackbits(np.frombuffer(encoded, dtype=np.uint8))

    def _rs_decode(
        self, bits: np.ndarray, ecc_rate: float, n_payload: int
    ) -> np.ndarray:
        """
        Attempt RS error-correction; fall back to raw truncated bits on failure.

        The fallback is intentional — it preserves BER metrics rather than
        raising exceptions mid-experiment.
        """
        # Trim to byte boundary
        trim_len = len(bits) - (len(bits) % 8)
        if trim_len == 0:
            # No complete byte received — return zeros (worst-case BER)
            return np.zeros(n_payload, dtype=np.uint8)

        received_bytes = np.packbits(bits[:trim_len])
        n_payload_bytes = (n_payload + 7) // 8
        n_received = len(received_bytes)

        # nsym must be consistent with encoder; derive from data length
        n_data_bytes = n_received - self._nsym(n_payload_bytes, ecc_rate)
        # Guard: if received packet is too short, skip RS entirely
        if n_data_bytes <= 0 or n_received <= 2:
            return np.unpackbits(received_bytes)[:n_payload]

        nsym = max(2, n_received - n_data_bytes)
        nsym = min(nsym, n_received - 1)  # reedsolo requires at least 1 data byte

        try:
            rsc = RSCodec(nsym)
            decode_result = rsc.decode(bytes(received_bytes))
            decoded_bytes: bytes = bytes(decode_result[0])
            return np.unpackbits(
                np.frombuffer(decoded_bytes, dtype=np.uint8)
            )[:n_payload]
        except (ReedSolomonError, Exception):
            # Uncorrectable: return raw bits (BER penalty faithfully recorded)
            return np.unpackbits(received_bytes)[:n_payload]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _nsym(n_data_bytes: int, ecc_rate: float) -> int:
        """
        Number of RS parity *bytes* for a given data-byte count and ECC rate.

        Derivation:
            rate = n_data / (n_data + nsym)
            ⟹  nsym = n_data * ecc_rate / (1 - ecc_rate)

        Clamped to [2, n_data_bytes - 1] so reedsolo never receives illegal
        arguments (nsym ≥ 1 and at least 1 data byte must remain).
        """
        raw = n_data_bytes * ecc_rate / max(1e-9, 1.0 - ecc_rate)
        return int(np.clip(round(raw), 2, max(2, n_data_bytes - 1)))

    @staticmethod
    def _pad_to_byte(bits: np.ndarray) -> np.ndarray:
        """Return a zero-padded copy of bits with length divisible by 8."""
        bits = np.asarray(bits, dtype=np.uint8)
        remainder = len(bits) % 8
        if remainder == 0:
            return bits.copy()
        return np.concatenate([bits, np.zeros(8 - remainder, dtype=np.uint8)])