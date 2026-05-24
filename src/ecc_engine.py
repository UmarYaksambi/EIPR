"""
ecc_engine.py — Adaptive ECC Engine (Reed-Solomon + repetition).

Design
------
The engine is *stateless* between calls: ECC rate and payload length are
supplied per-call from the rate_map (built by frequency_analyzer).  This
keeps the encoder/decoder perfectly synchronised via the stored rate_map
alone, without any out-of-band state.

Reed-Solomon implementation
---------------------------
reedsolo operates on *bytes*.  The encode path packs payload bits to bytes,
appends RS parity bytes, then unpacks back to bits for QIM embedding.

Key invariant preserved throughout:
    nsym  = max(2, int(n_payload_bytes * ecc_rate / (1 - ecc_rate)))
    n_total_bytes = n_payload_bytes + nsym   [must be ≤ 255 for GF(256)]

Decoding infers nsym from the received codeword length:
    nsym = total_received_bytes - n_payload_bytes
This is correct as long as the full codeword is received (normal case).
On uncorrectable errors the raw bits are returned (honest BER penalty).

Bug fixes vs first revision
----------------------------
* _nsym() had an incorrect upper cap of (n_data_bytes - 1), which clipped
  nsym from 24 → 7 at rate=0.75 and 8 → 7 at rate=0.50.  This made every
  rate ≥ 0.50 produce identical codeword lengths, destroying the adaptive
  ECC's advantage and inflating BER on median/blur/regeneration attacks.
  The only valid cap is the GF(256) field size: n_total ≤ 255 bytes.
* _rs_decode() is restored to the original simple formula
  (nsym = received_bytes - n_payload_bytes) which is exact, plus the
  zero-length guard added in the first revision.
"""
from __future__ import annotations

import numpy as np
from reedsolo import RSCodec, ReedSolomonError
from typing import Literal

ECCScheme = Literal["reed_solomon", "repetition"]

# GF(256) constraint: total RS codeword must fit in one symbol alphabet
_GF256_MAX_TOTAL: int = 255


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
            ecc_rate:     redundancy fraction in [0, 1).
                          e.g. 0.75 → 75 % parity, 25 % data (rate-¼ code).
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
            if len(clipped) < usable:
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

        nsym is inferred as (total_received_bytes - n_payload_bytes), which
        exactly mirrors what the encoder produced.  This avoids any rate
        arithmetic on the decoder side — the codeword length is self-describing.

        The fallback is intentional: it preserves BER metrics rather than
        raising exceptions mid-experiment.
        """
        # --- Guard: need at least one complete byte -----------------------
        trim_len = len(bits) - (len(bits) % 8)
        if trim_len == 0:
            return np.zeros(n_payload, dtype=np.uint8)

        received_bytes = np.packbits(bits[:trim_len])
        n_payload_bytes = (n_payload + 7) // 8

        # Guard: received packet shorter than or equal to payload alone
        if len(received_bytes) <= n_payload_bytes:
            return np.unpackbits(received_bytes)[:n_payload]

        # nsym is exactly the surplus over the data bytes
        nsym = len(received_bytes) - n_payload_bytes
        nsym = min(nsym, _GF256_MAX_TOTAL - n_payload_bytes)  # GF(256) safety
        nsym = max(2, nsym)

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
        Number of RS parity bytes for a given data-byte count and ECC rate.

        Derivation:
            ecc_rate = nsym / (n_data + nsym)
            => nsym = n_data * ecc_rate / (1 - ecc_rate)

        Lower bound: 2 (minimum for any RS correction capability).
        Upper bound: GF(256) field constraint — total codeword ≤ 255 bytes.

        Note: the previous revision incorrectly applied an upper bound of
        (n_data_bytes - 1), which silently clipped nsym from 24 → 7 at
        rate=0.75, making rates 0.50 and 0.75 indistinguishable.
        """
        raw = n_data_bytes * ecc_rate / max(1e-9, 1.0 - ecc_rate)
        # GF(256) safety: n_data + nsym must not exceed 255
        max_nsym = max(2, _GF256_MAX_TOTAL - n_data_bytes)
        return int(np.clip(int(raw), 2, max_nsym))

    @staticmethod
    def _pad_to_byte(bits: np.ndarray) -> np.ndarray:
        """Return a zero-padded copy of bits with length divisible by 8."""
        bits = np.asarray(bits, dtype=np.uint8)
        remainder = len(bits) % 8
        if remainder == 0:
            return bits.copy()
        return np.concatenate([bits, np.zeros(8 - remainder, dtype=np.uint8)])