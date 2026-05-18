from __future__ import annotations

import numpy as np
from reedsolo import RSCodec, ReedSolomonError
from typing import Literal

# Explicit type alias so Pylance resolves it correctly everywhere
ECCScheme = Literal["reed_solomon", "repetition"]


class AdaptiveECCEngine:
    """
    Encodes / decodes watermark bits with per-block adaptive rate selection.

    The ECC rate is supplied externally per block (from frequency_analyzer's
    rate_map) so the engine itself is stateless between calls.
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
        Encode payload_bits with the given ECC rate.

        Args:
            payload_bits: 1-D uint8 bit array (values 0 or 1)
            ecc_rate:     fraction of total codeword devoted to redundancy,
                          e.g. 0.5 => 50% redundancy (rate-1/2 code)
            scheme:       'reed_solomon' | 'repetition'

        Returns:
            codeword as 1-D uint8 bit array
        """
        if scheme == "reed_solomon":
            return self._rs_encode(payload_bits, ecc_rate)
        elif scheme == "repetition":
            reps = max(1, round(1.0 / max(1e-6, 1.0 - ecc_rate)))
            return np.tile(payload_bits, reps)
        else:
            raise ValueError(f"Unknown scheme: {scheme!r}")

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
            received_bits: 1-D uint8 bit array (possibly corrupted)
            ecc_rate:      same rate used during encoding
            scheme:        same scheme used during encoding
            n_payload:     expected number of payload bits; inferred if None

        Returns:
            decoded payload as 1-D uint8 bit array
        """
        if n_payload is None:
            n_payload = int(len(received_bits) * (1.0 - ecc_rate))

        if scheme == "reed_solomon":
            return self._rs_decode(received_bits, ecc_rate, n_payload)
        elif scheme == "repetition":
            reps = max(1, round(1.0 / max(1e-6, 1.0 - ecc_rate)))
            usable = n_payload * reps
            reshaped = received_bits[:usable].reshape(n_payload, reps)
            return (reshaped.mean(axis=1) >= 0.5).astype(np.uint8)
        else:
            raise ValueError(f"Unknown scheme: {scheme!r}")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _rs_encode(self, bits: np.ndarray, ecc_rate: float) -> np.ndarray:
        """Pack bits -> bytes -> RS-encode -> unpack to bits."""
        # Pad to multiple of 8 so np.packbits is lossless
        padded = self._pad_to_byte(bits)
        byte_array = np.packbits(padded)
        nsym = max(2, int(len(byte_array) * ecc_rate / max(1e-6, 1.0 - ecc_rate)))
        rsc = RSCodec(nsym)
        encoded_bytes: bytes = bytes(rsc.encode(bytes(byte_array)))
        return np.unpackbits(np.frombuffer(encoded_bytes, dtype=np.uint8))

    def _rs_decode(
        self, bits: np.ndarray, ecc_rate: float, n_payload: int
    ) -> np.ndarray:
        """Attempt RS decode; fall back to raw truncated bits on failure."""
        # Trim to byte boundary before packing
        trim_len = len(bits) - (len(bits) % 8)
        byte_array = np.packbits(bits[:trim_len])
        n_payload_bytes = (n_payload + 7) // 8
        nsym = max(2, len(byte_array) - n_payload_bytes)

        try:
            rsc = RSCodec(nsym)
            # reedsolo.decode() returns a 3-tuple:
            #   (decoded_msg, decoded_msg_with_ecc, errata_pos)
            decode_result = rsc.decode(bytes(byte_array))
            decoded_bytes: bytes = bytes(decode_result[0])
            return np.unpackbits(
                np.frombuffer(decoded_bytes, dtype=np.uint8)
            )[:n_payload]
        except (ReedSolomonError, Exception):
            # Uncorrectable errors: return raw bits (will incur BER penalty)
            return np.unpackbits(byte_array)[:n_payload]

    @staticmethod
    def _pad_to_byte(bits: np.ndarray) -> np.ndarray:
        """Zero-pad a bit array to the next multiple of 8."""
        remainder = len(bits) % 8
        if remainder == 0:
            return bits
        return np.concatenate([bits, np.zeros(8 - remainder, dtype=np.uint8)])