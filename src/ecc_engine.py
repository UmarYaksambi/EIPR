# src/ecc_engine.py
"""
ecc_engine.py — Adaptive ECC Engine (Reed-Solomon + repetition).
"""
from __future__ import annotations

import numpy as np
from reedsolo import RSCodec, ReedSolomonError
from typing import Literal

ECCScheme = Literal["reed_solomon", "repetition"]

_GF256_MAX_TOTAL: int = 255


class AdaptiveECCEngine:
    def encode_block(
        self,
        payload_bits: np.ndarray,
        ecc_rate: float,
        scheme: ECCScheme = "reed_solomon",
    ) -> np.ndarray:
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
    ) -> tuple[np.ndarray, bool]:
        """
        Decode received_bits back to payload bits.
        Returns:
            (decoded_bits, is_success) where is_success is True if the 
            underlying mathematical ECC succeeded without throwing errors.
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
            return (reshaped.mean(axis=1) >= 0.5).astype(np.uint8), True
        else:
            raise ValueError(f"Unknown ECC scheme: {scheme!r}")

    def _rs_encode(self, bits: np.ndarray, ecc_rate: float) -> np.ndarray:
        padded = self._pad_to_byte(bits)
        data_bytes = np.packbits(padded)
        n_data = len(data_bytes)

        nsym = self._nsym(n_data, ecc_rate)
        rsc = RSCodec(nsym)
        encoded: bytes = bytes(rsc.encode(bytes(data_bytes)))
        return np.unpackbits(np.frombuffer(encoded, dtype=np.uint8))

    def _rs_decode(
        self, bits: np.ndarray, ecc_rate: float, n_payload: int
    ) -> tuple[np.ndarray, bool]:
        trim_len = len(bits) - (len(bits) % 8)
        if trim_len == 0:
            return np.zeros(n_payload, dtype=np.uint8), False

        received_bytes = np.packbits(bits[:trim_len])
        n_payload_bytes = (n_payload + 7) // 8

        if len(received_bytes) <= n_payload_bytes:
            return np.unpackbits(received_bytes)[:n_payload], False

        nsym = len(received_bytes) - n_payload_bytes
        nsym = min(nsym, _GF256_MAX_TOTAL - n_payload_bytes)
        nsym = max(2, nsym)

        try:
            rsc = RSCodec(nsym)
            decode_result = rsc.decode(bytes(received_bytes))
            decoded_bytes: bytes = bytes(decode_result[0])
            return np.unpackbits(
                np.frombuffer(decoded_bytes, dtype=np.uint8)
            )[:n_payload], True
        except (ReedSolomonError, Exception):
            return np.unpackbits(received_bytes)[:n_payload], False

    @staticmethod
    def _nsym(n_data_bytes: int, ecc_rate: float) -> int:
        raw = n_data_bytes * ecc_rate / max(1e-9, 1.0 - ecc_rate)
        max_nsym = max(2, _GF256_MAX_TOTAL - n_data_bytes)
        return int(np.clip(int(raw), 2, max_nsym))

    @staticmethod
    def _pad_to_byte(bits: np.ndarray) -> np.ndarray:
        bits = np.asarray(bits, dtype=np.uint8)
        remainder = len(bits) % 8
        if remainder == 0:
            return bits.copy()
        return np.concatenate([bits, np.zeros(8 - remainder, dtype=np.uint8)])