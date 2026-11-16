"""Tahap 3b - Log-Mel (64 mel, shape (64, T)) per rekaman -> features/LogMel/{machine}/{file_id}.npy"""
from _extract_common import run, logmel

if __name__ == "__main__":
    run("LogMel", logmel)
