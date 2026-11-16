"""Tahap 3a - MFCC (20 koefisien, shape (20, T)) per rekaman -> features/MFCC/{machine}/{file_id}.npy"""
from _extract_common import run, mfcc

if __name__ == "__main__":
    run("MFCC", mfcc)
