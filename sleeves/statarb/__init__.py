"""Statistical arbitrage on cointegrated equity pairs.

Screen every pair in a universe for cointegration on a formation window, then
trade the survivors out of sample. notes/statarb.md is the write-up; the short
version is that the screen predicts almost nothing, and the sections after
that measure why.
"""
