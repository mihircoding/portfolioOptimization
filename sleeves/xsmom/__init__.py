"""Cross-sectional momentum on the S&P 100.

Rank every name on its trailing twelve-month return, skipping the most recent
month, then hold the top fifth against the bottom fifth. The oldest and most
widely replicated anomaly there is, included here not because it is clever but
because the allocator needed a third return stream and this one costs nothing
to build - the universe was already on disk for the pairs scan.
"""
