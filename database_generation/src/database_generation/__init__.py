"""Database generation stage of FE-LSTM (stage 1 of 3: database_generation -> training -> fe_lstm).

Builds the representative volume element (RVE) with microgen, then runs micro-scale fedoo/simcoon
simulations along random strain paths and stores the strain-stress (and tangent) histories that
the training stage reads.
"""

__version__ = "0.1.0"
