"""Training stage of FE-LSTM (stage 2 of 3: database_generation -> training -> fe_lstm).

Trains the LSTM surrogates (stress and tangent) on the histories from database_generation and
exports their weights in the ``.npz`` format read by the fe_lstm stage.
"""

__version__ = "0.1.0"
