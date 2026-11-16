"""Physical and instrumental constants, and the random seed"""

# Kepler integration time + readout time [days]
KEPLER_LCIT = 29.4243885 / 60 / 24
KEPLER_SCIT = 58.848777 / 3600 / 24

# exposure time [days] for each observing mode
EXPTIME = {"long cadence": KEPLER_LCIT, "short cadence": KEPLER_SCIT}

# random seed for every stochastic step, so runs are reproducible (Kepler's birth date)
SEED = 15711227
