"""Star: stellar identifiers and limb darkening coefficients"""

__all__ = ["Star"]

from alderaan.utils.astro import ld_u_to_q


class Star:
    """
    Star

    Parameters
    ----------
    mission, target : str
    koi_id : str
    kic_id : int
    u1, u2 : float
        quadratic limb darkening coefficients; held fixed in the quick fits and
        the starting point of the sampler
    ld_prior : bool
        if True, (u1, u2) are measured values and the transit model likelihood
        includes a Gaussian prior on limb darkening centered on them; if False,
        they are only a starting point and limb darkening is sampled freely
    """

    def __init__(self, mission, target, koi_id, kic_id, u1, u2, ld_prior=True):
        self.mission = mission
        self.target = target
        self.koi_id = koi_id
        self.kic_id = kic_id

        # quadratic limb darkening coefficients
        self.u1 = u1
        self.u2 = u2
        self.ld_prior = ld_prior

    @property
    def q1(self):
        """Limb darkening q1 (Kipping 2013)"""
        return ld_u_to_q(self.u1, self.u2)[0]

    @property
    def q2(self):
        """Limb darkening q2 (Kipping 2013)"""
        return ld_u_to_q(self.u1, self.u2)[1]
