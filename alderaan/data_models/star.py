__all__ = ["Star"]

from alderaan.utils.astro import ld_u_to_q


class Star:
    """Star"""

    def __init__(self, mission, target, koi_id, kic_id, u1, u2):
        self.mission = mission
        self.target = target
        self.koi_id = koi_id
        self.kic_id = kic_id

        # quadratic limb darkening coefficients
        self.u1 = u1
        self.u2 = u2

    @property
    def q1(self):
        """Limb darkening q1 (Kipping 2013)"""
        return ld_u_to_q(self.u1, self.u2)[0]

    @property
    def q2(self):
        """Limb darkening q2 (Kipping 2013)"""
        return ld_u_to_q(self.u1, self.u2)[1]
