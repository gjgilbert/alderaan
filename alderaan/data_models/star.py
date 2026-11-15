__all__ = ["Star"]


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
