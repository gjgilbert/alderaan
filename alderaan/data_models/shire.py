__all__ = ["Shire"]


class Shire:
    """Star, planets, and litecurve for a single system"""

    def __init__(self, star, planets, litecurve):
        self.star = star
        self.planets = planets
        self.litecurve = litecurve
