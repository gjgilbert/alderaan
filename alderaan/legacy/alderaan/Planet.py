# Yeah, there's not much here
# I've migrated features elsewhere, so this is now a glorifed dictionary

__all__ = ["Planet"]


class Planet:
    def __init__(
        self,
        period=None,
        epoch=None,
        depth=None,
        duration=None,
        impact=None,
        tts=None,
        index=None,
        quality=None,
        overlap=None,
        dtype=None,
    ):
        self.period = period  # orbital period
        self.epoch = epoch  # reference transit time in range (0, period)
        self.depth = depth  # transit depth
        self.duration = duration  # transit duration
        self.impact = impact  # impact parameter

        self.tts = tts  # midtransit times
        self.index = index  # index of each transit time
        self.quality = quality  # bool flag per transit; True=good
        self.overlap = overlap  # bool flag per transit; True = transit overlaps with another planet
