"""Private cache allocation compatibility for the measured historical harness."""
from lingbot_map.optimizations.thor.cache import FlashInferKVCacheManager


class HistoricalCache(FlashInferKVCacheManager):
    """Retain the old eager-to-graph special-page mapping in both test arms.

    Eager allocation pops descending IDs, while graph append addresses ascending
    IDs. The original workload retained this scale-special handoff gap. Do not
    install this adapter on the official streaming path or use it as a quality
    reference. The shared cache manager keeps its corrected ascending allocator.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._restore_historical_free_pages()

    def _restore_historical_free_pages(self):
        self.free_special_pages = [
            list(range(self.max_patch_pages, self.max_num_pages))
            for _ in range(self.num_blocks)
        ]

    def reset(self):
        super().reset()
        self._restore_historical_free_pages()
