from __future__ import annotations

import pandas as pd

from antakia.data_handler.rules import Rule
from antakia.utils.utils import colors

import antakia.config as cfg

class Region:
    region_colors = colors

    def __init__(self, X, rules: list[Rule] | None = None, mask: pd.Series | None = None, color=None):
        self.X = X
        self.num = 0
        self.rules = rules
        if mask is None:
            if rules is not None:
                self.mask = Rule.rules_to_mask(rules, X)
            else:
                self.mask = pd.Series([False] * len(X), index=X.index)
        else:
            self.mask = mask
        self.model = None
        self.score = None
        self._color = color
        self.validated = False

    @property
    def color(self):
        if self._color is None:
            return self.region_colors[self.num - 1 % len(self.region_colors)]
        return self._color

    @color.setter
    def color(self, c):
        self._color = c

    def set_model(self, model, score):
        self.model = model
        self.score = score

    def to_dict(self):
        rules_to_str= Rule.multi_rules_to_string(self.rules) if self.rules is not None else "auto-cluster"
        rules_to_str = (rules_to_str[:cfg.MAX_RULES_DESCR_LENGTH] + '..') if len(rules_to_str) > cfg.MAX_RULES_DESCR_LENGTH else rules_to_str
        return {
            "Region": self.num,
            "Rules": rules_to_str,
            "Points": self.mask.sum(),
            "% dataset": f"{round(self.mask.mean() * 100,3)}%",
            "Sub-model": self.model,
            "Score": self.score,
            'color': self.color
        }

    def num_points(self):
        return self.mask.sum()

    def dataset_cov(self):
        return self.mask.mean()

    def validate(self):
        self.validated = True


class RegionSet:
    def __init__(self, X):
        self.regions = {}
        self.insert_order = []
        self.X = X

    def get_new_num(self):
        if len(self.regions) == 0:
            return 1
        else:
            for i in range(1, len(self.regions) + 1):
                if self.regions.get(i) is None:
                    return i
            return len(self.regions) + 1

    def get_max_num(self):
        if not len(self.regions):
            return 0
        return max(self.insert_order)

    def add(self, region: Region):
        num = self.get_new_num()
        self.regions[num] = region
        self.insert_order.append(num)
        region.num = num

    def remove(self, region_num):
        del self.regions[region_num]
        self.insert_order.remove(region_num)

    def to_dict(self):
        return [self.regions[num].to_dict() for num in self.insert_order]

    def get_masks(self):
        return [self.regions[num].mask for num in self.insert_order]

    def get_colors(self):
        return [self.regions[num].color for num in self.insert_order]

    def get_color_serie(self):
        color = pd.Series(["grey"] * len(self.X), index=self.X.index)
        for num in self.insert_order:
            region = self.regions.get(num)
            color[region.mask] = region.color
        return color

    def __len__(self):
        return len(self.regions)

    def get(self, i):
        return self.regions.get(i)

    def add_region(self, rules=None, mask=None, color=None):
        if mask is not None:
            mask = mask.reindex(self.X.index).fillna(False)
        region = Region(X=self.X, rules=rules, mask=mask, color=color)
        self.add(region)
        return region

    def clear_unvalidated(self):
        for i in list(self.regions.keys()):
            if not self.regions[i].validated:
                self.remove(i)

    def pop_last(self):
        if len(self.insert_order) > 0:
            num = self.insert_order[-1]
            if not self.regions[num].validated:
                del self.regions[num]
                self.insert_order.remove(num)

    def stats(self) -> dict:
        """ Computes the number of distinct points in the regions and the coverage in %
        """
        union_mask = pd.Series([False] * len(self.X), index=self.X.index)
        for mask in self.get_masks():
            union_mask |= mask

        stats = {
            'regions': len(self),
            'points': union_mask.sum(),
            'coverage': round(100 * union_mask.mean())
        }
        return stats
