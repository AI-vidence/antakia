from __future__ import annotations

import pandas as pd

from antakia.compute.model_subtitution.model_interface import InterpretableModels
from antakia.data_handler.rules import Rule, RuleSet
from antakia.utils.utils import colors, boolean_mask

import antakia.config as cfg


class Region:
    """
    class to handle regions
    a region is defined either by a selection of point or by a set of rules
    """
    region_colors = colors

    def __init__(self, X, rules: RuleSet | None = None, mask: pd.Series | None = None, color=None):
        """

        Parameters
        ----------
        X : base dataframe to use for the rule
        rules : list of rules
        mask : selected points
        color: region color, if not provided, auto assigned
        """
        self.X = X
        self.num = 0
        self.rules = RuleSet(rules)
        if mask is None:
            # if no mask, compute it
            if rules is not None:
                self.mask = self.rules.get_matching_mask(X)
            else:
                self.mask = pd.Series([False] * len(X), index=X.index)
        else:
            self.mask = mask
        self._color = color
        self.validated = False
        self.auto_cluster = False

    @property
    def color(self):
        """
        get region color
        Returns
        -------

        """
        if self._color is None:
            return self.region_colors[(self.num - 1) % len(self.region_colors)]
        return self._color

    @color.setter
    def color(self, c):
        """
        set region color
        Parameters
        ----------
        c

        Returns
        -------

        """
        self._color = c

    @property
    def name(self):
        """
        get region name
        Returns
        -------

        """
        name = repr(self.rules)
        if self.auto_cluster:
            if name:
                name = 'AC: ' + name
            else:
                name = "auto-cluster"

        if len(name) > cfg.MAX_RULES_DESCR_LENGTH:
            name = name[:cfg.MAX_RULES_DESCR_LENGTH - 2] + '..'

        return name

    def to_dict(self) -> dict[str, str | int | None]:
        """
        get region as dict
        Returns
        -------

        """
        dict_form = {
            "Region": self.num,
            "Rules": self.name,
            "Points": self.mask.sum(),
            "% dataset": f"{round(self.mask.mean() * 100, 2)}%",
            "Sub-model": None,
            'color': self.color
        }
        return dict_form

    def num_points(self) -> int:
        """
        get the number of points on the region
        Returns
        -------

        """
        return self.mask.sum()

    def dataset_cov(self):
        """
        get Region's dataset coverage (% of points in the Region)
        Returns
        -------

        """
        return self.mask.mean()

    def validate(self):
        """
        set Region as validated
        will not be erased by auto clustering
        Returns
        -------

        """
        self.validated = True


class ModelRegion(Region):
    """
    supercharged Region with an explainable predictive model
    """
    def __init__(self, X, y, X_test, y_test, customer_model, rules: RuleSet | None = None,
                 mask: pd.Series | None = None, color=None,
                 score=None):
        """

        Parameters
        ----------
        X: base train dataset
        y: relative target
        X_test: test dataset
        y_test: relative target
        customer_model: customer model
        rules: list of rules definiing the region
        mask: mask defining the region
        color: region's color
        score: customer provided scoring method
        """
        super().__init__(X, rules, mask, color)
        self.y = y
        self.X_test = X_test
        self.y_test = y_test
        self.customer_model = customer_model
        self.interpretable_models = InterpretableModels(score)

    def to_dict(self):
        """
        transform region to dict
        Returns
        -------

        """
        dict_form = super().to_dict()
        if self.interpretable_models.selected_model is not None:
            dict_form['Sub-model'] = self.interpretable_models.selected_model_str()
        return dict_form

    def select_model(self, model_name: str):
        """
        select a model between all interpretable models
        Parameters
        ----------
        model_name : model to select

        Returns
        -------

        """
        self.interpretable_models.select_model(model_name)

    def train_subtitution_models(self):
        """
        train substitution models
        Returns
        -------

        """
        if self.X_test is not None and self.test_mask is not None:
            self.interpretable_models.get_models_performance(
                self.customer_model,
                self.X.loc[self.mask],
                self.y.loc[self.mask],
                self.X_test.loc[self.test_mask],
                self.y_test.loc[self.test_mask]
            )
        else:
            self.interpretable_models.get_models_performance(
                self.customer_model,
                self.X.loc[self.mask],
                self.y.loc[self.mask],
                None,
                None
            )

    @property
    def perfs(self):
        """
        get model performance statistics
        Returns
        -------

        """
        perfs = self.interpretable_models.perfs
        if len(perfs) == 0:
            return perfs
        return perfs.sort_values(self.interpretable_models.custom_score_str, ascending=True)

    @property
    def delta(self):
        """
        get performance difference between selected model and customer model
        Returns
        -------

        """
        if self.interpretable_models.selected_model:
            return self.interpretable_models.perfs.loc[self.interpretable_models.selected_model, 'delta']
        return 0

    @property
    def test_mask(self):
        """
        select testing sample from test set
        Returns
        -------

        """
        if self.rules:
            return self.rules.get_matching_mask(self.X_test)


class RegionSet:
    """
    group of regions
    """
    def __init__(self, X):
        """

        Parameters
        ----------
        X: reference dataset
        """
        self.regions: dict[int:Region] = {}
        self.insert_order: list[int] = []
        self.display_order: list[Region] = []
        self.X = X

    def get_new_num(self) -> int:
        """
        get a new Region id
        Returns
        -------

        """
        if len(self.regions) == 0:
            return 1
        else:
            for i in range(1, len(self.regions) + 1):
                if self.regions.get(i) is None:
                    return i
            return len(self.regions) + 1

    def get_max_num(self) -> int:
        """
        get biggest region id
        Returns
        -------

        """
        if not len(self.regions):
            return 0
        return max(self.insert_order)

    def add(self, region: Region) -> None:
        """
        add a new Region to the set
        prefer the add region method
        Parameters
        ----------
        region

        Returns
        -------

        """
        if region.num < 0 or self.get(region.num) is not None:
            num = self.get_new_num()
            region.num = num
        self.regions[region.num] = region
        self.insert_order.append(region.num)
        self.display_order.append(region)

    def add_region(self, rules:RuleSet=None, mask=None, color=None, auto_cluster=False) -> Region:
        """
        create a Region from a rule set or a mask
        Parameters
        ----------
        rules : rule list
        mask : selection mask
        color : region color
        auto_cluster: is from autoclustering ?

        Returns
        -------
        the created region

        """
        if mask is not None:
            mask = mask.reindex(self.X.index).fillna(False)
        region = Region(X=self.X, rules=rules, mask=mask, color=color)
        region.num = -1
        region.auto_cluster = auto_cluster
        self.add(region)
        return region

    def extend(self, region_set: RegionSet) -> None:
        """
        add the provided RegionSet into the current one
        rebuilds all Regions
        Parameters
        ----------
        region_set

        Returns
        -------

        """
        for region in region_set.regions.values():
            self.add_region(region.rules, region.mask, region._color, region.auto_cluster)

    def remove(self, region_num) -> None:
        """
        remove Region from set
        Parameters
        ----------
        region_num

        Returns
        -------

        """
        self.insert_order.remove(region_num)
        self.display_order.remove(self.regions[region_num])
        del self.regions[region_num]

    def to_dict(self) -> list[dict]:
        """
        dict like RegionSet
        Returns
        -------

        """
        return [region.to_dict() for region in self.display_order]

    def get_masks(self) -> list[pd.Series]:
        """
        returns all Region masks
        Returns
        -------

        """
        return [region.mask for region in self.display_order]

    @property
    def mask(self):
        """
        get the union mask of all regions
        Returns
        -------

        """
        union_mask = boolean_mask(self.X, False)
        for mask in self.get_masks():
            union_mask |= mask
        return union_mask

    def get_colors(self) -> list[str]:
        """
        get the list of Region colors
        Returns
        -------

        """
        return [region.color for region in self.display_order]

    def get_color_serie(self) -> pd.Series:
        """
        get a pd.Series with for each sample of self.X its region color
        the value is set to grey if the sample is not in any REgion of the region set
        Returns
        -------

        """
        color = pd.Series(["grey"] * len(self.X), index=self.X.index)
        for region in self.display_order:
            color[region.mask] = region.color
        return color

    def __len__(self) -> int:
        """
        size of the region set
        Returns
        -------

        """
        return len(self.regions)

    def get(self, i) -> Region | None:
        """
        get a specific region by id
        Parameters
        ----------
        i

        Returns
        -------

        """
        return self.regions.get(i)

    def clear_unvalidated(self):
        """
        remove all unvalidated regions
        Returns
        -------

        """
        for i in list(self.regions.keys()):
            if not self.regions[i].validated:
                self.remove(i)

    def pop_last(self) -> Region:
        """
        removes and return the last region
        Returns
        -------

        """
        if len(self.insert_order) > 0:
            num = self.insert_order[-1]
            region = self.get(num)
            if not self.regions[num].validated:
                self.remove(num)
            return region

    def sort(self, by, ascending=True):
        """
        sort the region set by id, size, insert order
        Parameters
        ----------
        by = 'region_num'|'size'|'insert'
        ascending

        Returns
        -------

        """
        if by == 'region_num':
            key = lambda x: x.num
        elif by == 'size':
            key = lambda x: x.num_points()
        elif by == 'insert':
            key = lambda x: self.insert_order.index(x)
        self.display_order.sort(key=key, reverse=not ascending)

    def stats(self) -> dict:
        """ Computes the number of distinct points in the regions and the coverage in %
        """
        union_mask = self.mask
        stats = {
            'regions': len(self),
            'points': union_mask.sum(),
            'coverage': round(100 * union_mask.mean()),
        }
        return stats


class ModelRegionSet(RegionSet):

    """
    Supercharged RegionSet to handle interpretable models
    """
    def __init__(self, X, y, X_test, y_test, model, score):
        """

        Parameters
        ----------
        X: reference DatafFrame
        y: target series
        X_test: test set
        y_test: target test set
        model: customer model
        score: scoring method
        """
        super().__init__(X)
        self.y = y
        self.X_test = X_test
        self.y_test = y_test
        self.model = model
        self.score = score

    def add_region(self, rules:RuleSet=None, mask=None, color=None, auto_cluster=False) -> Region:
        """
        add new ModelRegion
        Parameters
        ----------
        rules
        mask
        color
        auto_cluster

        Returns
        -------

        """
        if mask is not None:
            mask = mask.reindex(self.X.index).fillna(False)
        region = ModelRegion(
            X=self.X,
            y=self.y,
            X_test=self.X_test,
            y_test=self.y_test,
            customer_model=self.model,
            score=self.score,
            rules=rules,
            mask=mask,
            color=color
        )
        region.num = -1
        region.auto_cluster = auto_cluster
        self.add(region)
        return region

    def get(self, i) -> ModelRegion | None:
        return super().get(i)

    def stats(self) -> dict:
        base_stats = super().stats()
        delta_score = 0
        for region in self.regions.values():
            weight = region.mask.sum()
            delta = region.delta
            delta_score += weight * delta
        delta_score /= len(self.X)
        base_stats['delta_score'] = delta_score
        return base_stats
