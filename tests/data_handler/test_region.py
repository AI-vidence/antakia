import pandas as pd

from antakia.data_handler.rules import Rule
from antakia.data_handler.region import RegionSet, Region
from antakia.utils.variable import Variable


def test_regions():
    data = pd.DataFrame([
        [1, 2],
        [2, 1],
        [4, 2],
        [10, 1],
        [20, 2],
    ], columns=['var1', 'var2'])
    rs = RegionSet(data)
    assert len(rs) == 0
    var = Variable(0, 'var1', 'float')
    rule1 = Rule(None, None, var, '<', 10)
    rule2 = Rule(2, '<=', var, None, None)

    region = Region(
        X=data,
        rules=[rule1, rule2]
    )
    rs.add(region)
    assert rs.get(1) == region
    assert len(rs) == 1
    assert rs.get_max_num() == 1
    assert rs.get_num() == 2

    color = rs.get_color_serie()
    assert (color == pd.Series(['grey', 'red', 'red', 'grey', 'grey'])).all()

    rs.clear_unvalidated()
    rs.add_region([rule1, rule2], color='blue')
    assert rs.get_max_num() == 1
    assert rs.get_num() == 2
    assert rs.get(1).color == 'blue'

    color = rs.get_color_serie()
    assert (color == pd.Series(['grey', 'blue', 'blue', 'grey', 'grey'])).all()

    var2 = Variable(0, 'var2', 'float')
    rule3 = Rule(None, None, var2, '>', 1.5)
    rs.add_region([rule3], color='blue')
    rs.stats()

