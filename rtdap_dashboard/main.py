#libraries
import pandas as pd
import numpy as np
from datetime import datetime, date
from os.path import dirname, join

#bokeh
from bokeh.io import show, output_notebook, push_notebook
from bokeh.plotting import figure

from bokeh.layouts import layout, column, row, WidgetBox
from bokeh.models import CustomJS, Panel, Spacer, HoverTool, LogColorMapper, ColumnDataSource,FactorRange, RangeSlider,\
                         NumeralTickFormatter
from bokeh.models.widgets import Div, Tabs, Paragraph, Dropdown, Button, PreText, Toggle, Select,\
                                DatePicker,DateRangeSlider

from bokeh.tile_providers import STAMEN_TERRAIN_RETINA,CARTODBPOSITRON_RETINA

#mapping
from shapely.geometry import Polygon, Point, MultiPoint, MultiPolygon
import geopandas as gpd

from bokeh.transform import factor_cmap
from bokeh.application.handlers import FunctionHandler
from bokeh.application import Application
from bokeh.core.properties import value

#color
from bokeh.palettes import Spectral6

from bokeh.io import curdoc

from scripts.select import selection_tab

#table
dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d')
#vds_table = pd.read_csv(join(dirname(__file__),'data','test.csv.gz'), compression='gzip', dtype={'hour':int},
#parse_dates=['date'], date_parser=dateparse)
hwynet_shp = join(dirname(__file__),'data','shapefiles','hwynet_sensor_loc.shp')

#
vds_table['ROADNAME'] = vds_table['ROADNAME'].fillna('N/A')
vds_table['ROADNAME'] = np.where(vds_table['ROADNAME'] == '0', 'N/A',vds_table['ROADNAME'])

vds_table['hour'] = vds_table['hour'].replace([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23],
                                              [1,1,1,1,1,1,2,3,3,4, 5, 5, 5, 5, 6, 6, 7, 7, 8, 8, 1, 1, 1, 1])

vds_table['dow'] = vds_table['dow'].replace([1,2,3,4,5],['Monday','Tuesday','Wednesday','Thursday','Friday'])

#side panel and view
def analytics_tab():
    panel_title = Div(text="Highway Performance", css_classes = ["panel-title","text-center"])
    panel_text = Div(text="""Lorem Ipsum is simply dummy text of the printing and typesetting industry.
           Lorem Ipsum has been the industry's standard dummy text ever since the 1500s,
           when an unknown printer took a galley of type and scrambled it to make a type
           specimen book.""", css_classes = ["panel-content"])

    #Date Range
    date_range = Div(text="Data References:<br>MMDDYYYY - MMDDYYYY",
                     css_classes = ["panel-content","text-center"])


    #Panel Buttons
    period_select = Select(options=['test 1','test 2', 'test 3'], title = 'Time Period:',
                            height=60)

    date_picker_start = DatePicker(min_date = date(2015, 1, 1),max_date = date(2017, 12, 31),
                            css_classes = ["panel-content", "col-lg-4"], title = "Start Date:",
                            height=60)

    month = Select(options=["January","February","March","April","May","June",
                            "July","August","September","October","November","December"],
                   title = "Month:", css_classes = ["panel-content", "col-lg-4"],
                            height=60)

    day_of_week = Select(options=["Monday","Tuesday","Wednesday","Thursday","Friday"],
                        title = "Day of Week:",
                            height=60)

    time_of_day = RangeSlider(start = 1, end= 24,step=1, value=(1, 7),
                        title="Time of Day:", bar_color="black", height=60)

    l1 = Div(text="test", css_classes = ["text-center"])

    return row(column(panel_title, panel_text, date_range,
                                period_select, month, day_of_week,
                                time_of_day, height = 700, css_classes = ["panel","col-lg-4"]),
                      column(l1,css_classes = ["container-fluid","col-lg-8"]),
                      css_classes = ["container-fluid","col-lg-12"])

#side panel and view
def compare_tab():
    panel_title = Div(text="[Corridor Comparison]", css_classes = ["panel-title","text-center"])
    panel_text = Div(text="""Lorem Ipsum is simply dummy text of the printing and typesetting industry.
           Lorem Ipsum has been the industry's standard dummy text ever since the 1500s,
           when an unknown printer took a galley of type and scrambled it to make a type
           specimen book.""", css_classes = ["panel-content"])

    #Date Range
    date_range = Div(text="Data References:<br>MMDDYYYY - MMDDYYYY",
                     css_classes = ["panel-content","text-center"])


    #Panel Buttons
    year_text = Div(text="<b>Year:", height=10)
    year_v1 = RangeSlider(start = 2015, end= 2017,step=1, value=(2015, 2017), height=25,
                            bar_color="black",title = "View 1")
    year_v2 = RangeSlider(start = 2015, end= 2017,step=1, value=(2015, 2017), height=25,
                            bar_color="black",title = "View 2")

    season_text = Div(text="<b>Season</b><br> (1=Winter, 2=Fall, 3=Spring, 4=Summer):", height=25)
    season_v1 = RangeSlider(start = 1, end= 4,step=1, value=(1, 2), height=25,
                            bar_color="black",title = "View 1")
    season_v2 = RangeSlider(start = 1, end= 4,step=1, value=(1, 2), height=25,
                            bar_color="black",title = "View 2")

    month_text = Div(text="<b>Month:", height=10)
    month_v1 = RangeSlider(start = 1, end= 12,step=1, value=(1, 2), height=25,
                            bar_color="black",title = "View 1")
    month_v2 = RangeSlider(start = 1, end= 12,step=1, value=(1, 2), height=25,
                            bar_color="black",title = "View 2")

    week_text = Div(text="<b>Day of Week:", height=10)
    week_v1 = RangeSlider(start = 1, end= 5,step=1, value=(1, 5), height=25,
                            bar_color="black",title = "View 1")
    week_v2 = RangeSlider(start = 1, end= 5,step=1, value=(1, 5), height=25,
                            bar_color="black",title = "View 2")

    time_of_day_text = Div(text="<b>Time of Day:", height=10)
    time_of_day_v1 = RangeSlider(start = 1, end= 24,step=1, value=(1, 7),
                       bar_color="black", height=25)
    time_of_day_v2 = RangeSlider(start = 1, end= 24,step=1, value=(1, 7),
                        bar_color="black", height=25)

    l1 = Div(text="test", css_classes = ["text-center"])

    return row(column(panel_title, panel_text, date_range,
                              year_text, year_v1, year_v2, Spacer(height=25),
                              season_text, season_v1, season_v2, Spacer(height=25),
                              month_text, month_v1, month_v2, Spacer(height=25),
                              week_text, week_v1, week_v2, Spacer(height=25),
                              time_of_day_text, time_of_day_v1, time_of_day_v2,
                             height = 1000, css_classes = ["panel","col-lg-4"]),
                      column(l1,css_classes = ["container-fluid","col-lg-8"]),
                      css_classes = ["container-fluid","col-lg-12"])

l0=Div(text="Test")
l1=Div(text="Test")
l2=Div(text="Test")
l3=Div(text="Test")

tab_0 = Panel(child=l0, title ='Overview')
#tab_1 = Panel(child=selection_tab(vds_table,hwynet_shp), title ='Data Selection', css_classes = ["w3-light-grey"])
tab_2 = Panel(child=analytics_tab(), title ='Analytics')
tab_3 = Panel(child=compare_tab(), title ='Comparison')

tabs = Tabs(tabs = [tab_0, tab_1, tab_2, tab_3], sizing_mode = "scale_width")

curdoc().add_root(tabs)
