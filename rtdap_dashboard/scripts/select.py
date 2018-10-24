#libraries
import pandas as pd
import numpy as np
import random
from datetime import datetime, date

#bokeh
from bokeh.io import show, output_notebook, push_notebook
from bokeh.plotting import figure

from bokeh.layouts import layout, column, row, WidgetBox
from bokeh.models import CustomJS, Panel, Spacer, HoverTool, LogColorMapper, ColumnDataSource,FactorRange, RangeSlider,NumeralTickFormatter
from bokeh.models.widgets import Div, Tabs, Paragraph, Dropdown, Button, PreText, Toggle, Select,DatePicker,DateRangeSlider

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

def selection_tab(rtdap_data):

    def make_base_map(tile_map=CARTODBPOSITRON_RETINA,map_width=800,map_height=500, xaxis=None, yaxis=None,
                    xrange=(-9990000,-9619944), yrange=(5011119,5310000),plot_tools="pan,wheel_zoom,reset,save"):

        p = figure(tools=plot_tools, width=map_width,height=map_height, x_axis_location=xaxis, y_axis_location=yaxis,
                    x_range=xrange, y_range=yrange, toolbar_location="above")

        p.grid.grid_line_color = None
        #p.background_fill_color = None
        p.background_fill_alpha = 0.5
        p.border_fill_color = None

        p.add_tile(tile_map)

        return p

    def rtdap_avg(df,corr,value):

        """
        returns mean for specificed attribute by highway corridor

        Keyword arguments:
        df -- dataframe to filter by corridor and calculate mean
        corr -- corridor name
        value -- dataframe column name to calculate mean
        """

        df_corr = df.loc[df['corridor'] == corr]
        mean_value = df_corr[value].mean()

        return mean_value

    def filter_selection(df, corr, date_s, date_e, weekday, tod):

        """
        returns subset of data based on corridor and time selections

        Keyword arguments:
        df -- dataframe to filter by corridor and time selections
        corr -- corridor name
        date_s -- start date
        date_e -- end date
        weekday -- day of week (Monday - Friday)
        tod -- time of day (8 tod time periods)
        """

        tod_start = tod[0]
        tod_end = tod[1]
        date_start = datetime.strptime(date_s, '%Y-%m-%d')
        date_end = datetime.strptime(date_e, '%Y-%m-%d')

        if weekday == 'All':
            weekday = df['dow'].drop_duplicates().values.tolist()
        else:
            weekday = [weekday]

        select_df = df.loc[(df['corridor'] == corr) &\
                           (df['date']>=date_start) & (df['date']<=date_end) &\
                           (df['dow'].isin(weekday)) &\
                           (df['hour']>=tod_start) & (df['hour']<=tod_end)]

        return select_df

    def summarize_metrics(df, corr, group, avg, select, label, missing):

        """
        return a summary of frequency, mean, mean difference, and count of missing values

        Keyword arguments:
        df -- dataframe to summarize
        corr -- corridor name
        group -- dataframe column name used to group and summarize data
        avg -- mean value derived from rtdap_avg(), used calculate mean diff
        select -- dateframe column name to calculate mean
        label -- name for values being calculate (ie Speed, Volumne, Time etc)
        missing -- dataframe column name of missing values
        """

        df['freq'] = 1
        df_groupby = df.groupby(group).agg({'freq':'count',
                                        select:np.mean,
                                        missing:sum}).reset_index()

        df_groupby.loc[:,'Mean Diff'] = (avg - df_groupby[select])/df_groupby[select]
        df_groupby.loc[:, group] = label
        df_groupby.columns = [corr, 'Frequency','Mean', 'Missing Values','Mean Diff']
        df_groupby = df_groupby.set_index(corr)

        return df_groupby[['Frequency','Mean','Mean Diff','Missing Values']]

    def vbar_chart(full_df, df):
        """
        returns bokeh horizontal barchart representing mean % diff

        Keyword arguments:
        df -- dataframe to derive content of barchart
        col -- column name for values to diplay in graph
        """
        df_avg = full_df.groupby('FieldDeviceID').agg({'avgSpeed':np.mean})

        df_select= df.groupby('FieldDeviceID').agg({'avgSpeed':np.mean})
        df_select.columns = ['speed']

        diff = df_avg.merge(df_select,how='left',left_index=True, right_index=True).fillna(0)
        diff_no_zero = diff.loc[(diff['avgSpeed'] > 0 )& (diff['speed'] > 0)]
        diff_no_zero['speed_difference'] = diff_no_zero['avgSpeed'] - diff_no_zero['speed']
        diff_no_zero = diff_no_zero.reset_index()

        diff_no_zero['bins'] = pd.cut(diff_no_zero['speed_difference'],bins=list(range(-20,22)),
                                     labels = list(range(-20,22))[:-1])

        source = ColumnDataSource(data = diff_no_zero.groupby('bins').agg({'speed_difference':'count'}).reset_index())

        p = figure(plot_width=1000, plot_height=150, title="Speed Difference Distribution", toolbar_location="above")

        p.vbar(x='bins' , top='speed_difference', width=1, color='navy', alpha=0.5, source = source)

        #p.yaxis.visible = False
        #p.xaxis.formatter = NumeralTickFormatter(format="0.f%")
        p.xgrid.visible = False
        p.ygrid.visible = False
        #p.background_fill_color = None
        p.background_fill_alpha = 0.5
        p.border_fill_color = None

        return p

    def hbar_chart(df,col):
        """
        returns bokeh horizontal barchart representing mean % diff

        Keyword arguments:
        df -- dataframe to derive content of barchart
        col -- column name for values to diplay in graph
        """
        df_src = df[[col]]
        df_src['type'] = df_src.index
        df_src['order'] = 0

        df_src['order'] = np.where(df_src.index =='Speed',2,df_src['order'])
        df_src['order'] = np.where(df_src.index =='Occupancy',1,df_src['order'])
        df_src['order'] = np.where(df_src.index =='Volume',0,df_src['order'])
        df_src['color'] = '#C0C0C0'
        df_src['color'] = np.where(df_src['Mean Diff'] < -.05, '#FF0000', df_src['color'])
        df_src['color'] = np.where(df_src['Mean Diff'] > .05, '#008000', df_src['color'])
        source = ColumnDataSource(data = df_src.sort_values(by='order'))

        hover = HoverTool(
                tooltips=[
                    ("Corridor Attribute", "@type"),
                    ("% Difference", "@{%s}" % (col) + '{%0.2f}'),
                ]
            )
        tools = ['reset','save',hover]
        p = figure(plot_width=400, plot_height=175, toolbar_location="above",
                   title = 'Mean Difference', tools = tools)

        p.hbar(y='order', height=0.5, left=0,fill_color ='color',line_color=None,
               right=col, color="navy", source = source)

        p.yaxis.visible = False
        p.xaxis.formatter = NumeralTickFormatter(format="0.f%")
        p.xgrid.visible = False
        p.ygrid.visible = False
        #p.background_fill_color = None
        p.background_fill_alpha = 0.5
        p.border_fill_color = None

        return source, p


    def scatter_plot(title_text):
        rng = np.random.RandomState(0)
        x = rng.randn(100)
        y = rng.randn(100)

        source = ColumnDataSource(
                data=dict(
                    x=rng.randn(100),
                    y=rng.randn(100),
                    desc=['A', 'b', 'C', 'd', 'E']*20,
                )
            )

        hover = HoverTool(
                tooltips=[
                    ("index", "$index"),
                    ("(x,y)", "($x, $y)"),
                    ("desc", "@desc"),
                ]
            )

        p = figure(plot_width=300, plot_height=250, tools=[hover, 'box_select'], toolbar_location="above",
                   title=title_text)

        p.circle('x', 'y', size=5, source=source)
        #p.background_fill_color = None
        p.background_fill_alpha = 0.5
        p.border_fill_color = None

        return p



    """
    return selection tab contents

    Keyword arguments:
    rtdap_data - dataframe containing rtdap vds detail data
    """

    #-----------------------------------------------------------------------------------------------------------------
    #submit_selection -- Data Selection Update Function

    def submit_selection():

        """
        python callback to update table and visual content based on
        user selections in the data review panel
        """

        avgs_speed = rtdap_avg(rtdap_data, corridor_select.value,'avgSpeed')
        avgs_occ = rtdap_avg(rtdap_data, corridor_select.value,'avgOccupancy')
        avgs_volume = rtdap_avg(rtdap_data, corridor_select.value,'avgVolume')

        filtered_data = filter_selection(rtdap_data, corridor_select.value,
                                         str(date_picker_start.value),
                                         str(date_picker_end.value),
                                         day_of_week.value,
                                         time_of_day.value)

        speed = summarize_metrics(filtered_data, corridor_select.value, 'corridor',avgs_speed,'avgSpeed',
                                  'Speed','missing_speed')
        occ = summarize_metrics(filtered_data, corridor_select.value, 'corridor',avgs_occ,'avgOccupancy',
                                'Occupancy', 'missing_occ')
        volume = summarize_metrics(filtered_data, corridor_select.value, 'corridor',avgs_volume,'avgVolume',
                                   'Volume', 'missing_vol')

        summary_df = speed.append(occ)
        summary_df = summary_df.append(volume)

        summary_title.text = "<h1>"+corridor_select.value+" Summary</h1>"

        summary_df_tbl = summary_df.copy()
        summary_df_tbl['Mean Diff'] = summary_df_tbl['Mean Diff'] * 100
        summary_df_tbl = summary_df_tbl.reset_index()
        summary_table.text = str(summary_df_tbl.fillna(0).to_html(index=False,
                                                        formatters = [str,'{:20,}'.format,
                                                        '{:20,.1f}'.format,'{:20,.1f}%'.format,
                                                        '{:20,}'.format],classes=[ "w3-table" , "w3-hoverable","w3-small"]))
        if len(summary_df) > 0:
            new_df = summary_df.fillna(0)
            bar_viz_new = hbar_chart(new_df,'Mean Diff')[0]
            bar_viz_src.data.update(bar_viz_new.data)
    #-----------------------------------------------------------------------------------------------------------------
    #Data Review Panel

    panel_title = Div(text="Data Review", css_classes = ["panel-heading","text-center","w3-text-white"])
    panel_text = Div(text="""Lorem Ipsum is simply dummy text of the printing and typesetting industry.
           Lorem Ipsum has been the industry's standard dummy text ever since the 1500s,
           when an unknown printer took a galley of type and scrambled it to make a type
           specimen book.""", css_classes = ["panel-content","w3-text-white"])

    #Panel Buttons
    corridor_select = Select(options=rtdap_data['corridor'].drop_duplicates().values.tolist(), title = 'Corridor:',
                            height=60, value = 'Dan Ryan Express Lane',css_classes = ["panel-content"])

    date_picker_start = DatePicker(min_date = date(2015, 1, 1),max_date = date(2018, 12, 31),
                            css_classes = ["panel-content"], title = "Start Date:",
                            height=60, value = date(2015, 12, 31))

    date_picker_end = DatePicker(min_date = date(2015, 1, 1),max_date = date(2018, 12, 31),
                             css_classes = ["panel-content"], title = "End Date:",
                            height=60, value = date(2017, 12, 31))

    time_of_day = RangeSlider(start = 1, end= 8, step=1, value=(1, 2),
                              title="Time of Day:", bar_color="black",
                              css_classes = ["panel-content"])

    tod_description = Div(text="""Time of Day Categories:<br>
                          <ol>
                          <li>8pm-6am</li>
                          <li>6pm-7am</li>
                          <li>7am-8am</li>
                          <li>9am-10am</li>
                          <li>10am-2pm</li>
                          <li>2pm-4pm</li>
                          <li>4pm-6pm</li>
                          <li>6pm-8pm</li>
                          </ol>""",
                          css_classes = ["panel-content", "caption","w3-text-white"])

    day_of_week = Select(options=['All'] + rtdap_data['dow'].drop_duplicates().values.tolist(),
                        title = "Day of Week:",css_classes = ["panel-content"], height=60,
                        value = "All")

    select_data = Button(label="Select Subset",css_classes = ["panel-content"], height=60)

    select_data.on_click(submit_selection)
    #-----------------------------------------------------------------------------------------------------------------


    #-----------------------------------------------------------------------------------------------------------------
    #Create initial content
    avgs_speed = rtdap_avg(rtdap_data, corridor_select.value,'avgSpeed')
    avgs_occ = rtdap_avg(rtdap_data, corridor_select.value,'avgOccupancy')
    avgs_volume = rtdap_avg(rtdap_data, corridor_select.value,'avgVolume')

    filtered_data = filter_selection(rtdap_data, corridor_select.value, str(date_picker_start.value),
                                     str(date_picker_end.value),
                                     day_of_week.value, time_of_day.value)

    speed = summarize_metrics(filtered_data, corridor_select.value, 'corridor',avgs_speed,'avgSpeed',
                              'Speed','missing_speed')
    occ = summarize_metrics(filtered_data, corridor_select.value, 'corridor',avgs_occ,'avgOccupancy',
                            'Occupancy', 'missing_occ')
    volume = summarize_metrics(filtered_data, corridor_select.value, 'corridor',avgs_volume,'avgVolume',
                               'Volume', 'missing_vol')

    summary_df = speed.append(occ)
    summary_df = summary_df.append(volume)
    summary_df_tbl = summary_df.copy()
    summary_df_tbl['Mean Diff'] = summary_df_tbl['Mean Diff'] * 100
    summary_df_tbl = summary_df_tbl.reset_index()

    summary_title = Div(text= "<h1>"+corridor_select.value+" Summary</h1>", width = 2000, css_classes = ["w3-panel","w3-white"])
    summary_table = Div(text="", width = 550, height = 150)

    summary_table.text = str(summary_df_tbl.fillna(0).to_html(index=False,
                                                    formatters = [str,'{:20,}'.format,
                                                    '{:20,.1f}'.format,'{:20,.1f}%'.format,
                                                    '{:20,}'.format],classes=[ "w3-table" , "w3-hoverable","w3-small"]))

    line = Div(text="<hr>", css_classes = ["w3-container"], width = 1000)
    #-----------------------------------------------------------------------------------------------------------------


    #-----------------------------------------------------------------------------------------------------------------
    #Create initial graphics

    '''#horizontal bar chart
    p = figure(plot_width=300, plot_height=100)
    p.hbar(y=[1, 2, 3], height=0.5, left=0,
           right=[1.2, 2.5, 3.7], color="navy")
    p.yaxis.visible = False
    p.xaxis.formatter = NumeralTickFormatter(format="0.0f%")'''

    bar_viz = hbar_chart(summary_df.fillna(0),'Mean Diff')
    bar_viz_src = bar_viz[0]
    bar_viz_chart = bar_viz[1]


    volume_scatter = scatter_plot('Volumes')
    time_scatter = scatter_plot('Time')

    corr_df = rtdap_data.loc[rtdap_data['corridor'] == corridor_select.value]
    speed_diff_vbar = (vbar_chart(corr_df,filtered_data))
    occ_diff_vbar = (vbar_chart(corr_df,filtered_data))
    volume_diff_vbar = (vbar_chart(corr_df,filtered_data))

    base_map = make_base_map(map_width=450,map_height=960, xaxis=None, yaxis=None,
                xrange=(-9990000,-9619944), yrange=(5011119,5310000),plot_tools="pan,wheel_zoom,reset,save")

    select_content =  row(
           #PANEL
           column(panel_title, panel_text, corridor_select,date_picker_start,
               date_picker_end, day_of_week, time_of_day,tod_description,
               select_data, height = 1000, css_classes = ["w3-sidebar", "w3-bar-block","w3-darkgrey"]),
           column(css_classes=["w3-col"], width = 275 ),
          #CONTENT
           column(summary_title,
                row(Spacer(width=20),
                    column(Spacer(height=10),
                           row(summary_table,Spacer(width=50),bar_viz_chart,css_classes = ["w3-panel","w3-white","w3-card-4"]),
                           Spacer(height=10),
                           row(volume_scatter,Spacer(width=10),time_scatter, css_classes = ["w3-panel","w3-white","w3-card-4"], width = 650),
                           Spacer(height=10),
                           row(column(speed_diff_vbar,occ_diff_vbar,volume_diff_vbar), css_classes = ["w3-panel","w3-white","w3-card-4"], width = 1050),
                ),
                    row(Spacer(width=20),column(Spacer(height=10),column(base_map, css_classes = ["w3-panel","w3-white","w3-card-4"],width = 500)))
              ), css_classes=["w3-container", "w3-row-padding"]),
          css_classes = ["w3-container","w3-light-grey"], width = 2000, height = 1200)

    return select_content
