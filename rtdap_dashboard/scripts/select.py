#libraries
import pandas as pd
import numpy as np
from numpy.polynomial.polynomial import polyfit
import random
from datetime import datetime, date
from scipy import stats

#bokeh
from bokeh.io import show, output_notebook, push_notebook
from bokeh.plotting import figure

from bokeh.layouts import layout, column, row, WidgetBox
from bokeh.models import CustomJS, Panel, Spacer, HoverTool, LogColorMapper, ColumnDataSource,FactorRange, RangeSlider,NumeralTickFormatter,LinearColorMapper
from bokeh.models.widgets import Div, Tabs, Paragraph, Dropdown, Button, PreText, Toggle, Select,DatePicker,DateRangeSlider

from bokeh.tile_providers import STAMEN_TERRAIN_RETINA,CARTODBPOSITRON_RETINA
#mapping
from shapely.geometry import Polygon, Point, MultiPoint, MultiPolygon
import geopandas as gpd

from bokeh.transform import factor_cmap, linear_cmap,transform
from bokeh.application.handlers import FunctionHandler
from bokeh.application import Application
from bokeh.core.properties import value

#color
from bokeh.palettes import RdYlGn

from bokeh.io import curdoc

widget_width = 225

def selection_tab(rtdap_data,hwynet_shp):

    """
    return selection tab contents

    Keyword arguments:
    rtdap_data - dataframe containing rtdap vds detail data
    """

    def make_base_map(tile_map=CARTODBPOSITRON_RETINA,map_width=800,map_height=500, xaxis=None, yaxis=None,
                    xrange=(-9810000, -9745000), yrange=(5130000, 5130000),plot_tools="pan,wheel_zoom,reset,save,hover"):

        p = figure(tools=plot_tools, width=map_width,height=map_height, x_axis_location=xaxis, y_axis_location=yaxis,
                    x_range=xrange, y_range=yrange, toolbar_location="above")

        p.grid.grid_line_color = None
        #p.background_fill_color = None
        p.background_fill_alpha = 0.5
        p.border_fill_color = None

        p.add_tile(tile_map)

        return p

    def make_line_data(shp, roadname):

        def getLineCoords(row, geom, coord_type):
            """Returns a list of coordinates ('x' or 'y') of a LineString geometry"""
            if coord_type == 'x':
                return list( row[geom].coords.xy[0] )
            elif coord_type == 'y':
                return list( row[geom].coords.xy[1] )

        gpd_shp = gpd.read_file(shp)

        gpd_shp['x'] = gpd_shp.apply(getLineCoords, geom='geometry', coord_type='x', axis=1)

        # Calculate y coordinates of the line
        gpd_shp['y'] = gpd_shp.apply(getLineCoords, geom='geometry', coord_type='y', axis=1)

        # Make a copy and drop the geometry column
        shp_df = gpd_shp.drop('geometry', axis=1).copy()

        df = shp_df.loc[shp_df['ROADNAME'] == roadname]

        source = ColumnDataSource(df)

        return source



    def make_line_map(base_map,source):

        p = base_map

        # Point DataSource
        mapper = LinearColorMapper(palette=RdYlGn[5], low=source.data['POSTEDSPEE'].min(),
                               high=source.data['POSTEDSPEE'].max())
        p.multi_line('x', 'y', source=source, color=transform('POSTEDSPEE',mapper), line_width=2)

        return p

    def rtdap_avg(df,value):

        """
        returns average daily volume, speed, occupancy, time etc by highway corridor

        Keyword arguments:
        df -- dataframe to filter by corridor and calculate mean
        value -- dataframe column name to calculate mean
        """

        if value == 'avg_volume':
            df['hour_volume'] = df.groupby(['ROADNAME','year','doy','hour'])[value].transform('mean')
            df['sum_volume'] = df.groupby(['ROADNAME','year','doy'])['hour_volume'].transform('sum')
            #get average weekday volume
            mean_value = df['sum_volume'].mean()
        else:
            mean_value = df[value].mean()

        return mean_value

    def filter_selection(df, date_s, date_e):

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

        date_start = datetime.strptime(date_s, '%Y-%m-%d')
        date_end = datetime.strptime(date_e, '%Y-%m-%d')

        select_df = df.loc[(df['date']>=date_start) & (df['date']<=date_end)]

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

        #update to present average volumes for specific time period
        #compared to average day with same tod selected
        #add tod as argument to summarize summarize_metrics ***

        if select == 'avg_volume':
            #sum up to date to get total weekday volume
            df['hour_volume'] = df.groupby(['ROADNAME','year','doy','hour'])[select].transform('mean')
            df['sum_volume'] = df.groupby(['ROADNAME','year','doy'])['hour_volume'].transform('sum')
            select = 'sum_volume'

        df_groupby = df.groupby(group).agg({'frequency':sum,
                                        select:np.mean,
                                        missing:sum}).reset_index()

        df_groupby.loc[:,'Mean Diff'] = (avg - df_groupby[select])/df_groupby[select]
        df_groupby.loc[:, group] = label
        df_groupby.columns = [corr, 'Frequency','Mean', 'Missing Values','Mean Diff']
        df_groupby = df_groupby.set_index(corr)

        return df_groupby[['Frequency','Mean','Mean Diff','Missing Values']]

    def vbar_chart_src(full_df, df, value, label):
        """
        returns bokeh horizontal barchart representing mean % diff

        Keyword arguments:
        df -- dataframe to derive content of barchart
        col -- column name for values to diplay in graph
        """
        df_avg = full_df.groupby('ABB').agg({value:np.mean})

        df_select = df.groupby('ABB').agg({value:np.mean})
        df_select.columns = [label]

        diff = df_avg.merge(df_select,how='left',left_index=True, right_index=True).fillna(0)
        diff_no_zero = diff.loc[(diff[value] > 0 ) & (diff[label] > 0)]
        diff_no_zero[label+'_difference'] = ((diff_no_zero[value] - diff_no_zero[label])/diff_no_zero[label]) * 100
        diff_no_zero = diff_no_zero.reset_index()

        min_label = int(diff_no_zero[label+'_difference'].min())
        max_label = int(diff_no_zero[label+'_difference'].max())

        bin_values = list(range(min_label,max_label))
        labels = bin_values[:-2]

        diff_no_zero['bins'] = pd.cut(diff_no_zero[label+'_difference'],bins=list(range(-50,50)),
                                     labels = list(range(-50,50))[:-1])

        source = ColumnDataSource(data = diff_no_zero.groupby('bins').agg({label+'_difference':'count'}).round().reset_index())

        return source

    def vbar_chart(full_df, df):
        """
        returns bokeh horizontal barchart representing mean % diff

        Keyword arguments:
        df -- dataframe to derive content of barchart
        col -- column name for values to diplay in graph
        """
        df_avg = full_df.groupby('ROADNAME').agg({'avg_speed':np.mean})

        df_select = df.groupby('ROADNAME').agg({'avg_speed':np.mean})
        df_select.columns = ['speed']

        diff = df_avg.merge(df_select,how='left',left_index=True, right_index=True).fillna(0)
        diff_no_zero = diff.loc[(diff['avg_speed'] > 0 )& (diff['speed'] > 0)]
        diff_no_zero['speed_difference'] = ((diff_no_zero['avg_speed'] - diff_no_zero['speed'])/diff_no_zero['speed']) * 100
        diff_no_zero = diff_no_zero.reset_index()

        diff_no_zero['bins'] = pd.cut(diff_no_zero['speed_difference'],bins=list(range(-100,100)),
                                     labels = list(range(-100,99))[:-1])

        source = ColumnDataSource(data = diff_no_zero.groupby('bins').agg({'speed_difference':'count'}).round().reset_index())

        p = figure(plot_width=1000, plot_height=300, title="Speed Difference Distribution", toolbar_location="above")

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
        p = figure(plot_width=400, plot_height=275, toolbar_location="above",
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

    def scatter_data(df, select_df, value):

        df_avg = df.groupby('ABB').agg({value : np.mean})
        df_avg.columns = ['Average']

        select_avg = select_df.groupby('ABB').agg({value : np.mean})
        select_avg.columns = ['Selection']

        scatter_data = df_avg.merge(select_avg, how='left',
                                    left_index=True, right_index=True).reset_index().dropna()

        scatter_data_ = scatter_data.loc[(scatter_data['Average'] > 0) & (scatter_data['Selection'] > 0)]

        abb = scatter_data_['ABB'].values.tolist()

        avg_values = scatter_data_['Average']
        selection_averages = scatter_data_['Selection']

        #b, m = polyfit(avg_values, selection_averages, 1)
        slope, intercept, r_value, p_value, std_err = stats.linregress(avg_values,selection_averages)
        # We need to generate actual values for the regression line.
        #r_x, r_y = zip(*((i, i*regression[0] + regression[1]) for i in range(len(scatter_data))))


        return selection_averages, avg_values,  slope, intercept

    def scatter_src(x,y,slope, intercept):

        if type(y) is int:
            df = pd.DataFrame({'x': x})
            df['y'] = slope*df['x']+intercept

            x = df['x']
            y = df['y']

        source = ColumnDataSource(
                data=dict(
                    x=x,
                    y=y,
                )
            )

        return source

    def scatter_figure(title_text):
        p = figure(plot_width=425, plot_height=400, tools=['hover','lasso_select','reset'], toolbar_location="above",
                   title=title_text)

        p.background_fill_alpha = 0.5
        p.border_fill_color = None
        p.xaxis.axis_label = "Mean (2015 - 2017)"
        p.yaxis.axis_label = "Mean (Selection)"


        return p

    def scatter_plot(df, select_df, value, title_text):

        df_avg = df.groupby('ABB').agg({value : np.mean})
        df_avg.columns = ['Average']

        select_avg = select_df.groupby('ABB').agg({value : np.mean})
        select_avg.columns = ['Selection']

        scatter_data = df_avg.merge(select_avg, how='left',
                                    left_index=True, right_index=True).reset_index().fillna(0)

        abb = scatter_data['ABB'].values.tolist()

        avg_values = scatter_data['Average']
        selection_averages = scatter_data['Selection']

        b, m = polyfit(avg_values, selection_averages, 1)

        # We need to generate actual values for the regression line.
        r_x, r_y = zip(*((i, i*regression[0] + regression[1]) for i in range(len(scatter_data))))

        hover = HoverTool(
                tooltips=[
                    ("index", "$index"),
                    ("(x,y)", "($x, $y)"),
                    ("desc", "@desc"),
                ]
            )

        p = figure(plot_width=350, plot_height=300, tools=['hover','box_select'], toolbar_location="above",
                   title=title_text)

        p.line(r_x, r_y, color="red")
        #p.circle('x', 'y', size=5, source=source)
        p.scatter(avg_values, selection_averages,radius=.5)

        #p.background_fill_color = None
        p.background_fill_alpha = 0.5
        p.border_fill_color = None

        return p
    #-----------------------------------------------------------------------------------------------------------------
    #submit_selection -- Data Selection Update Function

    def submit_selection():

        """
        python callback to update table and visual content based on
        user selections in the data review panel
        """

        tod_start = time_of_day.value[0]
        tod_end = time_of_day.value[1]

        if day_of_week.value == 'All':
            weekday = rtdap_data['dow'].drop_duplicates().values.tolist()
        else:
            weekday = [day_of_week.value]

        df_for_avg = rtdap_data.loc[(rtdap_data['ROADNAME'] == corridor_select.value) &\
                                   (rtdap_data['dow'].isin(weekday)) &\
                                   (rtdap_data['hour']>=tod_start) & (rtdap_data['hour']<=tod_end)]

        avgs_speed = rtdap_avg(df_for_avg, 'avg_speed')
        avgs_occ = rtdap_avg(df_for_avg, 'avg_occupancy')
        avgs_volume = rtdap_avg(df_for_avg, 'avg_volume')

        filtered_data = filter_selection(df_for_avg,
                                         str(date_picker_start.value),
                                         str(date_picker_end.value))

        speed = summarize_metrics(filtered_data, corridor_select.value, 'ROADNAME',avgs_speed,'avg_speed',
                                  'Speed','missing_speed')
        occ = summarize_metrics(filtered_data, corridor_select.value, 'ROADNAME',avgs_occ,'avg_occupancy',
                                'Occupancy', 'missing_occ')
        volume = summarize_metrics(filtered_data, corridor_select.value, 'ROADNAME',avgs_volume,'avg_volume',
                                   'Volume', 'missing_vol')

        summary_df = speed.append(occ)
        summary_df = summary_df.append(volume)

        summary_title.text = "<h1>"+corridor_select.value+" Summary</h1>"
        sum_tbl_def.text ="The <b>" +corridor_select.value + """</b> summary includes a snap shot of speed, occupancy,
                        and volume based on the date and time period selections in the <b>Data Review panel</b>. Records from
                        the RTDAP were aggregated up to hourly averages for each sensor location. Each sensor was then related
                        to the CMAP major highway network (MHN) summarize speed, occupancy, and volume by corridor."""

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

            shp_df = make_line_data(hwynet_shp, corridor_select.value)

            #map data
            hwy_src.data.update(shp_df.data)

            #scatter data
            speed_data = scatter_data(df_for_avg, filtered_data, 'avg_speed')
            speed_pt_new = scatter_src(speed_data[0],speed_data[1],-1,-1)
            speed_line_new = scatter_src(speed_data[0],0,speed_data[2],speed_data[3])

            speed_line.data.update(speed_line_new.data)
            speed_pt.data.update(speed_pt_new.data)

            vol_data = scatter_data(df_for_avg, filtered_data, 'avg_volume')
            v_pt_new = scatter_src(vol_data[0],vol_data[1],-1,-1)
            v_line_new = scatter_src(vol_data[0],0,vol_data[2],vol_data[3])

            v_line.data.update(v_line_new.data)
            v_pt.data.update(v_pt_new.data)

            #vbar diff bins
            speed_vsrc_new = vbar_chart_src(df_for_avg,filtered_data,'avg_speed','Speed')
            speed_vsrc.data.update(speed_vsrc_new.data)

            occ_vsrc_new = vbar_chart_src(df_for_avg,filtered_data,'avg_occupancy','Occupancy')
            occ_vsrc.data.update(occ_vsrc_new.data)

            vol_vsrc_new = vbar_chart_src(df_for_avg,filtered_data,'avg_volume','Volume')
            vol_vsrc.data.update(vol_vsrc_new.data)

    #-----------------------------------------------------------------------------------------------------------------
    #Data Review Panel

    panel_title = Div(text="Data Review", css_classes = ["panel-heading","text-center","w3-text-white"])
    panel_text = Div(text="""Lorem Ipsum is simply dummy text of the printing and typesetting industry.
           Lorem Ipsum has been the industry's standard dummy text ever since the 1500s,
           when an unknown printer took a galley of type and scrambled it to make a type
           specimen book.""", css_classes = ["w3-bar-item","w3-text-white"], width = widget_width)

    #Panel Buttons
    gpd_shp = gpd.read_file(hwynet_shp)

    corridor_select = Select(options=rtdap_data['ROADNAME'].drop_duplicates().values.tolist(), title = 'Corridor:',
                            height=60, value = 'LAKE SHORE DR SB',css_classes = ["w3-bar-item"],width = widget_width)

    date_picker_start = DatePicker(min_date = date(2015, 1, 1),max_date = date(2018, 12, 31),
                            css_classes = ["w3-bar-item"], title = "Start Date:",
                            height=60, value = date(2015, 12, 31), width = widget_width)

    date_picker_end = DatePicker(min_date = date(2015, 1, 1),max_date = date(2018, 12, 31),
                             css_classes = ["w3-bar-item"], title = "End Date:",
                            height=60, value = date(2017, 12, 31), width = widget_width)

    time_of_day = RangeSlider(start = 1, end= 8, step=1, value=(1, 8),
                              title="Time of Day:", bar_color="black",
                              css_classes = ["w3-bar-item"], width = widget_width)

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
                          css_classes = ["w3-bar-item", "caption","w3-text-white"])

    day_of_week = Select(options=['All'] + rtdap_data['dow'].drop_duplicates().values.tolist(),
                        title = "Day of Week:",css_classes = ["w3-bar-item"], height=60,
                        value = "All", width = widget_width)

    select_data = Button(label="Select Subset",css_classes = ["w3-bar-item"], height=60, width = 150)

    select_data.on_click(submit_selection)
    #-----------------------------------------------------------------------------------------------------------------


    #-----------------------------------------------------------------------------------------------------------------
    #Create initial content
    tod_start = time_of_day.value[0]
    tod_end = time_of_day.value[1]

    if day_of_week.value == 'All':
        weekday = rtdap_data['dow'].drop_duplicates().values.tolist()
    else:
        weekday = [day_of_week.value]

    df_for_avg = rtdap_data.loc[(rtdap_data['ROADNAME'] == corridor_select.value) &\
                               (rtdap_data['dow'].isin(weekday)) &\
                               (rtdap_data['hour']>=tod_start) & (rtdap_data['hour']<=tod_end)]

    avgs_speed = rtdap_avg(df_for_avg, 'avg_speed')
    avgs_occ = rtdap_avg(df_for_avg, 'avg_occupancy')
    avgs_volume = rtdap_avg(df_for_avg, 'avg_volume')

    filtered_data = filter_selection(df_for_avg, str(date_picker_start.value),
                                     str(date_picker_end.value))

    speed = summarize_metrics(filtered_data, corridor_select.value, 'ROADNAME',avgs_speed,'avg_speed',
                              'Speed','missing_speed')
    occ = summarize_metrics(filtered_data, corridor_select.value, 'ROADNAME',avgs_occ,'avg_occupancy',
                            'Occupancy', 'missing_occ')
    volume = summarize_metrics(filtered_data, corridor_select.value, 'ROADNAME',avgs_volume,'avg_volume',
                               'Volume', 'missing_vol')

    summary_df = speed.append(occ)
    summary_df = summary_df.append(volume)
    summary_df_tbl = summary_df.copy()
    summary_df_tbl['Mean Diff'] = summary_df_tbl['Mean Diff'] * 100
    summary_df_tbl = summary_df_tbl.reset_index()

    summary_title = Div(text= "<h1>"+corridor_select.value+" Summary</h1>", width = 2000, css_classes = ["w3-bar-fixed","w3-white"])
    summary_table = Div(text="", width = 550, height = 150)

    summary_table.text = str(summary_df_tbl.fillna(0).to_html(index=False,
                                                    formatters = [str,'{:20,}'.format,
                                                    '{:20,.1f}'.format,'{:20,.1f}%'.format,
                                                    '{:20,}'.format],classes=[ "w3-table" , "w3-hoverable","w3-small"]))

    #Descriptive text
    sum_tbl_def = Div(text="The <b>" +corridor_select.value + """</b> summary includes a snap shot of speed, occupancy,
                    and volume based on the date and time period selections in the <b>Data Review panel</b>. Records from
                    the RTDAP were aggregated up to hourly averages for each sensor location. Each sensor was then related
                    to the CMAP major highway network (MHN) summarize speed, occupancy, and volume by corridor.""",
                    css_classes = ["small"], width = 250)

    tbl_def = Div(text="""<b>Table Definitions</b>:<br>
                          <ul>
                          <li><b>Frequency</b> - Number of RTDAP records</li>
                          <li><b>Mean</b> - Average for selected time period</li>
                          <li><b>Mean Diff</b> - Difference between mean value for selected year
                                                 mean for all years (2015 - 2017)</li>
                         <li><b>Missing Values</b> - Number of records with null values</li>
                          </ul>""",css_classes = ["small"], width = 250)

    line = Div(text="<hr>", css_classes = ["w3-container"], width = 500)
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


    #volume_scatter = scatter_plot(df_for_avg,filtered_data,'avg_speed','Volumes')
    #time_scatter = scatter_plot(df_for_avg,filtered_data,'avg_speed','Volumes')
    #scatter plot

    s_data = scatter_data(df_for_avg, filtered_data, 'avg_speed')
    speed_pt = scatter_src(s_data[0],s_data[1],-1,-1)
    speed_line = scatter_src(s_data[0],0,s_data[2],s_data[3])

    speed_scatter = scatter_figure('Speed')
    speed_scatter.line('x','y', source = speed_line, line_color = 'red')
    speed_scatter.circle('x', 'y', size=5, source=speed_pt)

    v_data = scatter_data(df_for_avg, filtered_data, 'avg_volume')
    v_pt = scatter_src(v_data[0],v_data[1],-1,-1)
    v_line = scatter_src(v_data[0],0,v_data[2],v_data[3])

    volume_scatter = scatter_figure('Volumes')
    volume_scatter.line('x','y', source = v_line, line_color = 'red')
    volume_scatter.circle('x', 'y', size=5, source=v_pt)

    scatter_def = Div(text="""The <b>Volume</b> and <b>Speed</b> scatter plots compare mean selected values by MHN link (AAB)
                            to mean values for all years.<br>
                            Review data points and outliers by using the selection lasso tool to select subset. <br>
                            Click the <i>Rest tool</i> to deselect points.""",
                    css_classes = ["small"], width = 850)



    corr_df = rtdap_data.loc[rtdap_data['ROADNAME'] == corridor_select.value]
    #speed_diff_vbar = (vbar_chart(corr_df,filtered_data))
    def style(p):
        p.xgrid.visible = False
        p.ygrid.visible = False
        #p.background_fill_color = None
        p.background_fill_alpha = 0.5
        p.border_fill_color = None
        p.xaxis.axis_label = "% Difference (Mean (Selection) - Mean (2015 - 2017))"
        p.yaxis.axis_label = "Frequency"

        return p

    speed_vsrc = vbar_chart_src(df_for_avg,filtered_data,'avg_speed','Speed')
    speed_diff_vbar = figure(plot_width=1000, plot_height=300, title="Speed Difference Distribution", toolbar_location="above")
    speed_diff_vbar.vbar(x='bins' , top='Speed_difference', width=1, color='navy', alpha=0.5, source = speed_vsrc)
    style(speed_diff_vbar)

    occ_vsrc = vbar_chart_src(df_for_avg,filtered_data,'avg_occupancy','Occupancy')
    occ_diff_vbar = figure(plot_width=1000, plot_height=300, title="Occupancy Difference Distribution", toolbar_location="above")
    occ_diff_vbar.vbar(x='bins' , top='Occupancy_difference', width=1, color='navy', alpha=0.5, source = occ_vsrc)
    style(occ_diff_vbar)

    vol_vsrc = vbar_chart_src(df_for_avg,filtered_data,'avg_volume','Volume')
    volume_diff_vbar = figure(plot_width=1000, plot_height=300, title="Volume Difference Distribution", toolbar_location="above")
    volume_diff_vbar.vbar(x='bins' , top='Volume_difference', width=1, color='navy', alpha=0.5, source = vol_vsrc)
    style(volume_diff_vbar)

    #p.yaxis.visible = False
    #p.xaxis.formatter = NumeralTickFormatter(format="0.f%")

    base_map = make_base_map(map_width=450,map_height=960, xaxis=None, yaxis=None,
                xrange=(-9810000, -9745000), yrange=(5130000, 5130000),plot_tools="pan,wheel_zoom,reset,save,hover")

    hwy_src = make_line_data(hwynet_shp, corridor_select.value)

    hwy_map = make_line_map(base_map, hwy_src)

    select_content =  row(
           #PANEL
           column(panel_title, panel_text, corridor_select,date_picker_start,
               date_picker_end, day_of_week, time_of_day,tod_description,
               select_data, css_classes = ["w3-sidebar", "w3-bar-block","w3-darkgrey"], width = widget_width + 70, height = 1500),
           column(css_classes=["w3-col"], width = 275 ),
          #CONTENT
           column(summary_title,
                row(Spacer(width=20),
                    column(Spacer(height=50),
                           row(column(summary_table,row(sum_tbl_def, Spacer(width=10), tbl_def)),column(Spacer(width=50)),
                           column(bar_viz_chart,height = 350),height = 350,css_classes = ["w3-panel","w3-white","w3-card-4"]),
                           Spacer(height=10),
                           column(row(volume_scatter,Spacer(width=125),speed_scatter),scatter_def,
                           css_classes = ["w3-panel","w3-white","w3-card-4"], width = 1000),
                           Spacer(height=10),
                           row(column(speed_diff_vbar,occ_diff_vbar,volume_diff_vbar), css_classes = ["w3-panel","w3-white","w3-card-4"]),
                ),
              ), css_classes=["w3-container", "w3-row-padding"]),
          column(hwy_map, css_classes = ["w3-sidebar-right","w3-panel","w3-white"],width = 500),
          css_classes = ["w3-container","w3-light-grey"], width = 2000, height = 1200)

    return select_content
