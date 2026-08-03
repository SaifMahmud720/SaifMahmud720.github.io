import pandas as pd
import hvplot.pandas
import panel as pn
import holoviews as hv
import numpy as np
from bokeh.models import LinearAxis, Range1d, BoxZoomTool
from pathlib import Path

# Initialize Panel extension for HoloViews/Bokeh rendering
pn.extension()

# ==============================================================================
# DATA CONFIGURATION & SCHEMA SETUP
# ==============================================================================

# Columns required from the main dataset for spatial, categoric, and performance analysis
columns_to_keep = [
    'route_id', 'speed_mph', 'Bike.Type', 'category_group', 
    'category', 'measure', 'participant', 'Filter_Keep', 'lat', 'long'
]

# Explicit data type enforcement to ensure proper string grouping and filtering
column_types = {
    'Bike.Type': str,
    'category_group': str,
    'category': str,
    'route_id': str,
    'participant': str
}

# Dynamically resolve root directory relative to this script file
script_dir = Path(__file__).resolve().parent

# ==============================================================================
# DATA IMPORT & CLEANING
# ==============================================================================

# Locate single Slickrock Route 4 CSV file
route_file = script_dir / 'Slickrock_Route_4.csv'

if not route_file.exists():
    raise FileNotFoundError(
        f"Could not find 'Slickrock_Route_4.csv' in directory:\n{script_dir}"
    )

# Read route 4 dataset with UTF-8 encoding
df = pd.read_csv(route_file, usecols=columns_to_keep, dtype=column_types, encoding='utf-8-sig')

# Locate elevation file in main folder or fallback to 'input' subfolder
elevation_file = script_dir / 'slick_rock_elevations.csv'
if not elevation_file.exists():
    elevation_file = script_dir / 'input' / 'slick_rock_elevations.csv'

if not elevation_file.exists():
    raise FileNotFoundError(
        f"Could not find 'slick_rock_elevations.csv' in repository directory or 'input' subfolder:\n{script_dir}"
    )

# Load elevation data and standardize column names
raw_alt_df = pd.read_csv(
    elevation_file,
    usecols=['MEASURE', 'ELEV'],
    encoding='utf-8-sig'
).rename(columns={'MEASURE': 'measure', 'ELEV': 'elev'})

# Clean elevation data: convert to numeric, drop NaNs, drop duplicate distance measures, and sort sequentially
raw_alt_df['measure'] = pd.to_numeric(raw_alt_df['measure'], errors='coerce')
raw_alt_df['elev'] = pd.to_numeric(raw_alt_df['elev'], errors='coerce')
raw_alt_df = raw_alt_df.dropna(subset=['measure', 'elev']).drop_duplicates(subset=['measure']).sort_values('measure')

# Clean main telemetry data: filter valid records and bike categories
df['Filter_Keep'] = pd.to_numeric(df['Filter_Keep'], errors='coerce')
df = df[df['Filter_Keep'] == 1]
df = df[df["Bike.Type"].notna() & (df["Bike.Type"] != "NA")].copy()
df['Bike.Type'] = df['Bike.Type'].astype(str)

# Ensure valid non-empty route IDs
df = df[df['route_id'].notna() & (df['route_id'] != "NA") & (df['route_id'].str.strip() != "")]

# Coerce spatial and measurement values to numeric
for col in ['measure', 'lat', 'long']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.dropna(subset=['measure', 'lat', 'long'])

# Convert Geographic Coordinates (Lat/Long) to Spherical Web Mercator Projection (meters) for OSM tiling
r_earth = 6378137.0
df['merc_x'] = df['long'] * (r_earth * np.pi / 180.0)
df['merc_y'] = np.log(np.tan((90.0 + df['lat']) * np.pi / 360.0)) * r_earth

# ==============================================================================
# COLOR MAPPING & WIDGET STYLING
# ==============================================================================

# Hex color definitions per terrain and feature category
category_colors = {
    'Climb I: Normal': '#fd8d3c',                    
    'Climb II: Technical / Off-Camber': "#d94801",    
    'Climb III: Steep': "#8c2d04",                    
    'Descent I: Normal': "#4e7e99",                   
    'Descent II: Technical': '#2171b5',               
    'Technical I: Rough / Loose': "#d4b9da",          
    'Technical II: Features / Off-Camber': "#df65b0", 
    'Tight Turn I: Single': "#a1d99b",                
    'Tight Turn II: Multi-S-Turn': "#41ab5d",          
    'Tight Turn III: Single - Follows Descent': "#238b45", 
    'Tight Turn IV: Multi-S-Turn - Follows Descent': "#005a32", 
    'Transition In (Down)': '#6a51a3',                
    'Transition Out (Up)': '#4a1486',                 
    'Narrow': "#999600",                              
    'Sand Wash': "#E5FF00"                            
}

# Dynamic CSS injection to color-code dropdown menu options matching terrain category colors
widget_stylesheet = "".join([
    f"""
    select option[value="{cat}"] {{
        background-color: {hex_c} !important;
        color: black !important;
        font-weight: bold !important;
        padding: 4px 6px;
    }}
    select option[value="{cat}"]:checked {{
        background: {hex_c} linear-gradient(0deg, rgba(0,0,0,0.15) 0%, rgba(0,0,0,0.15) 100%) !important;
        color: black !important;
    }}
    """
    for cat, hex_c in category_colors.items()
])

# ==============================================================================
# ANALYTICAL HELPER FUNCTIONS
# ==============================================================================

def get_route_delta_df(route_id):
    """
    Computes spatial speed differences (eMTB vs. Conventional MTB) along a given route.
    Groups telemetry spatially by measure and category to align both bike types.
    """
    sub_df = df[(df['route_id'] == route_id) & (df['Filter_Keep'] == 1)]
    if sub_df.empty:
        return pd.DataFrame()
    
    # Calculate mean speeds per spatial segment for each bike type
    conv = sub_df[sub_df['Bike.Type'] == 'Conventional MTB'].groupby(['measure', 'merc_x', 'merc_y', 'category'])['speed_mph'].mean().reset_index()
    elec = sub_df[sub_df['Bike.Type'] == 'Electric eMTB'].groupby(['measure', 'merc_x', 'merc_y', 'category'])['speed_mph'].mean().reset_index()
    
    # Merge both bike dataset profiles on spatial measure & category
    merged = pd.merge(elec, conv, on=['measure', 'category'], suffixes=('_electric', '_conventional'))
    merged['speed_difference_mph'] = merged['speed_mph_electric'] - merged['speed_mph_conventional']
    
    # Midpoint spatial coordinates for mapping merged records
    merged['merc_x'] = (merged['merc_x_electric'] + merged['merc_x_conventional']) / 2.0
    merged['merc_y'] = (merged['merc_y_electric'] + merged['merc_y_conventional']) / 2.0
    return merged.sort_values(by='measure')

# ==============================================================================
# UI CONTROLS & WIDGET INITIALIZATION
# ==============================================================================

route_options = sorted(list(df['route_id'].unique()))
all_categories = sorted(list(df['category'].dropna().unique()))

# Primary Filter Controls
route_dropdown = pn.widgets.Select(name="Select Route ID", options=route_options, value=route_options[0] if route_options else '4')
bike_dropdown = pn.widgets.Select(name='Filter Bike Type', options=['Both Bike Types', 'Conventional MTB Only', 'Electric eMTB Only'], value='Both Bike Types')
measure_slider = pn.widgets.RangeSlider(name='Measure Range Along Track', start=0.0, end=100.0, value=(0.0, 100.0), step=1.0)
reset_range_btn = pn.widgets.Button(name='Reset Measure Range', button_type='default')
set_zoom_btn = pn.widgets.Button(name='Set Zoom', button_type='primary')

# Plot Customization & Display Toggles
altitude_toggle = pn.widgets.Checkbox(name='Overlay Altitude Profiles', value=False)
smooth_altitude_toggle = pn.widgets.Checkbox(name='Smooth Altitude Curve', value=False)
mono_delta_toggle = pn.widgets.Checkbox(name='Monochromatic Delta Graph', value=False)
show_mean_speeds_toggle = pn.widgets.Checkbox(name='Show Mean Speed Lines', value=False)
search_threshold = pn.widgets.FloatInput(name='Delta Search Threshold (mph)', value=15.0, step=0.5)
search_direction = pn.widgets.Select(name='Find Delta Values', options=['Greater than Threshold', 'Less than -Threshold'], value='Greater than Threshold')
map_mode_toggle = pn.widgets.Select(name='Map Heatmap Color Coding', options=['Raw Speeds (Overall)', 'Speed Deltas (eMTB - Conventional)'], value='Speed Deltas (eMTB - Conventional)')
fullscreen_map_toggle = pn.widgets.Toggle(name='Map Fullscreen', button_type='warning', value=False)
category_selector = pn.widgets.MultiSelect(name='Select Terrain Categories to Display', options=all_categories, value=all_categories, size=min(len(all_categories), 12), stylesheets=[widget_stylesheet])
select_all_btn = pn.widgets.Button(name='Select All Categories', button_type='default')

# Speed Delta Range Inclusion Filter
include_delta_slider = pn.widgets.RangeSlider(name='Filter Speed Delta Range (mph)', start=-20.0, end=20.0, value=(-20.0, 20.0), step=0.1)

# Dynamic state tracking variables
click_coordinates = []
current_plot_zoom = {'x_range': None}

# ==============================================================================
# WIDGET CALLBACKS & DYNAMIC STATE SYNCHRONIZATION
# ==============================================================================

def rangex_hook(plot, element):
    """
    Bokeh hook to capture active x-axis zoom range events from interactive plots.
    """
    fig = plot.state
    if hasattr(fig, 'x_range') and fig.x_range is not None:
        def update_bounds(attr, old, new):
            if fig.x_range.start is not None and fig.x_range.end is not None:
                current_plot_zoom['x_range'] = (fig.x_range.start, fig.x_range.end)

        fig.x_range.on_change('start', update_bounds)
        fig.x_range.on_change('end', update_bounds)

def apply_zoom_to_slider(event):
    """
    Applies current interactive plot zoom bounds directly to the measure range slider.
    """
    x_rng = current_plot_zoom['x_range']
    if x_rng is not None:
        zoom_min, zoom_max = x_rng
        s_start, s_end = measure_slider.start, measure_slider.end
        new_min = max(s_start, float(zoom_min))
        new_max = min(s_end, float(zoom_max))
        
        if new_min < new_max:
            measure_slider.value = (round(new_min, 2), round(new_max, 2))

set_zoom_btn.on_click(apply_zoom_to_slider)

@pn.depends(route_dropdown, measure_slider, watch=True)
def update_include_delta_bounds(route_id, measure_range):
    """
    Dynamically adjusts delta slider min/max ranges based on selected route and measure segment.
    """
    m_low, m_high = measure_range
    delta_df = get_route_delta_df(route_id)
    
    if not delta_df.empty:
        sub_df = delta_df[(delta_df['measure'] >= m_low) & (delta_df['measure'] <= m_high)]
        if not sub_df.empty:
            d_min = round(float(sub_df['speed_difference_mph'].min()), 1)
            d_max = round(float(sub_df['speed_difference_mph'].max()), 1)
            
            if d_min == d_max:
                d_min -= 1.0
                d_max += 1.0
                
            include_delta_slider.param.update(
                start=d_min, 
                end=d_max, 
                value=(d_min, d_max)
            )

def update_slider_bounds(selected_route):
    """
    Resets slider bounds and terrain categories whenever a new route is selected.
    """
    click_coordinates.clear()
    current_plot_zoom['x_range'] = None
    sub_df = df[(df['route_id'] == selected_route) & (df['Filter_Keep'] == 1)]
    if not sub_df.empty:
        route_min = float(sub_df['measure'].min())
        route_max = float(sub_df['measure'].max())
        
        measure_slider.param.update(
            start=route_min,
            end=route_max,
            value=(route_min, route_max)
        )

        new_cats = sorted(list(sub_df['category'].dropna().unique()))
        category_selector.options = new_cats
        category_selector.value = new_cats

# Attach dynamic update observers to control widgets
pn.bind(update_slider_bounds, selected_route=route_dropdown, watch=True)
reset_range_btn.on_click(lambda e: update_slider_bounds(route_dropdown.value))
select_all_btn.on_click(lambda e: setattr(category_selector, 'value', category_selector.options))

# Initial execution to set slider limits
update_slider_bounds(route_dropdown.value)

# ==============================================================================
# GEOGRAPHIC MAP RENDERING & INTERACTION STREAMS
# ==============================================================================

aspect_box_zoom = BoxZoomTool(match_aspect=True)
map_tap_stream = hv.streams.Tap(transient=True)

def render_dynamic_map_points(route_id, bike_filter, measure_range, include_delta_range, map_mode, chosen_categories):
    """
    Renders map scatter points dynamically based on raw speeds or eMTB delta values.
    """
    m_low, m_high = measure_range
    e_low, e_high = include_delta_range

    if map_mode == 'Raw Speeds (Overall)':
        map_df = df[(df['route_id'] == route_id) & (df['Filter_Keep'] == 1)].copy()
        
        # Determine global scale min/max for raw speeds across route
        raw_cmin = float(map_df['speed_mph'].min()) if not map_df.empty else 0.0
        raw_cmax = float(map_df['speed_mph'].max()) if not map_df.empty else 30.0
        
        map_df = map_df[(map_df['measure'] >= m_low) & (map_df['measure'] <= m_high)]

        if chosen_categories:
            map_df = map_df[map_df['category'].isin(chosen_categories)]

        if bike_filter == 'Conventional MTB Only':
            map_df = map_df[map_df['Bike.Type'] == 'Conventional MTB']
        elif bike_filter == 'Electric eMTB Only':
            map_df = map_df[map_df['Bike.Type'] == 'Electric eMTB']

        if map_df.empty:
            points = hv.Points([], kdims=['merc_x', 'merc_y'], vdims=['category', 'speed_mph', 'measure']).opts(clabel="Speed (mph)")
        else:
            points = map_df.hvplot.points(
                x='merc_x', y='merc_y', geo=False,
                vdims=['category', 'speed_mph', 'measure'],
                color='speed_mph', cmap='viridis', alpha=0.8, size=6,
                clabel="Speed (mph)", hover_cols=['category', 'speed_mph', 'measure']
            ).redim.range(speed_mph=(raw_cmin, raw_cmax))

    else:
        full_delta_df = get_route_delta_df(route_id)
        if full_delta_df.empty:
            points = hv.Points([], kdims=['merc_x', 'merc_y'], vdims=['category', 'speed_difference_mph', 'measure', 'point_alpha']).opts(clabel="Delta Speed (mph)")
        else:
            delta_cmin = float(full_delta_df['speed_difference_mph'].min())
            delta_cmax = float(full_delta_df['speed_difference_mph'].max())

            delta_df = full_delta_df[(full_delta_df['measure'] >= m_low) & (full_delta_df['measure'] <= m_high)].copy()
            
            if chosen_categories:
                delta_df = delta_df[delta_df['category'].isin(chosen_categories)]

            if e_low < e_high:
                delta_df = delta_df[delta_df['speed_difference_mph'].between(e_low, e_high)]
            
            if delta_df.empty:
                points = hv.Points([], kdims=['merc_x', 'merc_y'], vdims=['category', 'speed_difference_mph', 'measure', 'point_alpha']).opts(clabel="Delta Speed (mph)")
            else:
                # Fade out small delta variations (-2 to 2 mph) to highlight significant speed differences
                delta_df['point_alpha'] = (~delta_df['speed_difference_mph'].between(-2, 2)).map({True: 0.85, False: 0.1})

                points = delta_df.hvplot.points(
                    x='merc_x', y='merc_y', geo=False,
                    vdims=['category', 'speed_difference_mph', 'measure', 'point_alpha'],
                    color='speed_difference_mph', cmap='coolwarm',
                    alpha='point_alpha', size=7,
                    clabel="Delta (eMTB - Conv) mph", hover_cols=['category', 'speed_difference_mph', 'measure']
                ).redim.range(speed_difference_mph=(delta_cmin, delta_cmax))

    return points.opts(
        data_aspect=1,
        frame_width=500,
        frame_height=650,
        default_tools=['pan', 'wheel_zoom', aspect_box_zoom, 'reset', 'save'],
        active_tools=['wheel_zoom']
    )

def handle_map_clicks(x, y, route_id):
    """
    Tracks dual map tap gestures to automatically zoom the measure slider to the clicked region.
    """
    if x is None or y is None: 
        return 
        
    click_coordinates.append((x, y))
    
    # Process range selection when two points are tapped on the map
    if len(click_coordinates) == 2:
        map_df = df[(df['route_id'] == route_id) & (df['Filter_Keep'] == 1)].copy()
        if not map_df.empty:
            pt1_x, pt1_y = click_coordinates[0]
            pt2_x, pt2_y = click_coordinates[1]
            
            # Find closest trajectory records to clicked coordinates
            dist1 = (map_df['merc_x'] - pt1_x)**2 + (map_df['merc_y'] - pt1_y)**2
            dist2 = (map_df['merc_x'] - pt2_x)**2 + (map_df['merc_y'] - pt2_y)**2
            
            m1 = map_df.loc[dist1.idxmin(), 'measure']
            m2 = map_df.loc[dist2.idxmin(), 'measure']
            
            low_m, high_m = sorted([float(m1), float(m2)])
            s_start, s_end = measure_slider.start, measure_slider.end
            measure_slider.value = (max(s_start, low_m), min(s_end, high_m))
            
        click_coordinates.clear()

def render_click_marker(x, y):
    """
    Renders visual indicator pin for the first tap coordinate on the map.
    """
    if len(click_coordinates) == 1:
        c_x, c_y = click_coordinates[0]
        return hv.Points([(c_x, c_y)], kdims=['merc_x', 'merc_y']).opts(
            color='red', size=14, marker='triangle', line_color='black', line_width=1.5, framewise=False
        )
    return hv.Points([], kdims=['merc_x', 'merc_y']).opts(framewise=False)

# Build map dynamic layers and attach user interactions
gis_map_layer = hv.DynamicMap(
    pn.bind(render_dynamic_map_points, 
            route_id=route_dropdown, 
            bike_filter=bike_dropdown, 
            measure_range=measure_slider,
            include_delta_range=include_delta_slider,
            map_mode=map_mode_toggle,
            chosen_categories=category_selector)
)

marker_layer = hv.DynamicMap(render_click_marker, streams=[map_tap_stream])
map_tap_stream.source = gis_map_layer
pn.bind(handle_map_clicks, x=map_tap_stream.param.x, y=map_tap_stream.param.y, route_id=route_dropdown, watch=True)

# Overlay points and selection markers onto OpenStreetMap basemap
composite_map_view = (hv.element.tiles.OSM() * gis_map_layer * marker_layer).opts(
    data_aspect=1,
    frame_width=500,
    frame_height=650,
    framewise=False
)

# ==============================================================================
# MAIN ANALYTICAL DASHBOARD PIPELINE
# ==============================================================================

@pn.depends(route_dropdown, measure_slider, bike_dropdown, altitude_toggle, smooth_altitude_toggle, 
            category_selector, mono_delta_toggle, show_mean_speeds_toggle, search_threshold, search_direction)
def update_analysis_callback(route_id, measure_range, bike_filter, show_altitude, smooth_altitude, 
                              chosen_categories, mono_color, show_mean_speeds, threshold_val, direction):
    """
    Main dynamic plotting function that updates speed distribution, terrain ribbons,
    elevation curves, and speed delta profiles in response to user inputs.
    """
    full_route_df = df[(df['route_id'] == route_id) & (df['Filter_Keep'] == 1)]
    if full_route_df.empty:
        return pn.pane.Alert("No records found for this route.", alert_type='warning')

    route_min_m = full_route_df['measure'].min()
    route_max_m = full_route_df['measure'].max()
    
    # Generate regular spatial grid for continuous elevation interpolation
    grid_measures = np.linspace(route_min_m, route_max_m, num=1000)
    path_df = pd.DataFrame({'measure': grid_measures})
    
    known_geometry = full_route_df[['measure', 'category']].copy().drop_duplicates(subset=['measure']).sort_values(by='measure')
    path_df = pd.merge_asof(path_df, known_geometry, on='measure', direction='nearest')
    
    # Process altitude profile (raw interpolation or rolling mean smooth)
    if smooth_altitude:
        path_df['elev'] = np.interp(path_df['measure'], raw_alt_df['measure'], raw_alt_df['elev'])
        path_df['elev'] = path_df['elev'].rolling(window=200, min_periods=1, center=True).mean()
    else:
        path_df = pd.merge_asof(path_df, raw_alt_df, on='measure', direction='nearest')
        path_df['elev'] = path_df['elev'].interpolate(method='linear')
    
    m_low, m_high = measure_range
    base_route_df = full_route_df[(full_route_df['measure'] >= m_low) & (full_route_df['measure'] <= m_high)]
    
    # Apply bike type filters
    if bike_filter == 'Conventional MTB Only':
        base_route_df = base_route_df[base_route_df['Bike.Type'] == 'Conventional MTB']
    elif bike_filter == 'Electric eMTB Only':
        base_route_df = base_route_df[base_route_df['Bike.Type'] == 'Electric eMTB']

    if base_route_df.empty:
        return pn.pane.Alert("No records found for this bike filter selection.", alert_type='warning')
    
    # Apply category selection filters
    if chosen_categories:
        base_route_df = base_route_df[base_route_df['category'].isin(chosen_categories)]

    if base_route_df.empty:
        return pn.pane.Alert("No matching scatter points found for the selected terrain categories.", alert_type='warning')

    # --------------------------------------------------------------------------
    # Graph 1: Rider Speed Scatter Distribution Plot
    # --------------------------------------------------------------------------
    scatter_plot = base_route_df.hvplot.scatter(
        x='measure', y='speed_mph', by='Bike.Type', 
        hover_cols=['category'], title=f'Rider Speed Distributions (Route: {route_id})',
        xlabel='', ylabel='Speed (mph)', legend='top_right', grid=True,
        size=8, height=330, width=1200, alpha=0.6, hover=False
    )
    
    # Overlay rolling mean speed trends if toggled on
    if show_mean_speeds:
        mean_curves = []
        for b_type, color in [('Conventional MTB', '#1f77b4'), ('Electric eMTB', '#ff7f0e')]:
            bike_sub = base_route_df[base_route_df['Bike.Type'] == b_type]
            if not bike_sub.empty:
                means_df = bike_sub.groupby('measure')['speed_mph'].mean().reset_index().sort_values(by='measure')
                means_df['rolling_speed'] = means_df['speed_mph'].rolling(window=15, min_periods=1, center=True).mean()
                mean_curve = hv.Curve(means_df, kdims=['measure'], vdims=['rolling_speed'], label=f'Mean {b_type}').opts(
                    color=color, line_width=3.5, line_dash='solid', alpha=0.9
                )
                mean_curves.append(mean_curve)
        if mean_curves:
            scatter_plot = scatter_plot * hv.Overlay(mean_curves)

    graph1 = scatter_plot.opts(
        hv.opts.Scatter(
            default_tools=['xpan', 'xwheel_zoom', 'box_zoom', 'reset', 'save'],
            active_tools=['xwheel_zoom'],
            hover_tooltips=[
                ('Distance (Measure)', '$x'),
                ('Speed (mph)', '$y'),
                ('Category', '@category'),
                ('Bike Type', '@{Bike.Type}')
            ],
            color=hv.dim('Bike.Type').categorize(
                {'Conventional MTB': "#78bef0", 'Electric eMTB': '#fdbf6f'}, 
                default='gray'
            )
        )
    )

    # Establish padded speed bounds for y-axis
    speed_min = float(base_route_df['speed_mph'].min())
    speed_max = float(base_route_df['speed_mph'].max())
    speed_pad = (speed_max - speed_min) * 0.10 if speed_max != speed_min else 2.0
    speed_bounds = (max(0, speed_min - speed_pad), speed_max + speed_pad)
    graph1 = graph1.opts(hv.opts.Scatter(ylim=speed_bounds))

    # Compute elevation display bounds
    active_elev_df = path_df[(path_df['measure'] >= m_low) & (path_df['measure'] <= m_high)]
    if not active_elev_df.empty:
        elev_min = float(active_elev_df['elev'].min())
        elev_max = float(active_elev_df['elev'].max())
        elev_pad = (elev_max - elev_min) * 0.05 if elev_max != elev_min else 10.0
        elev_range_bounds = (elev_min - elev_pad, elev_max + elev_pad)
    else:
        elev_range_bounds = (0, 100)

    def extra_y_axis_hook(plot, element):
        """
        Bokeh hook to append a dynamic second Y-axis for Elevation profile overlay.
        """
        fig = plot.state
        if 'elevation_axis' not in fig.extra_y_ranges:
            fig.extra_y_ranges['elevation_axis'] = Range1d(start=elev_range_bounds[0], end=elev_range_bounds[1])
            fig.add_layout(LinearAxis(y_range_name='elevation_axis', axis_label='Elevation (ft)'), 'right')
        else:
            fig.extra_y_ranges['elevation_axis'].start = elev_range_bounds[0]
            fig.extra_y_ranges['elevation_axis'].end = elev_range_bounds[1]
            
        for renderer in plot.handles.values():
            if hasattr(renderer, 'glyph') and hasattr(renderer, 'y_range_name'):
                renderer.y_range_name = 'elevation_axis'

    # Build Elevation Segment Lines
    if len(path_df) > 1:
        path_df['x0'] = path_df['measure']
        path_df['y0'] = path_df['elev']          
        path_df['x1'] = path_df['x0'].shift(-1)
        path_df['y1'] = path_df['elev'].shift(-1) 
        path_df = path_df.dropna(subset=['x1', 'y1'])
        
        altitude_composite = hv.Overlay()
        backdrop_gray = hv.Segments(path_df, kdims=['x0', 'y0', 'x1', 'y1']).opts(
            color='gray', line_width=2.5, line_alpha=0.6, line_dash='dotted'
        )
        altitude_composite *= backdrop_gray
        
        # Match trajectory segment positions to highlight active data regions
        data_measures = np.sort(base_route_df['measure'].unique())
        if len(data_measures) > 0:
            idx = np.searchsorted(data_measures, path_df['x0'].values)
            idx = np.clip(idx, 0, len(data_measures) - 1)
            left_idx = np.clip(idx - 1, 0, len(data_measures) - 1)
            close_right = np.abs(data_measures[idx] - path_df['x0'].values) <= 0.1
            close_left = np.abs(data_measures[left_idx] - path_df['x0'].values) <= 0.1
            presence_mask = close_right | close_left
            active_segments_df = path_df[presence_mask]
        else:
            active_segments_df = pd.DataFrame()
        
        if not active_segments_df.empty:
            colored_layer = hv.Segments(active_segments_df, kdims=['x0', 'y0', 'x1', 'y1'], vdims=['category']).opts(
                color=hv.dim('category').categorize(category_colors, default='gray'),
                line_width=4.5, line_alpha=1.0, line_dash='solid'
            )
            altitude_composite *= colored_layer
            
        altitude_composite = altitude_composite.opts(hv.opts.Segments(hooks=[extra_y_axis_hook]))
            
        altitude_gray_base = hv.Segments(path_df, kdims=['x0', 'y0', 'x1', 'y1']).opts(
            color='gray', line_width=2.0, line_alpha=0.45, line_dash='dotted',
            hooks=[extra_y_axis_hook] 
        )

        if show_altitude:
            graph1 = graph1 * altitude_composite

    # Bind range listener hook
    graph1 = graph1.opts(hooks=[rangex_hook])

    # --------------------------------------------------------------------------
    # Graph 2: Categorical Terrain Ribbon Component
    # --------------------------------------------------------------------------
    strip_df = base_route_df[['measure', 'category']].copy().drop_duplicates(subset=['measure']).sort_values(by='measure')
    strip_df['baseline_y'] = 0

    ribbon_bar = hv.Scatter(strip_df, kdims=['measure'], vdims=['baseline_y', 'category']).opts(
        color=hv.dim('category').categorize(category_colors, default='gray'),
        marker='square', size=16, alpha=0.95, height=95, width=1200,
        title='Terrain Category Ribbon', xlabel='', yaxis=None, ylim=(-0.4, 0.6),
        default_tools=['xpan', 'xwheel_zoom', 'reset'], active_tools=['xwheel_zoom']
    )

    # Calculate segment midpoints to place category text labels centered within ribbons
    strip_df['segment_id'] = (strip_df['category'] != strip_df['category'].shift()).cumsum()
    label_df = strip_df.groupby(['segment_id', 'category'])['measure'].median().reset_index()
    label_df['baseline_y'] = 0.0

    text_labels = hv.Labels(label_df, kdims=['measure', 'baseline_y'], vdims=['category']).opts(
        text_font_size='9pt', text_color='black', text_font_style='bold', text_align='center', yoffset=18
    )

    graph2 = ribbon_bar * text_labels

    # --------------------------------------------------------------------------
    # Graph 3: Comparative Speed Delta Analysis Plot
    # --------------------------------------------------------------------------
    if bike_filter != 'Both Bike Types':
        raw_layout = (graph1 + graph2).cols(1)
    else:
        merged_df = get_route_delta_df(route_id)
        if merged_df.empty:
            graph3 = hv.Div("<div style='padding:10px;'>No perfect spatial overlaps found between bike types.</div>")
        else:
            merged_df = merged_df[(merged_df['measure'] >= m_low) & (merged_df['measure'] <= m_high)]
            
            if chosen_categories:
                merged_df = merged_df[merged_df['category'].isin(chosen_categories)]

            if merged_df.empty:
                graph3 = hv.Div("<div style='padding:10px;'>No data points match selected categories.</div>")
            else:
                diff_scatter = merged_df.hvplot.scatter(
                    x='measure', y='speed_difference_mph', by='category',  
                    title='Speed Delta (Electric - Conventional)', xlabel='Measure (Distance Along Route)', 
                    ylabel='Delta (mph)', legend=False, size=10, height=310, width=1200,
                    alpha=0.55, grid=True, hover=False 
                )
                
                zero_line = hv.HLine(0).opts(color='black', line_dash='dashed', line_width=1.5)
                
                # Flag speed delta maxima/minima exceeding search threshold criteria
                if direction == 'Greater than Threshold':
                    target_points = merged_df[merged_df['speed_difference_mph'] > threshold_val]
                else:
                    target_points = merged_df[merged_df['speed_difference_mph'] < -threshold_val]

                if not target_points.empty:
                    maxima_highlights = hv.Scatter(
                        target_points, kdims=['measure'], vdims=['speed_difference_mph']
                    ).opts(
                        color='red', marker='circle', size=4, alpha=0.9, line_color='black', line_width=1.5
                    )
                    diff_scatter = diff_scatter * maxima_highlights

                delta_min = float(merged_df['speed_difference_mph'].min())
                delta_max = float(merged_df['speed_difference_mph'].max())
                delta_pad = (delta_max - delta_min) * 0.10 if delta_max != delta_min else 2.0
                delta_bounds = (delta_min - delta_pad, delta_max + delta_pad)

                delta_color = 'dodgerblue' if mono_color else hv.dim('category').categorize(category_colors, default='gray')

                graph3 = (diff_scatter * zero_line).opts(
                    hv.opts.Scatter(
                        default_tools=['xpan', 'xwheel_zoom', 'box_zoom', 'reset', 'save'],
                        active_tools=['xwheel_zoom'],
                        color=delta_color,
                        ylim=delta_bounds, 
                        hover_tooltips=[
                            ('Distance (Measure)', '$x'),
                            ('Speed Delta (mph)', '$y'),
                            ('Category', '@category')
                        ]
                    )
                )
                
                if show_altitude and len(path_df) > 1:
                    graph3 = graph3 * altitude_gray_base
                
        raw_layout = (graph1 + graph2 + graph3).cols(1)

    # Enforce synchronized shared horizontal axes (x-axis linked zoom/pan)
    fixed_layout = raw_layout.options({
        'Layout': {'shared_axes': True},
        'Scatter': {'xlim': (m_low, m_high)},
        'Overlay': {'xlim': (m_low, m_high)},
        'Segments': {'xlim': (m_low, m_high)},
        'Labels': {'xlim': (m_low, m_high)}
    })
    
    return fixed_layout

# ==============================================================================
# VIEW LAYOUT MANAGER & DASHBOARD ASSEMBLY
# ==============================================================================

@pn.depends(fullscreen_map_toggle)
def reactive_layout_manager(fullscreen_active):
    """
    Switches dynamic screen layout between side-by-side analysis view and full-width map mode.
    """
    if fullscreen_active:
        return pn.Column(
            pn.pane.Markdown("### 🗺️ Slickrock Trail Geographic View (Fullscreen Mode)"),
            composite_map_view,
            sizing_mode='stretch_both'
        )
    return pn.Row(
        update_analysis_callback,
        pn.Column(
            pn.pane.Markdown("### 🗺️ Slickrock Trail Map"),
            composite_map_view
        )
    )

# Assemble overall Panel web dashboard template
dashboard = pn.template.FastListTemplate(
    title='Slickrock Trail Rider Speed Analysis Dashboard',
    sidebar=[
        fullscreen_map_toggle,
        pn.layout.Divider(),
        route_dropdown, 
        bike_dropdown,
        measure_slider,
        pn.Row(set_zoom_btn, reset_range_btn),
        pn.layout.Divider(),
        pn.pane.Markdown("Map Options"),
        include_delta_slider, 
        map_mode_toggle,
        pn.layout.Divider(),
        pn.pane.Markdown("Overlay Options"),
        altitude_toggle,
        smooth_altitude_toggle,
        show_mean_speeds_toggle,
        mono_delta_toggle,
        pn.layout.Divider(),
        pn.pane.Markdown("Delta Maxima Search"),
        search_direction,
        search_threshold,
        pn.layout.Divider(),
        category_selector,
        select_all_btn 
    ], 
    main=[reactive_layout_manager]
)

# Launch the interactive server dashboard application
dashboard.show()