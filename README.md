# Hackathon 2026 Aircraft LiDAR

Interactive tool to create quicklooks of AIRflows data (https://www.imktro.kit.edu/english/12875.php). AIRflows is a 5-beam Doppler Lidar system that flies on board the TUBS Cessna F406.

The tool displays one flight track on a map. A section of the flight can be selected, cut, saved and plotted without the need to understand the structure of the dataset. This tool enables visual selection instead of complicated indexing.

## Dependencies
- numpy
- pandas
- xarray
- matplotlib
- panel
- hvplot
- geoviews
- bokeh

## Tutorial

#### Start the App:
- Clone this repository or download the source code.
- Load data of one flight into the data folder or specify the path in settings.ini.
- (Optional) Adjust output paths in the settings.ini file.
- Run python app.py from your console.
- Your browser with the app should open.
- Checkout the flight path: you can zoom in and hover over the points to get the time and location of one specific wind profile).

#### Cut a section of the data:
- Click 'Select starting point'.
- Click on the starting point. Hint: zoom in to select the point specifically. If you hover over the point you get the time and the profilenumer as well as the location of the profile.
- After selecting the starting point, the profile number is written underneath the buttons.
- Click 'Select end point'.
- Click on the end point. Here you can also zoom in and hover.
- After selecting the end point, the profile numer is displayed.
- Click 'Cut dataset' to cut the dataset. If successful 'ready' is displayed.
- Click 'Save dataset' to save the cut dataset. By default, it will be saved in the folder 'processed'.

#### Display the wind profiles along the selected section:
- Click 'plot2d'.
- The figure is displayed underneath the buttons. Scroll down to see it.
- The figure shows the vertical wind on the upper panel. Arrows show wind projected into the cross-section. CAUTION: If selected section is not straight, projected arrows will be wrong!
- The lower panel shows horizontal wind speed and wind direction according to meteorological convention. 
- The figure is saved in the 'results' folder.
- Click 'plot3d' if you want to display the selected flight section in 3D (CAUTION: only works if path to DEM for topography in settings.ini is specified!).
- A new window opens. You can zoom in and rotate the figure to investigate the data in the flight section.
- The figure is saved in the 'results' folder.
  
**The created figures should only be used as quicklooks and are not intended for publication!**

#### Use other settings:
- Open settings.ini.
- Change the settings as needed.
- You can read data from other folders (change input path) or save the figures to other folders (change output path).
- You can change the settings for plotting.
- Save and close.
- Run app.py and proceed as described above.

## License:
See licence file.


