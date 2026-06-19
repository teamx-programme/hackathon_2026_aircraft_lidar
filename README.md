# Hackathon 2026 Aircraft LiDAR

Interactive tool to create quicklooks of AIRflows data (https://www.imktro.kit.edu/english/12875.php). 
One flight is displayed on a map. A section of the flight can be selected, cutted, saved and plotted. 

## Dependecies
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
- Load data of one flight into the data folder.
- run python app.py from the console
- your browser with the app should open
- checkout the flight path: you can zoom in and hover over the points to get the time and location of one specific wind profile)

#### Cut a section of the data:
- Click 'Select starting point'
- Click on the starting point. Hint: zoom in to select the point specifically. If you hover over the point you get the time and the profilenumer as well as the location of the profile.
- After selecting the starting point, the profile numer is written underneath the buttons.
- Click 'Select end point'
- Click on the end point. Here you can also zoom in and hover.
- After selecting the end point, the profile numer is displayed.
- Click 'Cut dataset' to cut the dataset. If successful 'ready' is displayed.
- Click 'Save dataset' to save the cutted dataset. By default, it will be saved in the folder 'processed'.

#### Display the wind profiles along the selected section:
- Click 'plot2d'
- The figure is displayed underneath the buttons. Scroll down to see it.
- The figure shows the vertical wind on top as well as the horizontal wind speed on the bottom. 
- The figure is saved in the 'results' folder.
- Click 'plot3d' if you want to display the selected flight section in 3D.
- A new window opens. You can zoom in and rotate the figure to investigate the data in the flight section.
- The figure is saved in the 'reults' folder.

#### Use other settings:
- open settings.ini
- change the settings as needed
- you can read data from other folders (change input path) or save the figures to other folders (change output path)
- you can change the settings for plotting
- save and close
- run app.py and proceed as described above

