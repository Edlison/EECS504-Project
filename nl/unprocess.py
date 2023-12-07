# @Author  : Edlison
# @Date    : 12/6/23 21:22
import ee
import geemap as geemap
import numpy as np
import matplotlib.pyplot as plt

ee.Authenticate()
ee.Initialize()

dataset = ee.ImageCollection('NOAA/DMSP-OLS/NIGHTTIME_LIGHTS').filter(ee.Filter.date('2013-01-01', '2013-06-01'))
nighttimeLights = dataset.select('stable_lights')
m = geemap.Map(center=[60, 20], zoom=5)
m.add_layer(nighttimeLights, {'min': 0, 'max': 63.0}, 'night light')
display(m)
thresholdedNighttimeLights = nighttimeLights.map(thresholdImage)
yourRegionOfInterest = ee.Geometry.Polygon([[[103.56191119812802, 1.2097344225742634],
                                             [104.09474811219052, 1.2097344225742634],
                                             [104.08788165711239, 1.559820812052026],
                                             [103.57564410828427, 1.5557024615990354]]])  # longtitude,latitude


def thresholdImage(image):
    threshold = 10  # Adjust this threshold value based on dataset
    lightsAboveThreshold = image.gt(threshold)
    return lightsAboveThreshold


light_intensity = thresholdedNighttimeLights.sum().reduceRegion(
    reducer=ee.Reducer.sum(),
    geometry=yourRegionOfInterest,  # Define your region of interest
    scale=1000  # Set the scale according to dataset
)

print("intensity of lights above the threshold:", light_intensity.get('stable_lights').getInfo())
