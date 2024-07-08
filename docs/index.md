---
title: "Overview"
---
# Geograypher: Multiview Semantic Reasoning with Geospatial Data

Geograypher is a tool developed by the [Open Forest Observatory](https://openforestobservatory.org/) to help answer ecological questions using images from aerial surveys. Land managers and research ecologists are increasingly using small uncrewed aerial systems (sUAS or "drones") to survey large regions with overlapping high-resolution aerial images. This data can be used to generate predictions of some attribute, for example identifying the location of individual trees or deliniating the boundaries of different classes of vegetation. A common workflow is to take the raw images from the drone survey and generate a stiched top-down "orthomosaic" using photogrammetry software. Then, this orthomosaic is used as input to a machine learning model that predicts the attribute in question.

Geograypher was developed to support an alternative workflow where machine learning predictions are generated on individual images and then these predictions are projected into geospatial coordinates. This is beneficial because there are artifacts and errors in the orthomosaic because it is synthesized from many individual images. Additionally, there are multiple raw images of a given location which can each be used to make independent predictions. While geograypher primarily supports this "multiview" workflow, it also supports more conventional orthomosaic workflows which can be used as baseline or alternative approach.


