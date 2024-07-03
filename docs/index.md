---
title: "Docs home"
---

# Welcome to Geograypher documentation

Welcome to the documentation for the [Geograyher :material-arrow-top-right-bold-box-outline:](https://github.com/open-forest-observatory/multiview-mapping-toolkit).

# Conceptual workflow

Imagine you are trying to map the objects in a hypothetical region. Your world consists of three types of objects: cones, cubes, and cylinders. Cones are different shades of blue, cubes are difference shades of orange, and cylinders are different shades of green. Your landscape consists of a variety of these objects arranged randomly on a flat gray surface. You fly a drone survey and collect images of your scene, some of which are shown below.

<p align="center">
  <img alt="Image 1" src="images/texture_render_realistic_000.png" width="28%">
&nbsp; &nbsp; &nbsp; &nbsp;
  <img alt="Image 2" src="images/texture_render_realistic_001.png" width="28%">
&nbsp; &nbsp; &nbsp; &nbsp;
  <img alt="Image 2" src="images/texture_render_realistic_002.png" width="28%">
</p>

While you are there, you also do some field work and survey a small subset of your region. Field work is labor-intensive, so you can't survey the entire region your drone flew. You note down the class of the object and their location and shape in geospatial coordinates. This results in the following geospatial map.

<p align="center">
  <img alt="Field reference data" src="images/2D_map.png" width="50%">
</p>

You use structure from motion to build a 3D model of your scene and also estimate the locations that each image was taken from.

<p align="center">
  <img alt="Photogrammetry result" src="images/textured_scene_render.png" width="50%">
</p>

Up to this point, you have been following a fairly standard workflow. A common practice at this point would be to generate a top-down, 2D orthomosaic of the scene and do any prediction tasks, such as deep learning model training or inference, using this data. Instead, you decide it's important to maintain the high quality of the raw images and be able to see the sides of your objects when you are generating predictions. This is where geograypher comes in.

Using your field reference map and the 3D model from photogrammetry, you determine which portions of your 3D scene correspond to each object. This is shown below, with the colors now representing the classification label.

<p align="center">
  <img alt="Photogrammetry result" src="images/class_scene_render.png" width="50%">
</p>

Your end goal is to generate predictions on the entire region. For this, you need a machine learning model that can generate automatic predictions on your data. No one else has developed a model for your cone-cube-cylinder classification task, so you need to train your own using labeled example data. Using the mesh that is textured with the classification information from the field survey, and the pose of the camera, you can "render" the labels onto the images. They are shown below, color-coded by class.

<p align="center">
  <img alt="Image 1" src="images/class_render_flat000.png" width="28%">
&nbsp; &nbsp; &nbsp; &nbsp;
  <img alt="Image 2" src="images/class_render_flat001.png" width="28%">
&nbsp; &nbsp; &nbsp; &nbsp;
  <img alt="Image 2" src="images/class_render_flat002.png" width="28%">
</p>
These labels correspond to the images shown below.
<p align="center">
  <img alt="Image 1" src="images/texture_render_realistic_000.png" width="28%">
&nbsp; &nbsp; &nbsp; &nbsp;
  <img alt="Image 2" src="images/texture_render_realistic_001.png" width="28%">
&nbsp; &nbsp; &nbsp; &nbsp;
  <img alt="Image 2" src="images/texture_render_realistic_002.png" width="28%">
</p>

Now that you have pairs of real images and rendered labels, you can train a machine learning model to predict the class of the objects from the images. This model can be now used to generate predictions on un-labeled images. An example prediction is shown below.
<p align="center">
  <img alt="Image 1" src="images/texture_render_realistic_003.png" width="28%">
&nbsp; &nbsp; &nbsp; &nbsp;
  <img alt="Image 1" src="images/right-arrow.svg" width="10%">
&nbsp; &nbsp; &nbsp; &nbsp;
  <img alt="Image 2" src="images/class_render_flat003.png" width="28%">
</p>

To make these predictions useful, you need the information in geospatial coordinates. We again use the mesh model as an intermediate step between the image coordinates and 2D geospatial coordinates. The predictions are projected or "splatted" onto the mesh from each viewpoint.

<p align="center">
  <img alt="Image 1" src="images/class_render_flat003.png" width="45%">
&nbsp; &nbsp; &nbsp; &nbsp;
  <img alt="Image 2" src="images/projected_labels_003.png" width="45%">
</p>
<p align="center">
  <img alt="Image 1" src="images/class_render_flat004.png" width="45%">
&nbsp; &nbsp; &nbsp; &nbsp;
  <img alt="Image 2" src="images/projected_labels_004.png" width="45%">
</p>

As seen above, each prediction only captures a small region of the mesh, and cannot make any predictions about parts of the object that were occluded in the original viewpoint. Therefore, we need to aggregate the predictions from all viewpoints to have an understanding of the entire scene. This gives us added robustness, because we can tolerate some prediction errors for a single viewpoint, by choosing the most common prediction across all viewpoints of a single location. The aggregated prediction is shown below.

<p align="center">
  <img alt="Photogrammetry result" src="images/class_scene_render.png" width="50%">
</p>

Now, the final step is to transform these predictions back into geospatial coordinates.

<p align="center">
  <img alt="Photogrammetry result" src="images/2D_map.png" width="50%">
</p>