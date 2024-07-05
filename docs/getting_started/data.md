## Example data

The public example data is in `data/example_Emerald_Point_data` . You can run notebooks in the `examples` folder to see how to interact with this data. You can download this data using Google Drive from this [folder](https://drive.google.com/drive/folders/1gs5MkutQJEfg7tVnv01gzrf9NisAO5AT?usp=drive_link). Once you've downloaded it, extract it into the `data` folder.

## Using your own data

If you have a Metashape scene with the location of cameras, a mesh, and geospatial information, you can likely use geograypher. If you are using the Metashape GUI, you must do an important step before exporting the mesh model. Metashape stores the mesh in an arbitrary coordinate system that's optimized for viewing and will export it as such. To fix this, in the Metashape GUI you need to do `Model->Transform Object->Reset Transform` , then save the mesh with the local coordinates option. The cameras can be exported without any special considerations.

You can also use our scripted workflow for running Metashape, [automate-metashape](https://github.com/open-forest-observatory/automate-metashape). The cameras and the `local` mesh export will be properly formatted for use with geograypher.