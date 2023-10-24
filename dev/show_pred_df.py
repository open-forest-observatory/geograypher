from pathlib import Path
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely import transform, Polygon
import numpy as np

from multiview_prediction_toolkit.utils.prediction_metrics import eval_confusion_matrix

from multiview_prediction_toolkit.config import DATA_FOLDER

TILE = "002"

GASCOLA_VECTOR = Path(DATA_FOLDER, "gascola", "gascola.geojson")
PRED_VECTOR = Path(DATA_FOLDER, "gascola", f"pred_{TILE}.geojson")
gascola_df = gpd.read_file(GASCOLA_VECTOR)
pred_df = gpd.read_file(PRED_VECTOR)
pred_df["class"] = gascola_df["class"]
pred_df["geometry"] = [
    transform(geom, lambda x: np.vstack((x[:, 1], x[:, 0])).T)
    for geom in pred_df["geometry"]
]

f, axs = plt.subplots(1, 2)
axs[0].set_title("True")
axs[1].set_title("Predicted")

TRAIN_BOX_000 = (-79.7915, -79.7885, 4.045e1 + 0.0064, 4.045e1 + 0.0078)
TRAIN_BOX_001 = (-79.7915, -79.7885, 4.045e1 + 0.0078, 4.045e1 + 0.0087)
TRAIN_BOX_002 = (-79.7915, -79.7885, 4.045e1 + 0.0087, 4.045e1 + 0.01)

train_boxes = {
    "000": TRAIN_BOX_000,
    "001": TRAIN_BOX_001,
    "002": TRAIN_BOX_002,
}

l, r, b, t = train_boxes[TILE]

train_box = Polygon(((l, t), (r, t), (r, b), (l, b), (l, t)))
train_box_df = gpd.GeoDataFrame(
    {"geometry": [train_box, train_box, train_box]}, crs=gascola_df.crs
)

gascola_df.plot("class", ax=axs[0])
pred_df.plot("class", ax=axs[1])


train_box_df.plot(ax=axs[0], facecolor="none")
train_box_df.plot(ax=axs[1], facecolor="none")
plt.savefig(f"vis/predicted_classes_{TILE}.png")
plt.close()

gascola_df["geometry"] = gascola_df.difference(train_box_df)
pred_df["geometry"] = pred_df.difference(train_box_df)


cf_matrix, _ = eval_confusion_matrix(
    pred_df,
    gascola_df,
    column_name="class",
    savepath=f"vis/confusion_matrix_{TILE}.png",
    normalize=True,
    normalize_by_class=True,
)

print(np.sum(cf_matrix))
print(cf_matrix)
print(np.sum(cf_matrix * np.eye(3)))
