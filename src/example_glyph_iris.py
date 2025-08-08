from uapcar import dataset_iris
from uapcar import Glyph


dists_iris = dataset_iris()
glyph_iris = Glyph(dists_iris, 91, 181, True)
glyph_iris.save_off("glyph_iris", False)
