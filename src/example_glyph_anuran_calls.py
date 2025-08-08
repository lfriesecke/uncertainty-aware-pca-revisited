from uapcar import dataset_anuran_calls
from uapcar import Glyph


dists_anuran = dataset_anuran_calls()
glyph_anuran = Glyph(dists_anuran, 91, 181, True)
glyph_anuran.save_off("glyph_anuran_calls", False)
