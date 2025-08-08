from uapcar import dataset_students_grades
from uapcar import Glyph


dists_students = dataset_students_grades()
glyph_students = Glyph(dists_students, 91, 181, True)
glyph_students.save_off("glyph_student_grades", False)
