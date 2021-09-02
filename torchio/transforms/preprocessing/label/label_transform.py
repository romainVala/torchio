from typing import List, Dict

from ....data.image import LabelMap
from ....data.subject import Subject
from ...transform import Transform


class LabelTransform(Transform):
    """Transform that modifies label maps."""

    def get_images(self, subject: Subject) -> List[LabelMap]:
        images = subject.get_images(
            intensity_only=False,
            include=self.include,
            exclude=self.exclude,
        )
        return [im for im in images if isinstance(im, LabelMap)]

    def get_images_dict(self, subject: Subject) -> Dict[str, LabelMap]:
        images = subject.get_images_dict(
            intensity_only=False,
            include=self.include,
            exclude=self.exclude,
        )
        return {k: v for (k, v) in images.items() if isinstance(v, LabelMap)}
