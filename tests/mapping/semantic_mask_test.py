import numpy as np

from mapping import SemanticClass, SemanticMask


def test_semantic_mask_class_mask() -> None:
    palette = {
        SemanticClass.BLACK: (0, 0, 0),
        SemanticClass.VEGETATION: (1, 1, 1),
        SemanticClass.ROAD: (2, 2, 2),
    }
    mask = np.array([0, 1, 2], dtype=np.uint8).reshape((1, -1, 1)).repeat(3, axis=-1)
    mask = SemanticMask(mask, palette)
    np.testing.assert_equal(
        mask.class_mask(SemanticClass.BLACK).mask, [[True, False, False]]
    )
    np.testing.assert_equal(
        mask.class_mask([SemanticClass.BLACK, SemanticClass.ROAD]).mask,
        [[True, False, True]],
    )
