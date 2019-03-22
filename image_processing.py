# -------------------------------------------------------------------------------
# Image Preprocessing (Blurring, Noise Removal, Binarization, Deskewing)
# -------------------------------------------------------------------------------

# Noise Removal: https://docs.opencv.org/3.3.1/d5/d69/tutorial_py_non_local_means.html
# Deskewing: https://www.pyimagesearch.com/2017/02/20/text-skew-correction-opencv-python/
# Binarization + Blurring (Otsu): https://docs.opencv.org/3.3.1/d7/d4d/tutorial_py_thresholding.html

# ============ Read Image ============
import cv2
from numpy import ndarray


def prepare_image(file_name: str) -> ndarray:

    img: ndarray = cv2.imread(file_name, 0)

    # remove noise
    img: ndarray = cv2.fastNlMeansDenoising(img, None, 10, 7, 21)

    # thresholding
    _, img = cv2.threshold(img, 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imwrite('binarized.jpg', img)
    return img
