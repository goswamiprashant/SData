from setup_logging import setup_logging
import logging
import os
import numpy as np
from PIL import Image
import base64
import io
import pytesseract


def base64_to_image(enc_str):
    """
    Decodes a base64 string to an image and returns it as a Numpy array.
    The image will be resized using OpenCV to a resolution of 224x224 pixels.
    """
    dec_str = base64.b64decode(enc_str)
    img = Image.open(io.BytesIO(dec_str))
    img_arr = np.asarray(img)

    return img_arr


class Deployment:

    def __init__(self):
        """
        Initialisation method for the deployment. This will be called at start-up of the model in UbiOps.

        :param str base_directory: absolute path to the directory where this file is located.
        """

        setup_logging()
        logging.info("Initialising OCR model")


    def request(self, data):
        """
        Method for model requests, called for every individual request

        :param dict data: dictionary with the model data. In this case it will hold a key 'photo' with a base64 string
        as value.
        :return dict prediction: JSON serializable dictionary with the output fields as defined on model creation
        """

        logging.info("Processing model request")

        # Convert the base64 string input to a Numpy array of the right format. Using the function defined above.
        photo_data = base64_to_image(data['photo'])

        extractedInformation = pytesseract.image_to_string(photo_data)


        # Here we return a JSON with the estimated age as integer
        return {'output': extractedInformation}
