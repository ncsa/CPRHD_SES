"""Take huge detailed screenshots of Google Maps."""

from datetime import datetime
import os
import requests
import shutil
from PIL import Image


def create_map(lat_start: float, long_start: float,
               number_rows: int, number_cols: int,
               scale: float=1,outfile: str=None):
    """Create a big Google Map image from a grid of screenshots.

    ARGS:
        lat_start: Top-left coodinate to start taking screenshots.
        long_start: Top-left coodinate to start taking screenshots.
        number_rows: Number of screenshots to take for map.
        number_cols: Number of screenshots to take for map.
        scale: Percent to scale each image to reduce final resolution
            and filesize. Should be a float value between 0 and 1.
            Recommend to leave at 1 for production, and between 0.05
            and 0.2 for testing.
        outfile: If provided, the program will save the final image to
            this filepath. Otherwise, it will be saved in the current
            working directory with name 'testimg-<timestamp>.png'
    """
    if outfile:
        # Make sure the path doesn't exist, is writable, and is a .PNG
        assert not os.path.exists(outfile)
        assert os.access(os.path.dirname(os.path.abspath(outfile)), os.W_OK)
        assert outfile.upper().endswith('.PNG')
    # 2D grid of Images to be stitched together
    images = [[None for _ in range(number_cols)]
              for _ in range(number_rows)]
    # Calculate amount to shift lat/long (Magic number by trial and error)
    lat_shift = calc_latitude_shift()
    long_shift = calc_longitude_shift()
    # Number the images
    idx = 1
    for row in range(number_rows):
        for col in range(number_cols):
            latitude = lat_start + (lat_shift * row)
            longitude = long_start + (long_shift * col)
            # print('Center {}: '.format(idx) + str(latitude) + ',' + str(longitude))

            # default parameters. Get your own Google API key from
            # https://developers.google.com/maps/documentation/maps-static/get-api-key
            payload = {'center': str(latitude) + ',' + str(longitude),
                   'zoom': '20',
                   'size':'640x640',
                   'maptype':'satellite',
                   'key': '{Your API key here}'}
            r = requests.get('https://maps.googleapis.com/maps/api/staticmap', params = payload, stream=True)

            # save image to the local file
            if r.status_code == 200:
                with open('{Your path}/test-{}.png'.format(str(idx) + ':' + str(latitude) + ',' + str(longitude)), 'wb') as f:
                    r.raw.decode_content = True
                    shutil.copyfileobj(r.raw, f)
            # read image to scale later
            image = Image.open('{Your path}/test-{}.png'.format(str(idx) + ':' + str(latitude) + ',' + str(longitude)))
            images[row][col] = scale_image(image, scale)
            idx = idx + 1

    # Combine all the images into one, then save it to disk
    final = combine_images(images)
    if not outfile:
        timestamp = datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
        outfile = 'testimg-{}.png'.format(timestamp)
    final.save(outfile)

def calc_latitude_shift() -> float:
    """Return the amount to shift latitude per row of map tiles."""
    # return -0.0003173 * 2
    return -0.000325 * 2

def calc_longitude_shift() -> float:
    """Return the amount to shift longitude per column of map tiles."""
    # return 0.00042 * 2
    return 0.000428 * 2


def scale_image(image: Image, scale: float) -> Image:
    """Scale an Image by a proportion, maintaining aspect ratio."""
    width = round(image.width * scale)
    height = round(image.height * scale)
    image.thumbnail((width, height))
    return image


def combine_images(images: list) -> Image:
    """Return combined image from a grid of identically-sized images.

    images is a 2d list of Image objects. The images should
    be already sorted/arranged when provided to this function.
    """
    imgwidth = images[0][0].width
    imgheight = images[0][0].height
    newsize = (imgwidth * len(images[0]), imgheight * len(images))
    newimage = Image.new('RGB', newsize)

    # Add all the images from the grid to the new, blank image
    for rowindex, row in enumerate(images):
        for colindex, image in enumerate(row):
            location = (colindex * imgwidth, rowindex * imgheight)
            newimage.paste(image, location)

    return newimage
