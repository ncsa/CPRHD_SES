from satellite_map_creator import create_map

def test_calibration():
    """Quick single screenshot of Philly Art Museum for calibration.
    """
    create_map(
        lat_start=39.9644273,
        long_start=-75.1801129,
        number_rows=1,
        number_cols=1,
        scale=0.5,
        outfile='huge_gmap_calibration_0.5.png',
    )
    create_map(
        lat_start=39.9644273,
        long_start=-75.1801129,
        number_rows=1,
        number_cols=1,
        scale=1,
        outfile='huge_gmap_calibration_1.png',
    )


def test_small_area():
    """Small 3x3 grid of images to test combining images.
    Takes about 5 seconds to run. 5.4 MB.
    """
    create_map(
        lat_start = 40.724378,
        long_start= -74.010252,
        number_rows=3,
        number_cols=3,
        scale = 1,
        outfile='huge_gmap_small_area.png',
    )

def test_large_area():
    """large 10x10 grid of images to test combining images.
    Takes about 43 seconds to run. 63.4 MB.
    """
    create_map(
        lat_start = 40.724378,
        long_start= -74.010252,
        number_rows=10,
        number_cols=10,
        scale = 1,
        outfile='huge_gmap_large_area.png',
    )
# Test cases
test_calibration()
test_small_area()
test_large_area()
