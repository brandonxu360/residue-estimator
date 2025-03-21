from unittest.mock import patch
from src.data_utils import get_file_paths, get_valid_data

### TESTING SETUP ###

# Sample data for testing
SAMPLE_IMAGE_PATHS = [
    "/path/to/images/IMG_0766.jpg",
    "/path/to/images/IMG_0767.jpg",
]

SAMPLE_LABEL_PATHS = [
    "/path/to/labels/IMG_0766_res.tif",
    "/path/to/labels/IMG_0766_sunshad.tif",
    "/path/to/labels/IMG_0767_res.tif",
]

# Mocked return value for glob when passed 
mocked_glob_return_value = [
    "/path/to/images/IMG_0766.jpg",
    "/path/to/images/IMG_0767.jpg",
    # "/path/to/labels/IMG_0766_res.tif",
    # "/path/to/labels/IMG_0766_sunshad.tif",
    # "/path/to/labels/IMG_0767_res.tif",
]

### getFilePaths TESTS ###

def test_get_file_paths():
    """
    Tests getFilePaths on a normal use case - return .jpg filepaths from a 
    collection of filepaths including .jpg and .tif.
    """
    root = "/path/to/images"
    extension = "jpg"
    
    with patch('glob.glob', return_value=mocked_glob_return_value):
        result = get_file_paths(root, extension)
        assert result == ["/path/to/images/IMG_0766.jpg", "/path/to/images/IMG_0767.jpg"]

### getValidData TESTS ###

def test_get_valid_data():
    """
    Tests getValidData in normal use case - return the valid data point in
    the presence of other invalid data.
    """
    image_paths = SAMPLE_IMAGE_PATHS
    label_paths = SAMPLE_LABEL_PATHS
    
    result = get_valid_data(image_paths, label_paths)
    
    expected_result = [
        ("/path/to/images/IMG_0766.jpg", [
            "/path/to/labels/IMG_0766_res.tif",
            "/path/to/labels/IMG_0766_sunshad.tif"
        ])
    ]
    
    assert result == expected_result

def test_get_valid_data_with_no_labels():
    """
    Test get ValidData in edge case where there are no labels, so no valid
    data points.
    """
    image_paths = SAMPLE_IMAGE_PATHS
    label_paths = [
        "/path/to/labels/IMG_0767_res.tif",
        # Missing the sunshad label for IMG_0766
    ]
    
    result = get_valid_data(image_paths, label_paths)
    
    assert result == []  # No valid data should be found
