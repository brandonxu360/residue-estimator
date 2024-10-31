from unittest.mock import patch
from src.data_utils import get_file_paths, get_valid_data

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

# Mocked return value for glob
mocked_glob_return_value = [
    "/path/to/images/IMG_0766.jpg",
    "/path/to/images/IMG_0767.jpg",
    "/path/to/labels/IMG_0766_res.tif",
    "/path/to/labels/IMG_0766_sunshad.tif",
    "/path/to/labels/IMG_0767_res.tif",
]

def test_get_file_paths():
    root = "/path/to/images"
    extension = "jpg"
    
    with patch('glob.glob', return_value=mocked_glob_return_value):
        result = get_file_paths(root, extension)
        assert result == ["/path/to/images/IMG_0766.jpg", "/path/to/images/IMG_0767.jpg"]

def test_get_valid_data():
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
    image_paths = SAMPLE_IMAGE_PATHS
    label_paths = [
        "/path/to/labels/IMG_0767_res.tif",
        # Missing the sunshad label for IMG_0766
    ]
    
    result = get_valid_data(image_paths, label_paths)
    
    assert result == []  # No valid data should be found
