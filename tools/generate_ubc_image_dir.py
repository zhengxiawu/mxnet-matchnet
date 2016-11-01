import skimage,sys
import scipy.misc
sys.path.append('/home/sherwood/project/caffe/python/')
import cv2
from caffe.io import *
def GetPatchImage(patch_id, container_dir):
    """Returns a 64 x 64 patch with the given patch_id. Catch container images to
       reduce loading from disk.
    """
    # Define constants. Each container image is of size 1024x1024. It packs at
    # most 16 rows and 16 columns of 64x64 patches, arranged from left to right,
    # top to bottom.
    PATCHES_PER_IMAGE = 16 * 16
    PATCHES_PER_ROW = 16
    PATCH_SIZE = 64

    # Calculate the container index, the row and column index for the given
    # patch.
    container_idx, container_offset = divmod(patch_id, PATCHES_PER_IMAGE)
    row_idx, col_idx = divmod(container_offset, PATCHES_PER_ROW)

    # Read the container image if it is not cached.
    if GetPatchImage.cached_container_idx != container_idx:
        GetPatchImage.cached_container_idx = container_idx
        GetPatchImage.cached_container_img = \
            skimage.img_as_ubyte(skimage.io.imread('%s/patches%04d.bmp' % \
                (container_dir, container_idx), as_grey=True))

    # Extract the patch from the image and return.
    patch_image = GetPatchImage.cached_container_img[ \
        PATCH_SIZE * row_idx:PATCH_SIZE * (row_idx + 1), \
        PATCH_SIZE * col_idx:PATCH_SIZE * (col_idx + 1)]
    return patch_image
GetPatchImage.cached_container_idx = None
GetPatchImage.cached_container_img = None
if __name__ == '__main__':
    interest_path = '/home/sherwood/project/matchnet/data/phototour/liberty/interest.txt'
    info_path = '/home/sherwood/project/matchnet/data/phototour/liberty/info.txt'
    image_dir = '/home/sherwood/project/matchnet/data/phototour/liberty/'

    #open txt file
    with open(info_path) as f:
        point_id = [int(line.split()[0]) for line in f]
    with open(interest_path) as f:
        interest = [[float(x) for x in line.split()] for line in f]
    for i,metadata in enumerate(interest):
        img = GetPatchImage(point_id[i],image_dir)
        cv2.imshow(img)
