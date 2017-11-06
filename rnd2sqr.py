from PIL import Image
import numpy as np
import os


class Round2Square(object):
    """
    Convert round iris images into cropped and rotated square images.
    """
    def __init__(self, dst_image_size: int, rotation_increment: int):
        """
        Construct the Round2Square transformer and set the destination
        image size.

        :param dst_image_size:
        """
        self._dst_image_size = dst_image_size
        self._rotation_increment = rotation_increment

    def transform(self, src_dir_path: str, dst_dir_path: str, image_name: str):
        """
        Transform the source image into a square image and save to the destination location.

        :param src_dir_path: Source location of image to transform
        :param dst_dir_path: Destination location to store transformed image
        :param image_name: Image name used to generate output file names
        """

        # Load the image
        img = Image.open(os.path.join(src_dir_path, image_name))

        # Transformation pipeline
        img = self._crop(img)

        # Pad the destination image size by a pixel and make sure we can center evenly around zero.
        if self._dst_image_size % 2:
            working_image_size = self._dst_image_size + 2
            trim_mask = [True] * working_image_size
            trim_mask[0] = False
            trim_mask[-1] = False
        else:
            working_image_size = self._dst_image_size + 3
            trim_mask = [True] * working_image_size
            trim_mask[0] = False
            trim_mask[-1] = False
            trim_mask[-2] = False

        dst_pixel_org = np.floor_divide(working_image_size, 2)
        dst_radius_len = dst_pixel_org

        src_pixel_org = np.floor_divide(img.shape, 2)[:2]

        for alpha in range(0, 360, self._rotation_increment):
            # Allocate space for the destination image
            dst_img = np.zeros([working_image_size, working_image_size, 3], np.uint8)

            for dst_row in range(working_image_size):
                for dst_col in range(working_image_size):
                    dst_pixel = [dst_row - dst_pixel_org, dst_col - dst_pixel_org]

                    # Get the angle to the destination pixel
                    # Figure out if following this radial line will run into the sides or the top/bottom first.
                    # Calculate the radial distance to the corresponding edge.
                    dst_theta = np.arctan2(dst_pixel[1], dst_pixel[0])
                    dst_sin = np.sin(dst_theta)
                    dst_cos = np.cos(dst_theta)

                    if abs(dst_sin) >= abs(dst_cos):
                        # Run into top/bottom first
                        dst_edge_len = dst_pixel_org / np.sin(dst_theta)
                    else:
                        # Run into side first
                        dst_edge_len = dst_pixel_org / np.cos(dst_theta)

                    # Calculate the distance to the destination pixel
                    dst_pixel_len = np.hypot(dst_pixel[0], dst_pixel[1])

                    # Calculate the source row and column based on the stretch ratio
                    # and any non-squareness of the source image.
                    src_shape_ratio = [img.shape[0] / img.shape[1], 1]

                    # Target pixel in destination coordinates
                    ratio = abs(dst_radius_len / dst_edge_len)
                    dst_target_pixel = [dst_pixel[0] * ratio, dst_pixel[1] * ratio]

                    # Source pixel
                    src_pixel = [0, 0]
                    for i in range(2):
                        src_pixel[i] = dst_target_pixel[i] * src_shape_ratio[i] * src_pixel_org[i] / dst_pixel_org

                    # print('{}, {}, {:+}, {:+}, {:+5.2f}, {:+5.2f}, {:+5.2f}, {:+5.2f}, {}, {}'. \
                    #       format(dst_row, dst_col, dst_pixel[0], dst_pixel[1], np.degrees(dst_theta),
                    #              dst_edge_len, dst_pixel_len, dst_pixel_len / dst_edge_len,
                    #              int(src_pixel[0]), int(src_pixel[1])))

                    # Adjust for the rotation in source image coordinates in case the source image
                    # isn't square.
                    src_theta = np.arctan2(src_pixel[1], src_pixel[0])
                    pixel_len = np.hypot(src_pixel[0], src_pixel[1])
                    rotation = [np.cos(src_theta - np.radians(alpha)), np.sin(src_theta - np.radians(alpha))]

                    # print('Dst Pxl: {}, {}'.format(int(dst_pixel[0]), int(dst_pixel[1])))
                    # print('\tRot: {:5.2f}, {:5.2f}'.format(np.degrees(dst_theta), np.degrees(src_theta)))
                    # print('\tSrc Org Pxl: {:5.2f}, {:5.2f}'.format(src_pixel_org[0], src_pixel_org[1]))
                    for i in range(2):
                        src_pixel[i] = pixel_len * rotation[i]

                    # print('\tSrc Pxl 1:{}, {}'.format(int(src_pixel[0]), int(src_pixel[1])))

                    for i in range(2):
                        # Uncenter the image coordinates
                        src_pixel[i] = int(src_pixel_org[i] + src_pixel[i])

                    # print('\tSrc Pxl 2:{}, {}'.format(int(src_pixel[0]), int(src_pixel[1])))

                    # Check to make sure the source file indices are in range.  Not sure if this is a bug
                    # or just rounding error that is causing them to be out of bounds.
                    # It should be checked to see if these are just in the padding and will be deleted
                    # anyhow or if it is a bigger issue.
                    if any([src_pixel[0] >= img.shape[0],
                            src_pixel[0] < 0,
                            src_pixel[1] >= img.shape[1],
                            src_pixel[1] < 0]):
                        print('Index out of bounds {}, {}'.format(src_pixel[0], src_pixel[1]))
                    else:
                        dst_img[dst_row, dst_col] = img[int(src_pixel[0]), int(src_pixel[1])]

            # Trim the image to remove the padding.
            dst_img = dst_img[trim_mask][:, trim_mask]

            # Save the transformed image
            base_fn = os.path.splitext(image_name)
            dst_file_path = os.path.join(dst_dir_path, base_fn[0] + '_' + '{:03d}'.format(alpha) + base_fn[1])
            print(dst_file_path)
            pil_img = Image.fromarray(dst_img)
            pil_img.save(dst_file_path)

    @staticmethod
    def _crop(pil_img: Image) -> np.array:
        # Get a numpy grayscale image
        """
        Crop any rows and columns around the edge of the image
        that are all black.

        :param pil_img: Image to crop
        """
        np_gs_img = np.array(pil_img.convert('L'))

        # Find any rows or column where there is any image data.
        row_mask = [False] * np_gs_img.shape[0]
        for row in range(np_gs_img.shape[0]):
            row_mask[row] = any(np_gs_img[row])

        col_mask = [False] * np_gs_img.shape[1]
        for col in range(np_gs_img.shape[1]):
            col_mask[col] = any(np_gs_img[:, col])

        # Eliminate any of the rows or columns where there are only 0's
        np_rgb_img = np.array(pil_img)[row_mask][:, col_mask]

        return np_rgb_img


if __name__ == '__main__':
    src_dir_path = './images/src'
    dst_dir_path = './images/dst'
    image_name = '10_left.jpeg'

    transformer = Round2Square(128, 45)
    transformer.transform(src_dir_path, dst_dir_path, image_name)
