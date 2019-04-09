"""
Visualize LCM images from robotlocomotion `image_array_t` and `image_t`.

Example usage:

    drake-visualizer \
        --script drake/systems/sensors/visualization/show_images.py

This provides a simple image viewer widget, using portions of code from
director (https://github.com/RobotLocomotion/director, sha: aefc063),
specifically:
*   src/python/director/cameraview.py (CameraImageView)
*   src/app/ddBotImageQueue.cpp
"""

#######################chris's imports#########################
import os
from scipy import misc
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Activation
from keras.layers import Reshape
from keras.layers.convolutional import Conv2D
from keras.layers import *
from keras import Model
from keras import optimizers
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras import backend as K
# import cv2
# from sklearn.model_selection import train_test_split
import noise
# from scipy.misc import toimage

#######################################################

import sys
import argparse
import math
import time
import zlib
import threading

import numpy as np
import numpy.matlib

import vtk
from vtk.util.numpy_support import vtk_to_numpy, get_vtk_array_type,numpy_to_vtk

from director import applogic
from director import consoleapp
from director import lcmUtils
from director.timercallback import TimerCallback

import PythonQt
from PythonQt import QtGui

import robotlocomotion as rl

from drake.tools.workspace.drake_visualizer.plugin import scoped_singleton_func

_is_vtk_5 = vtk.vtkVersion().GetVTKMajorVersion() == 5

_verbose = False
_max_depth = -1  # m

DEFAULT_CHANNEL = "DRAKE_RGBD_CAMERA_IMAGES"


class ImageHandler(object):
    """
    Generic handler to update an image for `ImageWidget`.
    """
    def update_image(self, image):
        """
        This should update `image` in-place, either using `DeepCopy`, or by
        manually resizing, allocating, etc.
        If the image is updated, this should return True.
        Otherwise, it should return False.
        """
        raise NotImplementedError()

    def is_depth_image(self):
        """
        Returns true if the image is a depth image.
        """
        return False


class ImageWidget(object):
    """
    Wrapper for displaying vtkImageData on a director-style view.

    @note This is more like director's `CameraImageView` than its
    `ImageWidget`.
    """
    def __init__(self, image_handler):
        self.model_path = "/home/drc/Chris/DepthSim/python/models/net_depth_seg_v1.hdf5"
        self.model = self.load_trained_model(weights_path = self.model_path)
        self._name = 'Image View'
        self._view = PythonQt.dd.ddQVTKWidgetView()
        self._image_handler = image_handler

        self._image = vtk.vtkImageData()
        self._prev_attrib = None

        # Initialize the view.
        self._view.installImageInteractor()
        # Add actor.
        self._image_actor = vtk.vtkImageActor()
        vtk_SetInputData(self._image_actor, self._image)
        self._image_actor.SetVisibility(False)
        self._view.renderer().AddActor(self._image_actor)

        self._view.orientationMarkerWidget().Off()
        self._view.backgroundRenderer().SetBackground(0, 0, 0)
        self._view.backgroundRenderer().SetBackground2(0, 0, 0)

        self._depth_mapper = None

        # Add timer.
        self._render_timer = TimerCallback(
            targetFps=60,
            callback=self.render)
        self._render_timer.start()

    def get_widget(self):
        return self._view
    
    # This is the better model
    def create_model_2(self,img_height=480, img_width=640,channels=1):
       inputs = Input((img_height, img_width,channels))
       #crop = Cropping2D(cropping=((0, 0), (0, 0)), data_format=None)
       conv1 = Conv2D(4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',data_format='channels_last')(inputs)
       conv1 = Conv2D(4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',data_format='channels_last')(conv1)
       pool1 = MaxPooling2D(pool_size=(2, 2),data_format='channels_last')(conv1)

       conv2 = Conv2D(8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',data_format='channels_last')(pool1)
       conv2 = Conv2D(8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',data_format='channels_last')(conv2)
       pool2 = MaxPooling2D(pool_size=(2, 2),data_format='channels_last')(conv2)

       conv3 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',data_format='channels_last')(pool2)
       conv3 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',data_format='channels_last')(conv3)
       pool3 = MaxPooling2D(pool_size=(2, 2),data_format='channels_last')(conv3)


       conv5 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',data_format='channels_last')(pool3)
       conv5 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',data_format='channels_last')(conv5)
       drop5 = Dropout(0.5)(conv5)

       up7 = Conv2D(32, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',data_format='channels_last')(UpSampling2D(size = (2,2),data_format='channels_last')(conv5))
       merge7 = merge([conv3,up7], mode = 'concat', concat_axis = 3)
       conv7 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',data_format='channels_last')(merge7)
       conv7 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',data_format='channels_last')(conv7)

       up8 = Conv2D(8, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',data_format='channels_last')(UpSampling2D(size = (2,2),data_format='channels_last')(conv7))
       merge8 = merge([conv2,up8], mode = 'concat', concat_axis = 3)
       conv8 = Conv2D(8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',data_format='channels_last')(merge8)
       conv8 = Conv2D(8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',data_format='channels_last')(conv8)

       up9 = Conv2D(4, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',data_format='channels_last')(UpSampling2D(size = (2,2),data_format='channels_last')(conv8))
       merge9 = merge([conv1,up9], mode = 'concat', concat_axis = 3)
       conv9 = Conv2D(4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',data_format='channels_last')(merge9)
       conv9 = Conv2D(4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',data_format='channels_last')(conv9)
       conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',data_format='channels_last')(conv9)
       conv10 = Conv2D(1, 1, activation = 'sigmoid',data_format='channels_last')(conv9)

       model = Model(input = inputs, output = conv10)

       return model
    
    def corrupt(self,image):
        np_image = vtk_image_to_numpy(image).astype(np.float32)/3500.
        threshold = .5
        img_height,img_width = (480,640)
        stack = np.zeros((1,img_height,img_width,1)) 
        stack[0,:,:,:] = np_image
        predicted_prob_map = self.model.predict_on_batch(stack)
        im_final = self.apply_mask_simple(predicted_prob_map,np_image[:,:,0],.5)
        # im_final[im_final>1] = 1
        # plt.imshow(im_final)
        # plt.show()
        # self.apply_mask(predicted_prob_map,np_image[:,:,0],threshold)
        #depthsim_source+= (max_range * min_range) / (max_range + np.random.randn(camera_height, camera_width)*rgbd_noise * (min_range - max_range))
        corrupt_vtk_img = vtk.vtkImageData()
        corrupt_vtk_img.SetDimensions(im_final.shape[1], im_final.shape[0], 1)
        im_final = im_final.reshape(640,480).swapaxes(0,1)*3500
        im_final = im_final.astype(np.uint16)
        vtkarr = numpy_to_vtk(np.flip(im_final.reshape(640,480).swapaxes(0,1)*3500, axis=1).reshape((-1, 1), order='F'))
        vtkarr.SetName('Image')
        corrupt_vtk_img.GetPointData().AddArray(vtkarr)
        corrupt_vtk_img.GetPointData().SetActiveScalars('Image')
        # corrupt_vtk_img = create_image(640, 480, 1, dtype=np.float32)
        # corrupt_vtk_img.SetDimensions((640,480,1))

        # corrupt_vtk_img.GetPointData().SetScalars(corrupt_vtk_array)
    #     w, h = image.SetDimensions()[:2]
    # num_channels = image.GetNumberOfScalarComponents()
    # return (h, w, num_channels)
        return corrupt_vtk_img

    def load_trained_model(self,weights_path):
       model = self.create_model_2(channels=1)# only train on depth images
       model.load_weights(weights_path)
       return model
    def apply_mask(self, mask,depth,threshold):
       epsilon = .05
       h,w = np.shape(depth)
       mask = np.reshape(mask,(h,w))
       depth[mask>threshold]=0
       #img = np.random.random((480,640))
       img = self.sigmoid(self.perlin_map(scale=20.0,octaves = 7,base= np.random.randint(1000),lacunarity = 6.0))
       stochastic_mask = mask>=img
       #stochastic_mask = np.logical_and(np.logical_and((mask<=threshold), mask>epsilon), img<threshold)
       depth[stochastic_mask] = 0

    def apply_mask_simple(self, mask,depth,threshold):
       h,w = np.shape(depth)
       mask = np.reshape(mask,(h,w))
       depth[mask>threshold]=0
       return depth
    def perlin_map(self,shape = (480,640),scale = 10.0,octaves = 6,persistence = 0.5,lacunarity = 2.0,base = 0):
        img = np.zeros(shape)
        #this is a bottle neck for real time performance: this takes .3 seconds per image
        for i in range(shape[0]):
            for j in range(shape[1]):
                img[i][j] = noise.pnoise2(i/scale, 
                                           j/scale, 
                                           octaves=octaves, 
                                           persistence=persistence, 
                                           lacunarity=lacunarity, 
                                           repeatx=shape[1], 
                                           repeaty=shape[0], 
                                           base=base)
        return img

    def sigmoid(self,x):
        return 1. / (1 + np.exp(-10*x))

    def render(self):
        if not self._view.isVisible():
            return

        has_new = self._image_handler.update_image(self._image)
        assert isinstance(has_new, bool)
        if not has_new:
            return
        self._image = self.corrupt(self._image)
        cur_attrib = get_vtk_image_attrib(self._image)
        if self._prev_attrib != cur_attrib:
            if self._prev_attrib is None:
                # Initialization. Ensure it is visible.
                self._image_actor.SetVisibility(True)
            # Fit image to view.
            self._on_new_image_attrib(cur_attrib)
            # Update.
            self._prev_attrib = cur_attrib

        if self._depth_mapper is not None:
            depth_range = self._get_depth_range()
            for i in xrange(2):
                value = [0.] * 6
                coloring = self._depth_mapper.GetLookupTable()
                coloring.GetNodeValue(i, value)
                value[0] = depth_range[i]
                coloring.SetNodeValue(i, value)

        self._view.render()

    def _get_depth_range(self):
        lower_depth = 0
        upper_depth = _max_depth
        if upper_depth == -1:
            # @note `GetScalarRange` permits non-finite values, such as `inf`.
            # Use a custom mechanism to get min/max.
            data = vtk_image_to_numpy(self._image)
            if data.dtype == np.float32:
                good = np.isfinite(data[:])
            elif data.dtype == np.uint16:
                maxarray = np.full(data.shape, 65535)
                good = np.less(data[:], maxarray)
            else:
                raise RuntimeError(
                    "Unsupported depth format: {}".format(data.dtype))
            if np.any(good):
                upper_depth = np.max(data[good])
        return (lower_depth, upper_depth)

    def _on_new_image_attrib(self, attrib):
        ((w, h, num_channels), dtype) = attrib
        if self._image_handler.is_depth_image():
            assert num_channels == 1, num_channels
            assert dtype in (np.uint16, np.float32), dtype
            # TODO(eric.cousineau): Delegate to outside of `ImageWidget`?
            # This is depth-image specific.
            # For now, just set arbitrary values.

            depth_range = self._get_depth_range()
            lower_color = (1, 1, 1)  # White
            upper_color = (0, 0, 0)  # Black
            nan_color = (0.5, 0.5, 1)  # Light blue - No return.
            inf_color = (0.5, 0, 0.)  # Dark red - Too far / too close.

            # Use `vtkColorTransferFunction` as it provides a more intuitive
            # interpolating interface for me (Eric) than `vtkLookupTable`,
            # since it permits direct specification of RGB values.
            coloring = vtk.vtkColorTransferFunction()
            coloring.AddRGBPoint(depth_range[0], *lower_color)
            coloring.AddRGBPoint(depth_range[1], *upper_color)
            coloring.SetNanColor(*nan_color)
            # @note `coloring.SetAboveRangeColor` doesn't seem to work?
            coloring.AddRGBPoint(depth_range[1] + 10000, *inf_color)
            coloring.SetClamping(True)
            coloring.SetScaleToLinear()

            self._depth_mapper = vtk.vtkImageMapToColors()
            self._depth_mapper.SetLookupTable(coloring)
            vtk_SetInputData(self._depth_mapper, self._image)
            vtk_SetInputData(self._image_actor, self._depth_mapper.GetOutput())
            self._image_actor.GetMapper().SetInputConnection(
                self._depth_mapper.GetOutputPort())
        else:
            # Direct connection.
            self._depth_mapper = None
            vtk_SetInputData(self._image_actor, self._image)

        # Must render first.
        self._view.render()

        # Fit image to view.
        # TODO(eric.cousineau): No idea why this is needed; it worked for
        # VTK 5, but no longer for VTK 6+?
        camera = self._view.camera()
        camera.ParallelProjectionOn()
        camera.SetFocalPoint(0, 0, 0)
        camera.SetPosition(0, 0, -1)
        camera.SetViewUp(0, -1, 0)
        self._view.resetCamera()

        image_height, image_width = get_vtk_image_shape(self._image)[:2]
        view_width, view_height = self._view.renderWindow().GetSize()

        aspect_ratio = float(view_width) / view_height
        parallel_scale = max(image_width / aspect_ratio, image_height) / 2.0
        camera.SetParallelScale(parallel_scale)


class ImageArrayWidget(object):
    """
    Provides a widget to show images from multiple `ImageHandler`s.
    """
    def __init__(self, handlers):
        # Create widget and layouts
        self._widget = QtGui.QWidget()
        self._image_widgets = map(ImageWidget, handlers)
        self._layout = QtGui.QHBoxLayout(self._widget)
        for image_widget in self._image_widgets:
            self._layout.addWidget(image_widget.get_widget())
        self._layout.setContentsMargins(0, 0, 0, 0)

        default_width = 640
        default_height = 480
        dim = [
            default_width * len(self._image_widgets),
            default_height]

        self._widget.resize(*dim)
        self._widget.show()


class DrakeLcmImageViewer(object):
    """
    Visualize Drake LCM Images.
    """
    def __init__(self, channel=DEFAULT_CHANNEL, frame_names=None):
        """
        @param frame_names
            If None, this will defer creating the subscriber and
            widgets until the first message has been received.
        """
        self._channel = channel
        if frame_names is None:
            self._create_deferred()
        else:
            self._init_full(frame_names)

    def _init_full(self, frame_names):
        self._frame_names = frame_names
        self._subscriber = LcmImageArraySubscriber(
            self._channel, self._frame_names)
        self._widget = ImageArrayWidget(self._subscriber.get_handlers())

    def _create_deferred(self):
        # Defer creating viewer until we have a message.
        def callback(msg):
            # Create.
            frame_names = [image.header.frame_name for image in msg.images]
            print("DrakeLcmImageViewer: Received on '{}', frame_names = {}"
                  .format(self._channel, frame_names))
            self._init_full(frame_names)

        print("DrakeLcmImageViewer: Defer setup until '{}' is received".format(
            self._channel))
        self._defer_sub = lcmUtils.captureMessageCallback(
            self._channel, rl.image_array_t, callback)


def create_image(w, h, num_channels=1, dtype=np.uint8):
    """ Creates a VTK image. """
    image = vtk.vtkImageData()
    image.SetExtent(0, w - 1, 0, h - 1, 0, 0)
    image.SetSpacing(1., 1., 1.)
    image.SetOrigin(0., 0., 0.)
    if _is_vtk_5:
        image.SetWholeExtent(image.GetExtent())
        image.SetScalarType(get_vtk_array_type(dtype))
        image.SetNumberOfScalarComponents(num_channels)
        image.AllocateScalars()
    else:
        image.AllocateScalars(get_vtk_array_type(dtype), num_channels)
    return image


def create_image_if_needed(w, h, num_channels, dtype, image_in):
    """
    Creates a VTK image if `image_in` is not compatible with the desired
    attributes. Otherwise, passes `image_in` through.
    """
    if image_in is not None:
        dim = (w, h, num_channels)
        attrib_out = (dim, dtype)
        attrib_in = get_vtk_image_attrib(image_in)
        if attrib_in == attrib_out:
            return image_in
    # Otherwise, create new image.
    return create_image(w, h, num_channels, dtype)


def vtk_image_to_numpy(image):
    """
    Gets a properly shaped NumPy view of a VTK image's memory with the storage
    format `(h, w, num_channels)`.

    @note This coincides with most other NumPy-based image libraries (OpenCV,
    matplotlib, scipy).
    """
    data = vtk_to_numpy(image.GetPointData().GetScalars())
    data.shape = get_vtk_image_shape(image)
    return data

def get_vtk_image_shape(image):
    """
    Gets `(h, w, num_channels)`.

    @note `vtkImageData.GetDimensions()` returns `(w, h, num_arrays)`, where
    typically `num_arrays == 1 != num_channels`.
    """
    w, h = image.GetDimensions()[:2]
    num_channels = image.GetNumberOfScalarComponents()
    return (h, w, num_channels)


def get_vtk_image_attrib(image):
    """
    Gets `((h, w, num_channels), dtype)` to check if an existing image is
    compatible.
    """
    data = vtk_image_to_numpy(image)
    return (data.shape, data.dtype)


def vtk_SetInputData(obj, input):
    if _is_vtk_5:
        obj.SetInput(input)
    else:
        obj.SetInputData(input)


def decode_image_t(msg, image_in=None):
    """
    Decodes `image_t` to vtkImageData, using an existing image if it is
    compatible.
    """
    rli = rl.image_t
    w = msg.width
    h = msg.height
    pixel_desc = (msg.pixel_format, msg.channel_type)
    if pixel_desc == (rli.PIXEL_FORMAT_RGBA, rli.CHANNEL_TYPE_UINT8):
        num_channels = 4
        dtype = np.uint8
    elif pixel_desc == (rli.PIXEL_FORMAT_DEPTH, rli.CHANNEL_TYPE_FLOAT32):
        num_channels = 1
        dtype = np.float32
    elif pixel_desc == (rli.PIXEL_FORMAT_DEPTH, rli.CHANNEL_TYPE_UINT16):
        num_channels = 1
        dtype = np.uint16
    elif pixel_desc == (rli.PIXEL_FORMAT_LABEL, rli.CHANNEL_TYPE_INT16):
        num_channels = 1
        dtype = np.int16
    else:
        raise RuntimeError("Unsupported pixel type: {}".format(pixel_desc))
    bytes_per_pixel = np.dtype(dtype).itemsize * num_channels
    assert msg.row_stride == msg.width * bytes_per_pixel, msg.row_stride
    if msg.compression_method == rli.COMPRESSION_METHOD_NOT_COMPRESSED:
        data_bytes = msg.data
    elif msg.compression_method == rli.COMPRESSION_METHOD_ZLIB:
        # TODO(eric.cousineau): Consider using `data`s buffer, if possible.
        # Can decompress() somehow use an existing buffer in Python?
        data_bytes = zlib.decompress(msg.data)
    else:
        raise RuntimeError(
            "Unsupported compression type: {}".format(msg.compression_method))
    # Cast to desired type and shape.
    data = np.frombuffer(data_bytes, dtype=dtype)
    data.shape = (h, w, num_channels)
    # Copy data to VTK image.
    image = create_image_if_needed(w, h, num_channels, dtype, image_in)
    image_data = vtk_image_to_numpy(image)
    image_data[:] = data[:]
    return image


class LcmImageHandler(ImageHandler):
    """
    Provides a connection between `LcmImageArraySubscriber`, for a specific
    image frame, and `ImageWidget`.
    """
    def __init__(self):
        self._image = None  # vtkImageData
        self.utime = 0

        self.lock = threading.Lock()
        self.prev_utime = 0
        self._is_depth_image = False

    def receive_message(self, msg):
        """
        Receives and decodes `image_t` message into `vtkImageData`.
        """
        # TODO(eric.cousineau): Consider moving decode logic.
        with self.lock:
            self.utime = msg.header.utime
            self._image = decode_image_t(msg, self._image)
            self._is_depth_image = (msg.pixel_format ==
                                    rl.image_t.PIXEL_FORMAT_DEPTH)

    def update_image(self, image_out):
        """
        @see ImageHandler.update_image
        """
        with self.lock:
            if self.utime == self.prev_utime:
                return False
            elif self.utime < self.prev_utime:
                if _verbose:
                    print("Time went backwards. Resetting.")
            self.prev_utime = self.utime
            assert self._image is not None
            image_out.DeepCopy(self._image)
        return True

    def is_depth_image(self):
        return self._is_depth_image


class LcmImageArraySubscriber(object):
    """
    Provides a connection between the LCM `image_array_t` channel and LCM image
    handlers.
    """
    def __init__(self, channel=DEFAULT_CHANNEL, frame_names=[]):
        self._channel = channel
        self._frame_names = frame_names
        self._handlers = {}
        for frame_name in self._frame_names:
            self._handlers[frame_name] = LcmImageHandler()
        self._subscriber = lcmUtils.addSubscriber(
            channel, rl.image_array_t, self._on_message)

    def get_handlers(self):
        """ Returns ordered list of handlers. """
        # TODO(eric.cousineau): Consider just using `OrderedDict`.
        return map(self._handlers.get, self._frame_names)

    def _on_message(self, msg):
        issues = []
        msg_frame_names = [image.header.frame_name for image in msg.images]
        for (frame_name, handler) in self._handlers.iteritems():
            if frame_name not in msg_frame_names:
                issues.append("Did not find '{}' in message"
                              .format(frame_name))
                continue
            index = msg_frame_names.index(frame_name)
            msg_image = msg.images[index]
            handler.receive_message(msg_image)

        if _verbose:
            extra_frame_names = set(msg_frame_names) - set(self._frame_names)
            for frame_name in extra_frame_names:
                issues.append("Got extra image '{}'".format(frame_name))

        if issues:
            print("LcmImageArraySubscriber: For image channel '{}' " +
                  "({} images), with frames {}:"
                  .format(self._channel, msg.num_images, msg_frame_names))
            for issue in issues:
                print("  {}".format(issue))


class TestImageHandler(ImageHandler):
    """
    Provides a simple test image handler.
    This just shows an animated gradient, either in color or in grayscale.
    """
    def __init__(self, do_color):
        self.do_color = do_color
        self.start_time = time.time()
        if self.do_color:
            self._image = create_image(640, 480, 4, dtype=np.uint8)
        else:
            self._image = create_image(640, 480, 1, dtype=np.float32)
        self.has_switched = False

    def update_image(self, image_out):
        t = time.time() - self.start_time

        if t > 2 and not self.has_switched:
            # Try changing the size.
            if self.do_color:
                self._image = create_image(480, 640, 4, dtype=np.uint8)
            else:
                self._image = create_image(480, 640, 1, dtype=np.float32)
            self.has_switched = True

        if self.do_color:
            p = 255.
        else:
            p = 10.  # m

        data = vtk_image_to_numpy(self._image)
        h, w = data.shape[:2]

        x = np.matlib.repmat(
            np.linspace(0, 1, w).reshape(1, -1), h, 1)
        y = np.matlib.repmat(
            np.linspace(0, 1, h).reshape(-1, 1), 1, w)
        T = 1
        w = 2 * math.pi / T
        s = (math.sin(w * t) + 1) / 2.
        s2 = (math.sin(w * t * 5) + 1) / 2.

        data[:, :, 0] = p * y * s
        if self.do_color:
            data[:, :, 1] = p * (1 - y)
            data[:, :, 2] = p * x * s2
            data[:, :, 3] = p * s
        else:
            # Test `inf` and `nan`.
            gw = 50
            data[0:gw, 0:gw, 0] = np.inf
            data[0:gw, gw:(2 * gw), 0] = np.nan

        image_out.DeepCopy(self._image)
        return True


@scoped_singleton_func
def init_visualizer(debug=False):
    if not debug:
        return DrakeLcmImageViewer(DEFAULT_CHANNEL)
    else:
        print("Using test image viewer")
        return ImageArrayWidget([
            TestImageHandler(do_color=True),
            TestImageHandler(do_color=False),
            ])


# Activate the plugin if this script is run directly; store the results to keep
# the plugin objects in scope.
if __name__ == "__main__":
    image_viz = init_visualizer()
