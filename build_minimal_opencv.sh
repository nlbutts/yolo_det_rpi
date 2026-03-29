#!/bin/bash
# build_minimal_opencv.sh

# Exit on error
set -e

# Configuration
OPENCV_VERSION="4.11.0"
INSTALL_DIR=$(pwd)/opencv_minimal_install
BUILD_DIR=$(pwd)/opencv_build

echo "--- Initializing Minimal OpenCV Build for Raspberry Pi ---"

# 1. Download source
if [ ! -d "opencv-$OPENCV_VERSION" ]; then
    echo "Downloading OpenCV $OPENCV_VERSION source..."
    wget -O opencv.zip https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip
    unzip -q opencv.zip
    rm opencv.zip
fi

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# 2. Configure with minimal flags
# We keep: core, imgproc, imgcodecs, dnn
# We disable: highgui (GUI), video, videoio, features2d, calib3d, objdetect, ml, photo, etc.
cmake -D CMAKE_BUILD_TYPE=RELEASE \
      -D CMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
      -D BUILD_SHARED_LIBS=OFF \
      -D BUILD_EXAMPLES=OFF \
      -D BUILD_TESTS=OFF \
      -D BUILD_PERF_TESTS=OFF \
      -D BUILD_opencv_apps=OFF \
      -D BUILD_opencv_java=OFF \
      -D BUILD_opencv_python2=OFF \
      -D BUILD_opencv_python3=ON \
      -D OPENCV_GENERATE_PKGCONFIG=ON \
      -D ENABLE_NEON=ON \
      -D ENABLE_VFPV3=OFF \
      -D BUILD_opencv_calib3d=OFF \
      -D BUILD_opencv_features2d=OFF \
      -D BUILD_opencv_flann=OFF \
      -D BUILD_opencv_highgui=OFF \
      -D BUILD_opencv_ml=OFF \
      -D BUILD_opencv_objdetect=OFF \
      -D BUILD_opencv_photo=OFF \
      -D BUILD_opencv_stitching=OFF \
      -D BUILD_opencv_video=OFF \
      -D BUILD_opencv_videoio=OFF \
      -D BUILD_opencv_gapi=OFF \
      -D WITH_GTK=OFF \
      -D WITH_WIN32UI=OFF \
      -D WITH_FFMPEG=OFF \
      -D WITH_GSTREAMER=OFF \
      -D WITH_V4L=OFF \
      -D WITH_OPENEXR=OFF \
      -D WITH_TIFF=OFF \
      -D WITH_QUIRC=OFF \
      -D WITH_ADE=OFF \
      ../opencv-$OPENCV_VERSION

# 3. Build using all available cores
echo "--- Starting build with $(nproc) cores ---"
make -j$(nproc)

# 4. Install locally
echo "--- Installing locally to $INSTALL_DIR ---"
make install

# 5. Build Python wheel
echo "--- Building Python Wheel ---"
cd "$BUILD_DIR/python_loader"
python3 setup.py bdist_wheel --dist-dir "$INSTALL_DIR/wheel"

echo "--- Build complete! Minimal OpenCV wheel at: $INSTALL_DIR/wheel ---"
