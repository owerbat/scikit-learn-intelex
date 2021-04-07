#! /usr/bin/env python
#===============================================================================
# Copyright 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#===============================================================================

import os
import sys
import pathlib
import subprocess
from distutils import log

IS_WIN = False
IS_MAC = False
IS_LIN = False

if 'linux' in sys.platform:
    IS_LIN = True
elif sys.platform == 'darwin':
    IS_MAC = True
elif sys.platform in ['win32', 'cygwin']:
    IS_WIN = True

try:
    import dpctl
    _dpctrl_include_dir = str(os.path.abspath(dpctl.get_include()))
    _dpctrl_library_dir = str(os.path.abspath(os.path.join(dpctl.get_include(), "..")))
    _dpctrl_exists = "ON"
except ImportError:
    _dpctrl_include_dir = ""
    _dpctrl_library_dir = ""
    _dpctrl_exists = "OFF"


def custom_build_cmake_clib():
    root_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
    log.info(f"Project directory is: {root_dir}")

    builder_directory = os.path.join(root_dir, "scripts")
    abs_build_temp_path = os.path.join(root_dir, "build", "backend")
    install_directory = os.path.join(root_dir, "onedal")

    print('builder_directory: ', builder_directory)
    print('abs_build_temp_path: ', abs_build_temp_path)
    print('install_directory: ', install_directory)
    if IS_WIN:
        cmake_generator = "-GNinja"
    else:
        cmake_generator = ""

    cmake_args = [
        "cmake",
        cmake_generator,
        "-S" + builder_directory,
        "-B" + abs_build_temp_path,
        "-DCMAKE_INSTALL_PREFIX=" + install_directory,
        "-DCMAKE_PREFIX_PATH=" + install_directory,
        "-DDPCTL_ENABLE:BOOL=" + _dpctrl_exists,
        "-DDPCTL_INCLUDE_DIR=" + _dpctrl_include_dir,
        "-DDPCTL_LIB_DIR=" + _dpctrl_library_dir,
    ]

    import multiprocessing
    cpu_count = multiprocessing.cpu_count()

    make_args = [
        "cmake",
        "--build",
        abs_build_temp_path,
        "-j " + str(cpu_count)
    ]

    make_install = [
        "cmake",
        "--install",
        abs_build_temp_path,
    ]

    subprocess.check_call(cmake_args, stderr=subprocess.STDOUT, shell=False)
    subprocess.check_call(make_args, stderr=subprocess.STDOUT, shell=False)
    subprocess.check_call(make_install, stderr=subprocess.STDOUT, shell=False)