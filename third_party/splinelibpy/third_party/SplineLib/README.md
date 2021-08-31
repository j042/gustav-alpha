# SplineLib
Library for spline manipulation.

## Features
Functionalities:

  - BSplines/NURBSs of arbitrary dimensionalities;
  - knot insertion/removal and degree elevation/reduction;
  - IGES, IRIT, and XML input/output (converter) as well as VTK output (sampler).

Software Development:

  - [Spack](https://spack.readthedocs.io/en/latest/) support for handling dependencies and deployment of SplineLib.
  - C++20 complying with [Google C++ style guide](https://google.github.io/styleguide/cppguide.html) (a C++17 version
    is available using `git checkout c++17`).
  - CMake tools tested with apple-clang, clang, and gcc compilers (if your CMake version is less than 3.19, consider
    `git checkout cmake`).
  - Extensive unit tests using GTest and GMock.

## License (MIT)
Copyright (c) 2018–2021 SplineLib

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## Getting Started
SplineLib can be compiled using, e.g., *apple-clang12*, *clang11* or *GCC10*.  Start off by cloning the SplineLib
repository and changing to SplineLib's root directory

	 $ git clone https://github.com/SplineLib/SplineLib.git && cd SplineLib

SplineLib depends on *pugixml* and *GoogleTest* (if you do not disable testing).  The following deployment instructions
for SplineLib differ depending on whether you want to use 1.) *spack* or 2.) *CMake*.  Please skip the additional
instructions provided for using SplineLib on CLAIX if you use a system different from CLAIX.

<details><summary><strong>Preparations on CLAIX</strong></summary><p>
The following commands prepare your environment on CLAIX for deploying SplineLib

	$ module purge && module load DEVELOP clang/11 gcc/10 cmake
</p></details>

<details><summary><strong>Building/Installing using <em>Spack</em></strong></summary><p>
<em>Spack</em> (see <a href=https://spack.readthedocs.io/en/latest/getting_started.html#installation><em>spack</em>'s
installation instructions</a>) is convenient to manage SplineLib's dependencies

	$ spack install googletest+gmock~pthreads pugixml

In addition, <em>spack</em> can also be used for SplineLib itself.  Begin by registering SplineLib's *spack* package

	 $ spack repo add Scripts/Spack/splinelib

and continue by either installing SplineLib from GitHub and making it available in your current environment

	$ spack install splinelib && spack load splinelib

or by developing your modifiable copy of the SplineLib repository

	$ spack dev-build -b install --test root splinelib@github build_type=Debug

At some point in time, you can remove the `-b install` option from the previous command if you want *spack* to not stop
before the install phase.
</p></details>

<details><summary><strong>Building/Installing using <em>CMake</em></strong></summary><p>
If you do not want to use <em>spack</em>, you can also take care of SplineLib's dependencies and build system on your
own.  Start with creating a build directory

	 $ mkdir Build && cd Build

and then (after making sure that SplineLib's dependencies are available) build and test your version of SplineLib using
<em>CMake</em>'s Unix Makefiles generator

	$ ctest --build-and-test .. . --build-generator "Unix Makefiles" --build-options -DCMAKE_BUILD_TYPE=Debug
	  --test-command ctest

being a handy shortcut for invoking the equivalent command sequence

	$ cmake .. -DCMAKE_BUILD_TYPE=Debug && make clean && make -j && ctest

or create an Xcode project by using *CMake*'s Xcode generator

	$ cmake .. -G Xcode

If you eventually want to install SplineLib, you can type

	$ make install

or

	$ xcodebuild install
</p></details>

## Contributors in Alphabetical Order
Researchers:

  - Markus Frings
  - Konstantin Key

Research Assistants:

  - Alexander Jodlauk
  - Corinna M&uuml;ller
  - Max Spahn
  - Christoph Susen
