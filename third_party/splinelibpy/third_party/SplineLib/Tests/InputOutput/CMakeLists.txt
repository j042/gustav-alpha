# Copyright (c) 2018–2021 SplineLib
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE AUTHORS
# OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

set(
    TEST_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/iges_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/irit_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/input_output_operations_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/input_output_splines_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/vtk_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/xml_test.cpp
)

set(iges_file_one ${CMAKE_CURRENT_SOURCE_DIR}/test_one.iges)
set(iges_file_two ${CMAKE_CURRENT_SOURCE_DIR}/test_two.iges)
set(irit_file ${CMAKE_CURRENT_SOURCE_DIR}/test.itd)
set(xml_file ${CMAKE_CURRENT_SOURCE_DIR}/test.xml)
configure_file(config_iges.in.hpp config_iges.hpp)
configure_file(config_irit.in.hpp config_irit.hpp)
configure_file(config_xml.in.hpp config_xml.hpp)

set(itd_xml ${CMAKE_CURRENT_BINARY_DIR}/converted.xml)
set(log_converter ${CMAKE_CURRENT_BINARY_DIR}/converter.log)
set(log_invalid_numbers_of_parametric_coordinates_for_splines
    ${CMAKE_CURRENT_BINARY_DIR}/invalid_numbers_of_parametric_coordinates_for_splines.log)
set(log_invalid_splines ${CMAKE_CURRENT_BINARY_DIR}/invalid_splines.log)
set(log_sampler ${CMAKE_CURRENT_BINARY_DIR}/sampler.log)
set(xml_vtk ${CMAKE_CURRENT_BINARY_DIR}/sampled.vtk)
configure_file(converter.in.log ${log_converter})
configure_file(invalid_numbers_of_parametric_coordinates_for_splines.log
               ${log_invalid_numbers_of_parametric_coordinates_for_splines} COPYONLY)
configure_file(invalid_splines.log ${log_invalid_splines} COPYONLY)
configure_file(sampler.in.log ${log_sampler})
configure_file(config_log.in.hpp config_log.hpp)

add_executable(test_input_output ${TEST_SOURCES})
target_include_directories(test_input_output PRIVATE $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>)
target_include_directories(test_input_output SYSTEM PRIVATE ${GMOCK_INCLUDE_DIRS})
target_link_libraries(test_input_output PRIVATE utilities splines input_output ${GMOCK_LDFLAGS} mock_splines)
target_compile_definitions(test_input_output PRIVATE ${GMOCK_CFLAGS_OTHER})
target_compile_options(test_input_output PRIVATE ${COMPILE_OPTIONS})

add_test(NAME input_output_test COMMAND test_input_output)
