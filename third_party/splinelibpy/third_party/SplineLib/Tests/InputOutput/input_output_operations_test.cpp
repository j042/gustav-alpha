/* Copyright (c) 2018–2021 SplineLib

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE. */

#include <gtest/gtest.h>
#include "Sources/InputOutput/input_output_operations.hpp"
#include "Sources/Utilities/error_handling.hpp"
#include "Tests/InputOutput/config_iges.hpp"
#include "Tests/InputOutput/config_irit.hpp"
#include "Tests/InputOutput/config_xml.hpp"

namespace splinelib::tests::input_output::input_output_operations {

using namespace sources::input_output::input_output_operations;  // NOLINT(build/namespaces)
using sources::input_output::kModeIn;

TEST(InputOutputOperationsSuite, Open) {
  using sources::input_output::kModeAppend, sources::input_output::kModeOut;

  String const kIges{"out.iges"}, kIrit{"out.itd"}, kXml{"out.xml"};

  EXPECT_TRUE(Open<InputStream>(iges_file_one, kModeIn).is_open());
  EXPECT_TRUE(Open<InputStream>(iges_file_two, kModeIn).is_open());
  EXPECT_TRUE(Open<InputStream>(irit_file, kModeIn).is_open());
  EXPECT_TRUE(Open<InputStream>(xml_file, kModeIn).is_open());

  EXPECT_TRUE(Open<OutputStream>(kIges, kModeOut).is_open());
  EXPECT_TRUE(Open<OutputStream>(kIges, kModeAppend).is_open());
  remove(kIges.c_str());
  EXPECT_TRUE(Open<OutputStream>(kIrit, kModeOut).is_open());
  EXPECT_TRUE(Open<OutputStream>(kIrit, kModeAppend).is_open());
  remove(kIrit.c_str());
  EXPECT_TRUE(Open<OutputStream>(kXml, kModeOut).is_open());
  EXPECT_TRUE(Open<OutputStream>(kXml, kModeAppend).is_open());
  remove(kXml.c_str());
}

#ifndef NDEBUG
TEST(InputOutputOperationsSuite, ThrowIfFileCannotBeOpened) {
  EXPECT_THROW(Open<InputStream>("a", kModeIn), RuntimeError);
}
#endif

}  // namespace splinelib::tests::input_output::input_output_operations
