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
#include "Sources/InputOutput/iges.hpp"
#include "Sources/InputOutput/input_output_splines.hpp"
#include "Sources/InputOutput/irit.hpp"
#include "Sources/InputOutput/vtk.hpp"
#include "Sources/InputOutput/xml.hpp"
#include "Sources/Splines/nurbs.hpp"
#include "Sources/Utilities/error_handling.hpp"
#include "Sources/Utilities/std_container_operations.hpp"
#include "Tests/InputOutput/config_iges.hpp"
#include "Tests/InputOutput/config_irit.hpp"
#include "Tests/InputOutput/config_log.hpp"
#include "Tests/InputOutput/config_xml.hpp"

namespace splinelib::tests::input_output::input_output_splines {

using namespace sources::input_output::input_output_splines;  // NOLINT(build/namespaces)
using SplinesRead = sources::input_output::Splines;

class InputOutputSplinesSuite : public testing::Test {
 protected:
  using Index_ = Splines::value_type;

  constexpr static Index_ const kIndex0_{}, kIndex1_{1}, kIndex2_{2}, kIndex3_{3};
  inline static Splines const kFirstAndSecond_{kIndex0_, kIndex1_},
                              kFirstThroughSixth_{kIndex0_, kIndex1_, kIndex2_, kIndex3_, Index{4}, Index{5}};
  inline static String const kFileIges_{"converted.iges"}, kFileIrit_{"converted.itd"}, kFileXml_{"converted.xml"},
                             kFileSampled_{"sampled.vtk"};
  inline static SplinesRead const kIgesOne_{sources::input_output::iges::Read(iges_file_one)},
      kIgesTwo_{sources::input_output::iges::Read(iges_file_two)}, kIrit_{sources::input_output::irit::Read(irit_file)},
      kXml_{sources::input_output::xml::Read(xml_file)};
};

TEST_F(InputOutputSplinesSuite, ConvertSplines) {
  EXPECT_EQ(ConvertSplines(iges_file_one, kFileIrit_), kFirstAndSecond_);
  remove(kFileIrit_.c_str());
  EXPECT_EQ(ConvertSplines(iges_file_two, kFileIrit_), kFirstAndSecond_);
  remove(kFileIrit_.c_str());
  EXPECT_EQ(ConvertSplines(iges_file_one, kFileXml_), kFirstAndSecond_);
  remove(kFileXml_.c_str());
  EXPECT_EQ(ConvertSplines(iges_file_two, kFileXml_), kFirstAndSecond_);
  remove(kFileXml_.c_str());

  EXPECT_EQ(ConvertSplines(irit_file, kFileIges_, kFirstThroughSixth_), (Splines{kIndex0_, kIndex1_, kIndex2_,
                                                                                 kIndex3_}));
  remove(kFileIges_.c_str());
  EXPECT_EQ(ConvertSplines(irit_file, kFileXml_), kFirstThroughSixth_);
  remove(kFileXml_.c_str());

  EXPECT_EQ(ConvertSplines(xml_file, kFileIges_), kFirstAndSecond_);
  remove(kFileIges_.c_str());
  EXPECT_EQ(ConvertSplines(xml_file, kFileIrit_), kFirstAndSecond_);
  remove(kFileIrit_.c_str());
}

TEST_F(InputOutputSplinesSuite, SampleSplines) {
  using NumberOfParametricCoordinates = NumberOfParametricCoordinatesForSplines::value_type;

  constexpr NumberOfParametricCoordinates::value_type kLength10{10};
  NumberOfParametricCoordinates const kEmpty{}, kCurve{kLength10}, kSurface{kLength10, kLength10},
                                      kVolume{kLength10, kLength10, kLength10};
  NumberOfParametricCoordinatesForSplines const kCurveAndSurface{kSurface, kCurve};

  EXPECT_EQ(SampleSplines(iges_file_one, kFileSampled_, kCurveAndSurface), kFirstAndSecond_);
  remove(kFileSampled_.c_str());
  EXPECT_EQ(SampleSplines(iges_file_two, kFileSampled_, kCurveAndSurface), kFirstAndSecond_);
  remove(kFileSampled_.c_str());

  EXPECT_EQ(SampleSplines(irit_file, kFileSampled_, {kCurve, kCurve, kSurface, kSurface, kVolume, kVolume}),
            kFirstThroughSixth_);
  remove(kFileSampled_.c_str());

  EXPECT_EQ(SampleSplines(xml_file, kFileSampled_, {kSurface, kSurface, kEmpty, kEmpty}), kFirstAndSecond_);
  remove(kFileSampled_.c_str());
}

TEST_F(InputOutputSplinesSuite, Append) {
  String const kInput{"# Replace by path to input file!"}, kOutput{"# Replace by path to output file!"};

  EXPECT_NO_THROW(Append(log_invalid_splines, (LogInformation{kInput, kOutput, {}, {}})));
  EXPECT_NO_THROW(Append(log_invalid_splines, (LogInformation{kInput, kOutput, {kIndex0_}, {}})));
  EXPECT_NO_THROW(Append(log_invalid_splines, (LogInformation{kInput, kOutput, {kIndex0_, kIndex2_}, {}})));
}

TEST_F(InputOutputSplinesSuite, Read) {
  using NumberOfParametricCoordinates = NumberOfParametricCoordinatesForSplines::value_type;

  constexpr NumberOfParametricCoordinates::value_type const kLength10{10};
  NumberOfParametricCoordinates const kCurve{kLength10}, kSurface{kLength10, kLength10}, kVolume{kLength10, kLength10,
                                                                                                 kLength10};

  EXPECT_EQ(Read<LogType::kConverter>(log_converter), (LogInformation{irit_file, itd_xml, {}, {}}));
  EXPECT_EQ(Read<LogType::kSampler>(log_sampler), (LogInformation{itd_xml, xml_vtk, kFirstThroughSixth_,
                                                       {kCurve, kCurve, kSurface, kSurface, kVolume, kVolume}}));
}

#ifndef NDEBUG
TEST_F(InputOutputSplinesSuite, ThrowIfVtkFileAsInputOrAsOutputOfConversion) {
  EXPECT_THROW(ConvertSplines(kFileSampled_, kFileSampled_), RuntimeError);
  EXPECT_THROW(ConvertSplines(iges_file_one, kFileSampled_), RuntimeError);
}

TEST_F(InputOutputSplinesSuite, ThrowIfSplinesOrNumbersOfParametricCoordinatesForSplinesAreInvalid) {
  EXPECT_THROW(Read<LogType::kConverter>(log_invalid_splines), RuntimeError);
  EXPECT_THROW(Read<LogType::kSampler>(log_converter), RuntimeError);
  EXPECT_THROW(Read<LogType::kSampler>(log_invalid_numbers_of_parametric_coordinates_for_splines), RuntimeError);
}
#endif

}  // namespace splinelib::tests::input_output::input_output_splines
