/**********************************************************************************************
 * Copyright (C) 2019 by MorphStore-Team                                                      *
 *                                                                                            *
 * This file is part of MorphStore - a compression aware vectorized column store.             *
 *                                                                                            *
 * This program is free software: you can redistribute it and/or modify it under the          *
 * terms of the GNU General Public License as published by the Free Software Foundation,      *
 * either version 3 of the License, or (at your option) any later version.                    *
 *                                                                                            *
 * This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;  *
 * without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  *
 * See the GNU General Public License for more details.                                       *
 *                                                                                            *
 * You should have received a copy of the GNU General Public License along with this program. *
 * If not, see <http://www.gnu.org/licenses/>.                                                *
 **********************************************************************************************/

#include <tags.h>
#include <vector>
#include <cstdint>
#include <iostream>

#include <catch.hpp>

#include <runtime/local/datagen/GenGivenVals.h>
#include <runtime/local/datastructures/ValueTypeUtils.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datastructures/Frame.h>
#include <runtime/local/kernels/CheckEq.h>

#include <runtime/local/kernels/MorphStore/project.h>

TEST_CASE("Morphstore Project: Test the operator", TAG_KERNELS) {
    /// Data generation
    auto lhs_col0 = genGivenVals<DenseMatrix<uint64_t>>(10, { 0,  1,  2,  3,  4,  5,  6,  7,  8,  9});
    auto lhs_col1 = genGivenVals<DenseMatrix<uint64_t>>(10,  { 0, 11, 22, 33, 44, 55, 66, 77, 88, 99});
    std::vector<Structure *> lhsCols = {lhs_col0, lhs_col1};
    std::string lhsLabels[] = {"R.idx", "R.a"};
    auto lhs = DataObjectFactory::create<Frame>(lhsCols, lhsLabels);

    auto pos = genGivenVals<DenseMatrix<uint64_t>>(10, {   1,  0,  0,  1,  0,  1,  0,  1,  0,  1});


    Frame * expectedResult;
    /// create expected result set
    {
        std::vector<uint64_t> er_col0_val;
        std::vector<uint64_t> er_col1_val;

        for (uint64_t innerLoop = 0; innerLoop < lhs_col0->getNumRows(); ++ innerLoop) {
            /// condition to check
            if (pos->get(innerLoop, 0) == 1) {
                er_col0_val.push_back(lhs_col0->get(innerLoop, 0));
                er_col1_val.push_back(lhs_col1->get(innerLoop, 0));
            }
        }
        uint64_t size = er_col0_val.size();
        auto er_col0 = genGivenVals<DenseMatrix<uint64_t>>(size, er_col0_val);
        auto er_col1 = genGivenVals<DenseMatrix<uint64_t>>(size, er_col1_val);
        std::string labels[] = {"R.idx", "R.a"};
        /// create result data
        expectedResult = DataObjectFactory::create<Frame>(
                std::vector<Structure*>{er_col0,er_col1},
                labels);
        /// cleanup
        DataObjectFactory::destroy(er_col0, er_col1);
        DataObjectFactory::destroy(lhs_col0, lhs_col1);
    }

    /// test execution
    Frame * resultFrame = nullptr;

    project(resultFrame, lhs, pos, nullptr);

    resultFrame->print(std::cout);
    expectedResult->print(std::cout);

    /// test if result matches expected result
    CHECK(*resultFrame == *expectedResult);

    /// cleanup
    DataObjectFactory::destroy(resultFrame, expectedResult, lhs, pos);

}