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
#include <mlir/IR/Location.h>

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

#include <runtime/local/kernels/MorphStore/sum.h>

TEST_CASE("Morphstore Sum: Test the operator for empty input", TAG_KERNELS) {
    /// Data generation
    auto lhs_col0 = DataObjectFactory::create<DenseMatrix<uint64_t>>(0, 1, false);
    
    uint64_t expectedResult;
    /// create expected result set
    {
        uint64_t sum = 0;
        for (uint64_t i = 0; i < lhs_col0->getNumRows(); ++ i) {
            sum += lhs_col0->get(i, 0);
        }
        /// create result data
        expectedResult = sum;
    }


    uint64_t result = aggMorph<uint64_t, DenseMatrix<uint64_t>>(AggOpCode::SUM, lhs_col0, nullptr);

    /// test if result matches expected result
    CHECK(result == expectedResult);

    /// cleanup
    DataObjectFactory::destroy(lhs_col0);

}

TEST_CASE("Morphstore Sum: Test the AggSum operation", TAG_KERNELS) {
    /// Data generation
    auto lhs_col0 = genGivenVals<DenseMatrix<uint64_t>>(10,  { 0, 11, 20, 33, 44, 55, 60, 77, 88, 99});

    uint64_t expectedResult;
    /// create expected result set
    {
        uint64_t sum = 0;
        for (uint64_t i = 0; i < lhs_col0->getNumRows(); ++ i) {
            sum += lhs_col0->get(i, 0);
        }
        /// create result data
        expectedResult = sum;
    }


    uint64_t result = aggMorph<uint64_t, DenseMatrix<uint64_t>>(AggOpCode::SUM, lhs_col0, nullptr);

    /// test if result matches expected result
    CHECK(result == expectedResult);

    /// cleanup
    DataObjectFactory::destroy(lhs_col0);

}