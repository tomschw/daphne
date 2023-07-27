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
//#include <runtime/local/kernels/CheckEq.h>

#include <runtime/local/kernels/MorphStore/calc.h>

TEST_CASE("Morphstore Calc: Test the operator with empty input", TAG_KERNELS) {
    /// Data generation
    auto lhs_col0 = DataObjectFactory::create<DenseMatrix<uint64_t>>(0, 1, false);

    auto rhs_col0 = DataObjectFactory::create<DenseMatrix<uint64_t>>(0, 1, false);


    DenseMatrix<uint64_t> * expectedResult;
    /// create expected result set
    {
        expectedResult = DataObjectFactory::create<DenseMatrix<uint64_t>>(0, 1, false);
    }

    /// test execution
    DenseMatrix<uint64_t> * resultMatrix = nullptr;

    calcBinary<DenseMatrix<uint64_t>, DenseMatrix<uint64_t>,DenseMatrix<uint64_t>>(BinaryOpCode::ADD, resultMatrix, lhs_col0, rhs_col0, nullptr);

    /// test if result matches expected result
    //CHECK(*resultMatrix == *expectedResult);

    /// cleanup
    DataObjectFactory::destroy(expectedResult, lhs_col0, rhs_col0);

}

TEST_CASE("Morphstore Calc: Test the Calc operation with the Add Operation", TAG_KERNELS) {
    /// Data generation
    auto lhs_col0 = genGivenVals<DenseMatrix<uint64_t>>(10,  { 0, 11, 20, 33, 44, 55, 60, 77, 88, 99});

    auto rhs_col0 = genGivenVals<DenseMatrix<uint64_t>>(10,  { 100, 90, 80, 70, 60, 50, 40, 30, 20, 10});

    DenseMatrix<uint64_t> * expectedResult;
    /// create expected result set
    {
        std::vector<uint64_t> er_col0_val;
        for (uint64_t outerLoop = 0; outerLoop < lhs_col0->getNumRows(); ++ outerLoop) {
                er_col0_val.push_back(lhs_col0->get(outerLoop, 0) + rhs_col0->get(outerLoop, 0));
        }
        uint64_t size = er_col0_val.size();
        expectedResult = genGivenVals<DenseMatrix<uint64_t>>(size, er_col0_val);
    }

    /// test execution
    DenseMatrix<uint64_t> * resultMatrix = nullptr;

    calcBinary<DenseMatrix<uint64_t>, DenseMatrix<uint64_t>,DenseMatrix<uint64_t>>(BinaryOpCode::ADD, resultMatrix, lhs_col0, rhs_col0, nullptr);

    resultMatrix->print(std::cout);
    expectedResult->print(std::cout);

    /// test if result matches expected result
    //CHECK(*resultMatrix == *expectedResult);

    /// cleanup
    DataObjectFactory::destroy(expectedResult, lhs_col0, rhs_col0);

}

TEST_CASE("Morphstore Calc: Test the Calc operation with the Sub Operation", TAG_KERNELS) {
    /// Data generation
    auto lhs_col0 = genGivenVals<DenseMatrix<uint64_t>>(10,  { 0, 11, 20, 33, 44, 55, 60, 77, 88, 99});

    auto rhs_col0 = genGivenVals<DenseMatrix<uint64_t>>(10,  { 100, 90, 80, 70, 60, 50, 40, 30, 20, 10});

    DenseMatrix<uint64_t> * expectedResult;
    /// create expected result set
    {
        std::vector<uint64_t> er_col0_val;
        for (uint64_t outerLoop = 0; outerLoop < lhs_col0->getNumRows(); ++ outerLoop) {
                er_col0_val.push_back(lhs_col0->get(outerLoop, 0) - rhs_col0->get(outerLoop, 0));
        }
        uint64_t size = er_col0_val.size();
        expectedResult = genGivenVals<DenseMatrix<uint64_t>>(size, er_col0_val);
    }

    /// test execution
    DenseMatrix<uint64_t> * resultMatrix = nullptr;

    calcBinary<DenseMatrix<uint64_t>, DenseMatrix<uint64_t>,DenseMatrix<uint64_t>>(BinaryOpCode::SUB, resultMatrix, lhs_col0, rhs_col0, nullptr);

    /// test if result matches expected result
    //CHECK(*resultMatrix == *expectedResult);

    /// cleanup
    DataObjectFactory::destroy( expectedResult, lhs_col0, rhs_col0);

}

TEST_CASE("Morphstore Calc: Test the Calc operation with the Mul Operation", TAG_KERNELS) {
    /// Data generation
    auto lhs_col0 = genGivenVals<DenseMatrix<uint64_t>>(10,  { 0, 11, 20, 33, 44, 55, 60, 77, 88, 99});

    auto rhs_col0 = genGivenVals<DenseMatrix<uint64_t>>(10,  { 100, 90, 80, 70, 60, 50, 40, 30, 20, 10});

    DenseMatrix<uint64_t> * expectedResult;
    /// create expected result set
    {
        std::vector<uint64_t> er_col0_val;
        for (uint64_t outerLoop = 0; outerLoop < lhs_col0->getNumRows(); ++ outerLoop) {
                er_col0_val.push_back(lhs_col0->get(outerLoop, 0) * rhs_col0->get(outerLoop, 0));
        }
        uint64_t size = er_col0_val.size();
        expectedResult = genGivenVals<DenseMatrix<uint64_t>>(size, er_col0_val);
    }

    /// test execution
    DenseMatrix<uint64_t> * resultMatrix = nullptr;

    calcBinary<DenseMatrix<uint64_t>, DenseMatrix<uint64_t>,DenseMatrix<uint64_t>>(BinaryOpCode::MUL, resultMatrix, lhs_col0, rhs_col0, nullptr);

    /// test if result matches expected result
    //CHECK(*resultMatrix == *expectedResult);

    /// cleanup
    DataObjectFactory::destroy( expectedResult, lhs_col0, rhs_col0);
}

TEST_CASE("Morphstore Calc: Test the Calc operation with the Div Operation", TAG_KERNELS) {
    /// Data generation
    auto lhs_col0 = genGivenVals<DenseMatrix<uint64_t>>(10,  { 1, 11, 20, 33, 44, 55, 60, 77, 88, 99});

    auto rhs_col0 = genGivenVals<DenseMatrix<uint64_t>>(10,  { 100, 90, 80, 70, 60, 50, 40, 30, 20, 10});

    DenseMatrix<uint64_t> * expectedResult;
    /// create expected result set
    {
        std::vector<uint64_t> er_col0_val;
        for (uint64_t outerLoop = 0; outerLoop < lhs_col0->getNumRows(); ++ outerLoop) {
                er_col0_val.push_back(lhs_col0->get(outerLoop, 0) / rhs_col0->get(outerLoop, 0));
        }
        uint64_t size = er_col0_val.size();
        expectedResult = genGivenVals<DenseMatrix<uint64_t>>(size, er_col0_val);
    }

    /// test execution
    DenseMatrix<uint64_t> * resultMatrix = nullptr;

    calcBinary<DenseMatrix<uint64_t>, DenseMatrix<uint64_t>,DenseMatrix<uint64_t>>(BinaryOpCode::DIV, resultMatrix, lhs_col0, rhs_col0, nullptr);

    /// test if result matches expected result
    //CHECK(*resultMatrix == *expectedResult);

    /// cleanup
    DataObjectFactory::destroy( expectedResult, lhs_col0, rhs_col0);

}
