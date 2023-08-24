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


#ifndef DAPHNE_PROTOTYPE_SUM_H
#define DAPHNE_PROTOTYPE_SUM_H

#include <cstdint>

#include <core/storage/column.h>
#include <core/operators/otfly_derecompr/agg_sum_all.h>
#include "core/morphing/uncompr.h"
#include <runtime/local/kernels/AggOpCode.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/Frame.h>
#include <runtime/local/datastructures/Matrix.h>

template<typename VTRes, class DTIn, typename ve>
class AggSum {
public:
    static VTRes apply(AggOpCode agg, const DTIn * in, DCTX(ctx)) = delete;
};

template<typename VTRes, class DTIn, typename ve>
VTRes aggMorph(AggOpCode agg, const DTIn * in, DCTX(ctx)) {
    return AggSum<VTRes, DTIn, ve>::apply(agg, in, ctx);
}

template<typename VTRes, typename VTArg, typename ve>
class AggSum<VTRes, DenseMatrix<VTArg>, ve> {
public:
    static VTRes apply(AggOpCode agg, const DenseMatrix<VTArg> * in, DCTX(ctx)) {
        auto colData = reinterpret_cast<const uint64_t*>(in->getValues());
        const morphstore::column<morphstore::uncompr_f> * const aggCol = new morphstore::column<morphstore::uncompr_f>(sizeof(uint64_t) * in->getNumRows(), colData);

        morphstore::column<morphstore::uncompr_f> * aggResult = nullptr;
        if(AggOpCode::SUM == agg) {
            aggResult = const_cast<morphstore::column<morphstore::uncompr_f> *>(morphstore::agg_sum_all<ve, morphstore::uncompr_f, morphstore::uncompr_f>(aggCol));
        }
        /// Change the persistence type to disable the deletion and deallocation of the data.
        //aggResult->set_persistence_type(morphstore::storage_persistence_type::externalScope);

        VTRes * ptr = aggResult->get_data();

        /**std::shared_ptr<uint64_t[]> shrdPtr(ptr);

        auto result = DataObjectFactory::create<DenseMatrix<uint64_t>>(aggResult->get_count_values(), 1, shrdPtr);

        const std::string columnLabels[] = {"Agg_sum"};

        std::vector<Structure *> resultCols = {result};

        res = DataObjectFactory::create<Frame>(resultCols, columnLabels); **/

        return *ptr;

        delete aggResult, delete aggCol;
    }

};
#endif //DAPHNE_PROTOTYPE_SUM_H
