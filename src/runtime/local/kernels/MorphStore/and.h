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


#ifndef DAPHNE_PROTOTYPE_AND_H
#define DAPHNE_PROTOTYPE_AND_H

#include <cstdint>

#include <core/storage/column.h>
#include <core/operators/otfly_derecompr/intersect.h>
#include "core/morphing/uncompr.h"
#include <runtime/local/kernels/BinaryOpCode.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datagen/GenGivenVals.h>

#include <ir/daphneir/Daphne.h>


template<class DTRes, class DTLhs, class DTRhs>
class AndMorph {
public:
    static void apply(DTRes * & res, const DTLhs * inLhs, const DTRhs inRhs, DCTX(ctx)) = delete;
};

template<class DTRes, class DTLhs, class DTRhs, typename ve=vectorlib::scalar<vectorlib::v64<uint64_t>>>
void andMorph(DTRes * & res, const DTLhs * inLhs, const DTRhs * inRhs, DCTX(ctx)) {
    AndMorph<DTRes, DTLhs, DTRhs>::apply(res, inLhs, inRhs, ctx);
}


template<typename VTRes, typename VTLhs, typename VTRhs>
class AndMorph<DenseMatrix<VTRes>, DenseMatrix<VTLhs>, DenseMatrix<VTRhs>> {
public:
    template<typename ve=vectorlib::scalar<vectorlib::v64<uint64_t>>>
    static void apply(DenseMatrix<VTRes> * & res, const DenseMatrix<VTLhs> * inLhs, const DenseMatrix<VTRhs> * inRhs, DCTX(ctx)) {
        assert((inLhs->getNumRows() == inRhs->getNumRows()) && "number of input rows not the same");

        auto colDataLeft = reinterpret_cast<const uint64_t*>(inLhs->getValues());
        std::vector<uint64_t> dataLeft;
        for (uint64_t i = 0; i < inLhs->getNumRows(); ++i) {
            if (colDataLeft[i] != 0){
                dataLeft.push_back(i);
            }
        }
        auto dataLeftAlig = new (std::align_val_t(64)) uint64_t[dataLeft.size()];
        for (uint64_t i = 0; i < dataLeft.size(); ++i) {
            dataLeftAlig[i] = dataLeft[i];
        }
        const morphstore::column<morphstore::uncompr_f> * const opColLeft = new morphstore::column<morphstore::uncompr_f>(sizeof(uint64_t) * dataLeft.size(), dataLeftAlig);

        auto colDataRight = reinterpret_cast<const uint64_t*>(inRhs->getValues());
        std::vector<uint64_t> dataRight;
        for (uint64_t i = 0; i < inLhs->getNumRows(); ++i) {
            if (colDataRight[i] != 0){
                dataRight.push_back(i);
            }
        }
        auto dataRightAlig = new (std::align_val_t(64)) uint64_t[dataRight.size()];
        for (uint64_t i = 0; i < dataRight.size(); ++i) {
            dataRightAlig[i] = dataRight[i];
        }
        const morphstore::column<morphstore::uncompr_f> * const opColRight = new morphstore::column<morphstore::uncompr_f>(sizeof(uint64_t) * dataRight.size(), dataRightAlig);

        morphstore::column<morphstore::uncompr_f> *result;

        result = const_cast<morphstore::column<morphstore::uncompr_f> *>(morphstore::intersect_sorted<ve, morphstore::uncompr_f, morphstore::uncompr_f, morphstore::uncompr_f>(opColLeft, opColRight));
        /// Change the persistence type to disable the deletion and deallocation of the data.
        //result->set_persistence_type(morphstore::storage_persistence_type::externalScope);
        

        VTRes * ptr = result->get_data();

        //std::shared_ptr<VTRes[]> shrdPtr(new (std::align_val_t(64)) VTRes[result->get_count_values()]);//(ptr);
        std::vector<VTRes> data;
        for (uint64_t i = 0; i < inLhs->getNumRows(); ++i) {
            if (ptr[0] == i){
                data.push_back(1);
                ptr++;
            } else {
                data.push_back(0);
            }
        }
        if (data.size() == 0) {
            res = DataObjectFactory::create<DenseMatrix<VTRes>>(0, 1, false);
        } else {
            res = genGivenVals<DenseMatrix<VTRes>>(data.size(), data);
        }

        delete result, delete opColLeft, delete opColRight;
    }

};
#endif //DAPHNE_PROTOTYPE_AND_H