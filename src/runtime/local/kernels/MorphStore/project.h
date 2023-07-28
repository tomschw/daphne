/*
 * Copyright 2021 The DAPHNE Consortium
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef SRC_RUNTIME_LOCAL_KERNELS_MORPHSTORE_PROJECT_H
#define SRC_RUNTIME_LOCAL_KERNELS_MORPHSTORE_PROJECT_H

#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datastructures/Frame.h>
#include <runtime/local/datastructures/ValueTypeCode.h>
#include <runtime/local/datastructures/ValueTypeUtils.h>

#include <core/storage/column.h>
#include "core/operators/otfly_derecompr/project.h"
#include "core/morphing/uncompr.h"

#include <stdexcept>

#include <cstddef>
#include <cstdint>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template<class DTRes, class DTArg, typename VTSel>
struct Project {
    static void apply(DTRes *& res, const DTArg * arg, const DenseMatrix<VTSel> * sel, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template<class DTRes, class DTArg, typename VTSel>
void projectMorph(DTRes *& res, const DTArg * arg, const DenseMatrix<VTSel> * sel, DCTX(ctx)) {
    Project<DTRes, DTArg, VTSel>::apply(res, arg, sel, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// Frame <- Frame
// ----------------------------------------------------------------------------

template<typename VTSel>
struct Project<Frame, Frame, VTSel> {
    template<typename ve=vectorlib::scalar<vectorlib::v64<uint64_t>>>
    static void apply(Frame *& res, const Frame * arg, const DenseMatrix<VTSel> * sel, DCTX(ctx)) {
        auto colData =  reinterpret_cast<uint64_t const *>(sel->getValues());
        std::vector<uint64_t> positions;
        for (size_t i = 0; i < sel->getNumRows(); i++) {
            if(colData[i] == 1) {
                positions.push_back(i);
            }
        }
        auto aligned_pos = new (std::align_val_t(64)) uint64_t[positions.size()];
        for (size_t i = 0; i < positions.size(); i++) {
            aligned_pos[i] = positions[i];
        }
        morphstore::column<morphstore::uncompr_f> * const selectCol = new morphstore::column<morphstore::uncompr_f>(sizeof(uint64_t) * positions.size(), aligned_pos);

        const std::string *columnLabels = arg->getLabels();

        std::vector<Structure *> resultCols = {};

        for (size_t i = 0; i < arg->getNumCols(); ++ i) {
            auto colProjData = static_cast<uint64_t const *>(arg->getColumnRaw(arg->getColumnIdx(*(columnLabels + i))));
            auto colProj = new morphstore::column<morphstore::uncompr_f>(sizeof(uint64_t) * arg->getNumRows(), colProjData);
            auto* projCol = const_cast<morphstore::column<morphstore::uncompr_f> *>(morphstore::my_project_wit_t<ve, morphstore::uncompr_f, morphstore::uncompr_f,
                    morphstore::uncompr_f>::apply(
                    colProj, selectCol));

            /// Change the persistence type to disable the deletion and deallocation of the data.
            projCol->set_persistence_type(morphstore::storage_persistence_type::externalScope);

            uint64_t * ptr = projCol->get_data();

            std::shared_ptr<uint64_t[]> shrdPtr(ptr);

            auto result = DataObjectFactory::create<DenseMatrix<uint64_t>>(projCol->get_count_values(), 1, shrdPtr);

            resultCols.push_back(result);
            delete projCol, delete colProj;
        }

        res = DataObjectFactory::create<Frame>(resultCols, columnLabels);

    }
};


#endif //SRC_RUNTIME_LOCAL_KERNELS_MORPHSTORE_PROJECT_H