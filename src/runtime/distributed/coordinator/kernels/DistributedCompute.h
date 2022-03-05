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

#ifndef SRC_RUNTIME_DISTRIBUTED_COORDINATOR_KERNELS_DISTRIBUTEDCOMPUTE_H
#define SRC_RUNTIME_DISTRIBUTED_COORDINATOR_KERNELS_DISTRIBUTEDCOMPUTE_H

#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/distributed/coordinator/datastructures/Handle.h>

#include <runtime/distributed/proto/worker.pb.h>
#include <runtime/distributed/proto/worker.grpc.pb.h>

#include <runtime/distributed/worker/ProtoDataConverter.h>

#include <cassert>
#include <cstddef>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template<class DTRes, class DTArgs>
struct DistributedCompute
{
    static void apply(Handle<DTRes> *&res, Handle<DTArgs> **args, size_t num_args, const char *mlirCode, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template<class DTRes, class DTArgs>
void distributedCompute(Handle<DTRes> *&res, Handle<DTArgs> *args, size_t num_args, const char *mlirCode, DCTX(ctx))
{
    DistributedCompute<DTRes, DTArgs>::apply(res, args, num_args, mlirCode, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

template<class DTRes>
struct DistributedCompute<DTRes, Structure>
{
    static void apply(Handle<DTRes> *&res,
                      Handle<Structure> *args,
                      size_t num_args,
                      const char *mlirCode,
                      DCTX(ctx))
    {
        if(res == nullptr) {
            auto envVar = std::getenv("DISTRIBUTED_WORKERS");
            // assert(envVar && "Environment variable has to be set");
            std::string workersStr(envVar);            
            std::string delimiter(",");

            size_t pos;
            std::vector<std::string> workers;
            while ((pos = workersStr.find(delimiter)) != std::string::npos) {
                workers.push_back(workersStr.substr(0, pos));
                workersStr.erase(0, pos + delimiter.size());
            }
            workers.push_back(workersStr);
            res = DataObjectFactory::create<Handle<DTRes>>(workers);
        }

        struct StoredInfo {
            std::string addr;
            DistributedIndex *ix;
        };
        DistributedCaller<StoredInfo, distributed::Task, distributed::ComputeResult> caller;
        auto map = args->getMap();
        DistributedIndex *ix = new DistributedIndex(0, 0);
        for (auto &pair : map){
            auto addr = pair.first;
            auto dataVector = pair.second.distributedDataArray;
            distributed::Task task;
            
            // Pass all the nessecary arguments for the pipeline
            for (auto data : dataVector){
                *task.add_inputs()->mutable_stored() = data.getData();
            }
            task.set_mlir_code(mlirCode);
            StoredInfo storedInfo ({addr, ix});

            // TODO the next DistributedIndex must be decided based on combines (probably...)
            // For now, assume rows combining and set DistrIdx accordingly
            ix = new DistributedIndex(ix->getRow() + 1, ix->getCol());
            // TODO for now resuing channels seems to slow things down... 
            // It is faster if we generate channel for each call and let gRPC handle resources internally
            // We might need to change this in the future and re-use channels ( data.getChannel() )
            caller.asyncComputeCall(addr, storedInfo, task);
        }
        // Get Results
        while (!caller.isQueueEmpty()){
            auto response = caller.getNextResult();
            auto addr = response.storedInfo.addr;
            auto ix = response.storedInfo.ix;
            
            auto computeResult = response.result;
            // Recieve all outputs and store it to Handle_v2
            for (size_t i = 0; i < computeResult.outputs_size(); i++){
                DistributedData data(*ix, computeResult.outputs(i).stored());
                res->insertData(addr, data);
            }
        }
        // res = new Handle<DTRes>(resMap, resultRows, resultColumns);        
    }
};

#endif //SRC_RUNTIME_DISTRIBUTED_COORDINATOR_KERNELS_DISTRIBUTEDCOMPUTE_H