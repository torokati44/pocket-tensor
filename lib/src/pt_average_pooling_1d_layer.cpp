/*
 * pocket-tensor (c) 2019 Gustavo Valiente gustavo.valiente@protonmail.com
 * Kerasify (c) 2016 Robert W. Rose
 *
 * MIT License, see LICENSE file.
 */

#include "pt_average_pooling_1d_layer.h"

#include <array>
#include "pt_parser.h"
#include "pt_dispatcher.h"
#include "pt_layer_data.h"
#include "pt_add.h"

namespace pt
{

namespace
{
    template<class AddType>
    void averageImpl(int poolSize, LayerData& layerData)
    {
        struct Task
        {
            int poolSize;
            LayerData* layerData;
            int threads;
            int taskId;

            void operator()() noexcept
            {
                const Tensor& in = layerData->in;
                Tensor& out = layerData->out;

                const auto& iw = in.getDims();
                const auto& ow = out.getDims();
                auto inInc2 = int(iw[1]);
                auto inInc = inInc2 * poolSize;
                auto outInc2 = int(iw[1] * ow[1]);
                auto outInc = outInc2 * int(ow[0]);

                auto inData = in.getData().data();
                auto outData = const_cast<Tensor::Type*>(out.getData().data());
                AddType sum;

                int its = outInc / outInc2;
                int taskIts = its / threads;
                int taskBegin = taskIts * taskId;
                int taskEnd;

                if(taskId == threads - 1)
                {
                    taskEnd = its;
                }
                else
                {
                    taskEnd = taskBegin + taskIts;
                }

                inData += taskIts * taskId;

                for(auto outIt = outData + (taskBegin * outInc2), outEnd = outData + (taskEnd * outInc2);
                    outIt != outEnd; outIt += outInc2)
                {
                    auto inIt = inData;

                    for(auto outIt2 = outIt, outEnd2 = outIt + outInc2; outIt2 != outEnd2; outIt2 += inInc2)
                    {
                        for(auto inIt3 = inIt, inEnd3 = inIt + inInc; inIt3 != inEnd3; inIt3 += inInc2)
                        {
                            add(&*inIt3, &*outIt2, inInc2);
                        }

                        inIt += inInc;
                    }
                }

                // TODO divide by poolSize (and check the above)
            }
        };

        std::array<Task, PT_MAX_CPU_THREADS> tasks;
        Dispatcher& dispatcher = layerData.dispatcher;
        auto threads = int(dispatcher.threads());

        for(int taskId = 0; taskId != threads; ++taskId)
        {
            Task& task = tasks[std::size_t(taskId)];
            task = Task{ poolSize, &layerData, threads, taskId };
            dispatcher.add([&task]{ task(); });
        }

        dispatcher.join();
    }
}

std::unique_ptr<AveragePooling1DLayer> AveragePooling1DLayer::create(std::istream& stream)
{
    unsigned int poolSize = 0;

    if(! Parser::parse(stream, poolSize))
    {
        PT_LOG_ERROR << "Pool size parse failed" << std::endl;
        return nullptr;
    }

    return std::unique_ptr<AveragePooling1DLayer>(new AveragePooling1DLayer(int(poolSize)));
}

bool AveragePooling1DLayer::apply(LayerData& layerData) const
{
    const Tensor& in = layerData.in;
    const auto& iw = in.getDims();

    if(iw.size() != 2)
    {
        PT_LOG_ERROR << "Input tensor dims count must be 2" <<
                            " (input dims: " << VectorPrinter<std::size_t>{ iw } << ")" << std::endl;
        return false;
    }

    Tensor& out = layerData.out;
    out.resize(iw[0] / std::size_t(_poolSize), iw[1]);
    out.fill(-std::numeric_limits<Tensor::Type>::infinity());

    Dispatcher& dispatcher = layerData.dispatcher;
    auto threads = int(dispatcher.threads());
    auto threadSize = int(iw[1]) / threads;

    if(PT_LOOP_UNROLLING_ENABLE && threadSize && threadSize % (Tensor::VectorSize * 2) == 0)
    {
        averageImpl<Vector2Add>(_poolSize, layerData);
    }
    else if(threadSize && threadSize % Tensor::VectorSize == 0)
    {
        averageImpl<VectorAdd>(_poolSize, layerData);
    }
    else
    {
        averageImpl<ScalarAdd>(_poolSize, layerData);
    }

    return true;
}

}
