/*
 * pocket-tensor (c) 2019 Gustavo Valiente gustavo.valiente@protonmail.com
 * Kerasify (c) 2016 Robert W. Rose
 *
 * MIT License, see LICENSE file.
 */

#include "pt_global_average_pooling_1d_layer.h"

#include <array>
#include "pt_parser.h"
#include "pt_dispatcher.h"
#include "pt_layer_data.h"

namespace pt
{

namespace
{
    void averageImpl(LayerData& layerData)
    {
        struct Task
        {
            LayerData* layerData;
            int threads;
            int taskId;

            void operator()() noexcept
            {
                const Tensor& in = layerData->in;
                Tensor& out = layerData->out;

                const auto& iw = in.getDims();
                auto its = out.getSize();
                auto taskIts = its / std::size_t(threads);
                auto taskBegin = taskIts * std::size_t(taskId);
                std::size_t taskEnd;

                if(taskId == threads - 1)
                {
                    taskEnd = its;
                }
                else
                {
                    taskEnd = taskBegin + taskIts;
                }

                auto l = iw[0];

                for(std::size_t z = taskBegin; z != taskEnd; ++z)
                {
                    Tensor::Type val = 0;

                    for(std::size_t i = 0; i != l; ++i)
                    {
                        val += in(i, z);
                    }

                    out(z) = val / l;
                }
            }
        };

        std::array<Task, PT_MAX_CPU_THREADS> tasks;
        Dispatcher& dispatcher = layerData.dispatcher;
        auto threads = int(dispatcher.threads());

        for(int taskId = 0; taskId != threads; ++taskId)
        {
            Task& task = tasks[std::size_t(taskId)];
            task = Task{ &layerData, threads, taskId };
            dispatcher.add([&task]{ task(); });
        }

        dispatcher.join();
    }
}

std::unique_ptr<GlobalAveragePooling1DLayer> GlobalAveragePooling1DLayer::create(std::istream&)
{
    return std::unique_ptr<GlobalAveragePooling1DLayer>(new GlobalAveragePooling1DLayer());
}

bool GlobalAveragePooling1DLayer::apply(LayerData& layerData) const
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
    out.resize(iw[1]);
    averageImpl(layerData);
    return true;
}

}
