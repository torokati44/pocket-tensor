/*
 * pocket-tensor (c) 2019 Gustavo Valiente gustavo.valiente@protonmail.com
 * Kerasify (c) 2016 Robert W. Rose
 *
 * MIT License, see LICENSE file.
 */

#ifndef PT_AVERAGE_POOLING_1D_LAYER_H
#define PT_AVERAGE_POOLING_1D_LAYER_H

#include "pt_layer.h"

namespace pt
{

class AveragePooling1DLayer : public Layer
{

public:
    static std::unique_ptr<AveragePooling1DLayer> create(std::istream& stream);

    bool apply(LayerData& layerData) const final;

protected:
    int _poolSize;

    AveragePooling1DLayer(int poolSize) noexcept :
        _poolSize(poolSize)
    {
    }
};

}

#endif
