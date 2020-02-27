/*
 * pocket-tensor (c) 2019 Gustavo Valiente gustavo.valiente@protonmail.com
 * Kerasify (c) 2016 Robert W. Rose
 *
 * MIT License, see LICENSE file.
 */

#ifndef PT_GLOBAL_AVERAGE_POOLING_1D_LAYER_H
#define PT_GLOBAL_AVERAGE_POOLING_1D_LAYER_H

#include "pt_layer.h"

namespace pt
{

class GlobalAveragePooling1DLayer : public Layer
{

public:
    static std::unique_ptr<GlobalAveragePooling1DLayer> create(std::istream& stream);

    bool apply(LayerData& layerData) const final;

protected:
    GlobalAveragePooling1DLayer() = default;
};

}

#endif
