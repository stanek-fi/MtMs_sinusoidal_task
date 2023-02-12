FFNN <- nn_module(
    initialize = function(input_size, layer_sizes, layer_transforms, layer_biases = NULL, layer_dropout_ps = NULL, skip_layer = FALSE) {
        self$input_size <- input_size
        self$layer_sizes <- layer_sizes
        self$layer_transforms <- layer_transforms
        layer_sizes_combined <- c(input_size, layer_sizes)

        if (is.null(layer_biases)) {
            layer_biases <- rep(T, length(self$layer_sizes))
        } else {
            if (!(length(layer_biases) == length(self$layer_sizes))) {
                stop("invalid layer_biases parameter")
            }
        }

        if (is.null(layer_dropout_ps)) {
            layer_dropout_ps <- rep(0, length(self$layer_sizes))
        } else {
            if (!(length(layer_dropout_ps) == length(self$layer_sizes))) {
                stop("invalid layer_dropout_ps parameter")
            }
        }
        self$layer_dropout_ps <- layer_dropout_ps

        for (i in seq_along(self$layer_sizes)) {
            self[[str_c("layer_", i)]] <- nn_linear(layer_sizes_combined[i], layer_sizes_combined[i + 1], bias = layer_biases[i])
        }

        if (skip_layer) {
            self$skip_layer <- nn_linear(layer_sizes_combined[1], self$layer_sizes[length(self$layer_sizes)], bias = T)
        }
    },
    forward = function(x, horizon = 1) {
        y <- x
        for (h in 1:max(horizon)) {
            x <- y[, 1:self$input_size]

            if (!is.null(self$skip_layer)) {
                xskip <- self$skip_layer(x)
            }

            for (i in seq_along(self$layer_sizes)) {
                x <- self$layer_transforms[[i]](self[[str_c("layer_", i)]](x))
                x <- nnf_dropout(x, p = self$layer_dropout_ps[[i]], training = self$training)
            }

            if (!is.null(self$skip_layer)) {
                x <- x + xskip
            }

            y <- torch_cat(list(x, y), dim = 2)
        }
        y <- y[, (max(horizon) + 1 - rev(horizon)), drop = F]
        return(y)
    },
    fforward = function(x, state, horizon = 1) {
        y <- x
        for (h in 1:max(horizon)) {
            x <- y[, 1:self$input_size]

            if (!is.null(self$skip_layer)) {
                xskip <- nnf_linear(x, weight = state[[str_c("skip_layer.weight")]], bias = state[[str_c("skip_layer.bias")]])
            }

            for (i in seq_along(self$layer_sizes)) {
                x <- self$layer_transforms[[i]](nnf_linear(x, weight = state[[str_c("layer_", i, ".weight")]], bias = state[[str_c("layer_", i, ".bias")]]))
                x <- nnf_dropout(x, p = self$layer_dropout_ps[[i]], training = self$training)
            }

            if (!is.null(self$skip_layer)) {
                x <- x + xskip
            }

            y <- torch_cat(list(x, y), dim = 2)
        }
        y <- y[, (max(horizon) + 1 - rev(horizon)), drop = F]
        return(y)
    }
)
