#' Create a meta_module
#'
#' An internal function that creates a simple MetaModule with no hidden layers.
#'
#' @param base_model Base torch model for construction of MtMs. It must have a `fforward(x, state)` method, that is, a functional, state-independent, version of the model. See the helper function `FFNN` for an example of what it might look like. It should also preferably already be trained on the pooled data.
#' @param mesa_parameter_size An integer specifying the number of mesa parameters per task.
#' @param meta_bias A boolean specifying whether the meta module should contain a trainable constant on its output layer connecting to the base model.
#' @param meta_dropout_p A float specifying the probability of dropout in the meta module.
#' @param meta_weight_init A tensor of dimensions `num_non_orphaned_output_nodes` x `mesa_parameter_size` containing the initial values of the weights of the meta module. If it is `NULL`, the weights are initialized randomly.
#' @param meta_connections A string vector of names of parameters of the base model to be connected to the meta module. If it is `NULL`, the meta module connects to all parameters of the base model.
#'
#' @return A meta_module to be used inside of MtMs.

MetaModule <- nn_module(
    initialize = function(base_model, mesa_parameter_size = 1, meta_bias = T, meta_dropout_p = 0, meta_weight_init = NULL, meta_connections = NULL) {
        self$mesa_parameter_size <- mesa_parameter_size
        self$state_size <- base_model$state_size

        if (is.null(meta_connections)) {
            meta_connections <- names(base_model$state_structure)
        } else {
            valid <- meta_connections %in% names(base_model$state_structure)
            if (!all(valid)) {
                stop(paste0("invalid supplied meta_connections:", meta_connections[!valid]))
            }
        }
        connections <- rapply(base_model$state_structure, function(x) torch_tensor(array(F, dim = x)), how = "list")
        for (i in meta_connections) {
            connections[[i]] <- torch_tensor(array(T, dim = dim(connections[[i]])))
        }
        self$connections <- as.array(base_model$vectorize_state(connections))

        if (is.null(meta_weight_init)) {
            INIT_RANGE <- 1
            meta_weight_init <- matrix(runif(sum(self$connections) * self$mesa_parameter_size, -INIT_RANGE, INIT_RANGE), nrow = sum(self$connections))
        } else {
            if (!((nrow(meta_weight_init) == sum(self$connections)) & (ncol(meta_weight_init) == self$mesa_parameter_size))) {
                stop("ivalid supplied meta_weight_init")
            }
        }
        self$meta_weight <- nn_parameter(torch_tensor(meta_weight_init, dtype = torch_float()))

        if (meta_bias) {
            self$meta_bias <- nn_parameter(torch_tensor(rep(0, self$state_size), dtype = torch_float()))
        }

        self$dropout <- nn_dropout(p = meta_dropout_p)
    },
    forward = function(mesa_parameter) {
        base_state_diff <- torch_tensor(rep(0, self$state_size))
        base_state_diff[self$connections] <- nnf_linear(mesa_parameter, self$meta_weight)
        if (!is.null(self$meta_bias)) {
            base_state_diff <- self$meta_bias + base_state_diff
        }
        base_state_diff <- self$dropout(base_state_diff)
        return(base_state_diff)
    }
)
