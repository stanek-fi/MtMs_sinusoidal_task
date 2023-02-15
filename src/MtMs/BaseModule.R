#' Create a base_module
#'
#' An internal function that creates a base_module based on base_model.
#'
#' @param base_model Base torch model for construction of MtMs. It must have a `fforward(x, state)` method, that is, a functional, state-independent, version of the model. See the helper function `FFNN` for an example of what it might look like. It should also preferably already be trained on the pooled data.
#'
#' @return A base_module to be used inside of MtMs.

BaseModule <- nn_module(
    initialize = function(base_model) {
        state <- base_model$state_dict()
        self$state_structure <- rapply(state, function(x) dim(x), how = "list")

        self$vectorize_state <- function(state) {
            return(torch_cat(rapply(state, function(x) x$view(-1), how = "unlist")))
        }
        self$unvectorize_state <- function(state, state_structure) {
            counter <- 1
            state <- rapply(state_structure, function(dimensions) {
                totsize <- prod(dimensions)
                out <- state[(counter):(counter - 1 + totsize)]$view(dimensions)
                counter <<- counter + totsize
                return(out)
            }, how = "list")
            return(state)
        }
        state_control <- self$unvectorize_state(self$vectorize_state(state), self$state_structure)
        vectorization_test <- unlist(mapply(function(x, y) {
            all.equal(as.array(x), as.array(y))
        }, x = state, y = state_control), recursive = T)
        if (!all(vectorization_test == "TRUE")) {
            stop("something went wrong with vectorization of state")
        }

        self$base_state_init <- as.array(self$vectorize_state(base_model$state_dict())) # TODO: storing it as an array to avoid an external pointer error when saving the model, fix it
        if (!is.function(base_model$fforward)) {
            stop("base_model must by supplied with working fforward function")
        }
        self$fforward <- base_model$fforward
        self$state_size <- self$vectorize_state(state)$size()
    },
    forward = function(x, base_state_diff, ...) {
        base_state_init <- torch_tensor(self$base_state_init)
        base_state <- base_state_init + base_state_diff
        base_state <- self$unvectorize_state(base_state, self$state_structure)
        y <- self$fforward(x, base_state, ...)
        return(y)
    }
)
