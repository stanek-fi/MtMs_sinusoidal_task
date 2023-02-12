#' Create a mtms model
#'
#' The function `MtMs()` creates a mtms model based on the supplied `base_model` and other parameters specifying the meta and mesa modules.
#'
#' @param base_model Base torch model for construction of MtMs. It must have a `fforward(x, state)` method, that is, a functional, state-independent, version of the model. See the helper function `FFNN` for an example of what it might look like. It should also preferably already be trained on the pooled data.
#' @param num_tasks An integer specifying the number of tasks that will be used for training. It serves for pre-allocating appropriately sized matrices of parameters.
#' @param mesa_parameter_size An integer specifying the number of mesa parameters per task.
#' @param meta_bias A boolean specifying whether the meta module should contain a trainable constant on its output layer connecting to the base model.
#' @param meta_dropout_p A float specifying the probability of dropout in the meta module.
#' @param mesa_parameters_init A tensor of dimensions `mesa_parameter_size` x `num_tasks` containing the initial values of mesa parameters. If it is `NULL`, mesa parameters are initialized as zeros.
#' @param meta_weight_init A tensor of dimensions `num_non_orphaned_output_nodes` x `mesa_parameter_size` containing the initial values of the weights of the meta module. If it is `NULL`, the weights are initialized randomly.
#' @param meta_connections A string vector of names of parameters of the base model to be connected to the meta module. If it is `NULL`, the meta module connects to all parameters of the base model.
#'
#' @return A mtms model that can be trained and later used to deploy a mesa_model on new tasks by invoking its `MesaModel()` method.

MtMs <- nn_module(
    initialize = function(base_model, num_tasks, mesa_parameter_size = 1, meta_bias = T, meta_dropout_p = 0, mesa_parameters_init = NULL, meta_weight_init = NULL, meta_connections = NULL) {
        self$base_module <- BaseModule(base_model)
        self$meta_module <- MetaModule(self$base_module, mesa_parameter_size = mesa_parameter_size, meta_bias = meta_bias, meta_dropout_p = meta_dropout_p, meta_weight_init = meta_weight_init, meta_connections = meta_connections)
        self$mesa_module <- MesaModule(num_tasks, mesa_parameter_size, mesa_parameters_init)
    },
    forward = function(x, task, ...) {
        task_ids <- unique(as.array(task)) # TODO: add a check if tasks are within range
        res_y <- vector(mode = "list", length = length(task_ids))
        res_rows <- vector(mode = "list", length = length(task_ids))
        for (i in seq_along(task_ids)) {
            task_id <- task_ids[i]
            task_rows <- torch_squeeze(task) == task_id
            mesa_parameter <- self$mesa_module(task_id)
            base_state_diff <- self$meta_module(mesa_parameter)
            res_y[[i]] <- self$base_module(x[task_rows], base_state_diff, ...)
            res_rows[[i]] <- torch_nonzero(task_rows)
        }
        y <- torch_cat(res_y)
        rows <- torch_squeeze(torch_cat(res_rows))
        y <- y[torch_sort(rows)[[2]]]
        return(y)
    },
    MesaModel = function(mtms) {
        nn_module(
            initialize = function() {
                self$base_module <- mtms$base_module
                self$meta_module <- mtms$meta_module # TODO: meta_module is still affected by reference, fix it
                parameters <- self$meta_module$parameters
                for (i in seq_along(parameters)) {
                    parameters[[i]]$requires_grad <- FALSE
                }

                self$mesa_parameter <- nn_parameter(torch_tensor(rep(0, mtms$mesa_module$mesa_parameter_size), dtype = torch_float()))
            },
            forward = function(x, ...) {
                mesa_parameter <- self$mesa_parameter
                base_state_diff <- self$meta_module(mesa_parameter)
                y <- self$base_module(x, base_state_diff, ...)
                return(y)
            }
        )
    }
)
