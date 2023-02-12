#' Create a mesa_module
#'
#' An internal function that creates a mesa_module.
#'
#' @param num_tasks An integer specifying the number of tasks that will be used for training. It serves for pre-allocating appropriately sized matrices of parameters.
#' @param mesa_parameter_size An integer specifying the number of mesa parameters per task.
#' @param mesa_parameters_init A tensor of dimensions `mesa_parameter_size` x `num_tasks` containing the initial values of mesa parameters. If it is `NULL`, mesa parameters are initialized as zeros.
#'
#' @return A mesa_module to be used inside of MtMs.

MesaModule <- nn_module(
    initialize = function(num_tasks, mesa_parameter_size, mesa_parameters_init = NULL) {
        self$mesa_parameter_size <- mesa_parameter_size
        self$num_tasks <- num_tasks

        if (is.null(mesa_parameters_init)) {
            INIT_RANGE <- 0
            mesa_parameters_init <- matrix(runif(self$mesa_parameter_size * self$num_tasks, -INIT_RANGE, INIT_RANGE), nrow = self$mesa_parameter_size, ncol = self$num_tasks)
        } else {
            if (!((nrow(mesa_parameters_init) == self$mesa_parameter_size) & (ncol(mesa_parameters_init) == self$num_tasks))) {
                stop("ivalid supplied mesa_parameters_init")
            }
        }
        self$mesa_parameters <- nn_parameter(torch_tensor(mesa_parameters_init, dtype = torch_float()))
    },
    forward = function(task_id) {
        mesa_parameter <- self$mesa_parameters[, task_id]
        return(mesa_parameter)
    }
)
