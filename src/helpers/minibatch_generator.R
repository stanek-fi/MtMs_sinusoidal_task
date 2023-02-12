minibatch_generator <- function(batch_size, task = NULL, num_rows = NULL) {
    if ("torch_tensor" %in% class(task)) {
        fun <- function() {
            task <- as.array(task)
            task_ids <- unique(task)
            mbs <- sample(seq_along(task_ids), replace = F)
            mbs <- split(mbs, ceiling(seq_along(mbs) / batch_size))
            mbs <- lapply(mbs, function(x) task_ids[x])
            mbs <- lapply(mbs, function(x) which(task %in% x))
            return(mbs)
        }
    } else if (is.numeric(num_rows)){
        fun <- function() {
            temp <- sample(seq_len(num_rows))
            mbs <- split(temp, ceiling(seq_along(temp) / batch_size))
            return(mbs)
        }
    } else {
        stop("either task or num_rows argument must be provided")
    }
    return(fun)
}
