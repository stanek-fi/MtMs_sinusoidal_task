rbind_list <- function(x) {
    out <- setNames(lapply(names(x[[1]]), function(q) {
        torch_tensor(do.call(rbind, lapply(x, function(element) {
            matrix(element[[q]])
        })))
    }), names(x[[1]]))
    return(out)
}
