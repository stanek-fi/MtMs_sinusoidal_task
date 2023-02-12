train_model <- function(model, criterion, train, test = NULL, validation = NULL, epochs = 10, minibatch = Inf, temp_file_path = NULL, patience = 1, print_every = 1, lr = 0.001, weight_decay = 0, optimizer_type = "adam", lr_modifiers = NULL, ...) {
    model_last <- model

    out <- lapply(seq_along(lr), function(rs) {
        model <- model_last

        if (is.null(lr_modifiers)) {
            params <- model$parameters
        } else {
            if (length(lr_modifiers) == length(model$parameters)) {
                params <- lapply(seq_along(model$parameters), function(i) {
                    list("params" = model$parameters[i], "lr" = lr_modifiers[i] * lr[rs])
                })
            } else {
                stop("invalid lr_modifiers")
            }
        }
        optimizer <- switch(optimizer_type,
            "adam" = optim_adam(params, lr = lr[rs], weight_decay = weight_decay),
            "sgd" = optim_sgd(params, lr = lr[rs], weight_decay = weight_decay),
            "adadelta" = optim_adadelta(params, lr = lr[rs], weight_decay = weight_decay),
            "asgd" = optim_asgd(params, lr = lr[rs], weight_decay = weight_decay),
            "lbfgs" = optim_lbfgs(params, lr = lr[rs]),
            "rmsprop" = optim_rmsprop(params, lr = lr[rs], weight_decay = weight_decay),
            "rprop" = optim_rmsprop(params, lr = lr[rs])
        )

        if (is.numeric(minibatch)) {
            minibatch_sampler <- minibatch_generator(minibatch, num_rows = train[[1]]$size(1))
        } else {
            minibatch_sampler <- minibatch
        }

        log <- data.table(
            epoch = seq_len(epochs),
            loss_train = rep(Inf, epochs),
            loss_test = rep(Inf, epochs),
            loss_validation = rep(Inf, epochs)
        )

        for (e in 1:epochs) {
            mbs <- minibatch_sampler()

            if (e > 1) {
                model$train()
                for (mb in seq_along(mbs)) {
                    rows <- mbs[[mb]]
                    train_mb <- lapply(seq_along(train), function(i) train[[i]][rows])
                    optimizer$zero_grad()
                    y_pred_mb <- do.call(model, c(train_mb[-1], list(...)))
                    loss <- criterion(y_pred_mb, train_mb[[1]])
                    loss$backward()
                    optimizer$step()
                }
                model$eval()
            }

            log[e, loss_train := as.array(criterion(do.call(model, c(train[-1], list(...))), train[[1]]))]
            if (!is.null(test)) {
                log[e, loss_test := as.array(criterion(do.call(model, c(test[-1], list(...))), test[[1]]))]
            }
            if (!is.null(validation)) {
                log[e, loss_validation := as.array(criterion(do.call(model, c(validation[-1], list(...))), validation[[1]]))]
            }
            if (e %% print_every == 0) {
                print(str_c("lr:", lr[rs], " epoch:", e, " train:", round(log[e, loss_train], 5), " test:", round(log[e, loss_test], 5), " validation:", round(log[e, loss_validation], 5), " Time:", Sys.time()))
            }

            if (!is.null(test)) {
                ebest <- log[, which.min(loss_test)]
                if ((e == ebest) & !is.null(temp_file_path)) {
                    torch_save(model, file.path(temp_file_path, str_c("temp", ".t7")))
                }
                if (e - ebest >= patience) {
                    log <- log[1:e, ]
                    break()
                }
            }
        }

        if (!is.null(temp_file_path) & !is.null(test)) {
            model <- torch_load(file.path(temp_file_path, str_c("temp", ".t7")))
            file.remove(file.path(temp_file_path, str_c("temp", ".t7")))
        }

        model_last <<- model
        return(
            log
        )
    })

    model <- model_last
    log <- do.call(rbind, out)

    return(list(
        model = model,
        log = log
    ))
}
