rm(list = ls())
library(yaml)
library(data.table)
library(ggplot2)
library(stringr)
library(torch)
library(latex2exp)
library(cowplot)
torch_set_num_threads(2)

par <- yaml.load_file(file.path("par.yaml"))
source_path <- file.path("src", "MtMs")
sapply(file.path(source_path, list.files(source_path)), source)
source_path <- file.path("src", "helpers")
sapply(file.path(source_path, list.files(source_path)), source)
temp_file_path <- file.path("temp")

torch_manual_seed(par$seed)
set.seed(par$seed)
start_time <- Sys.time()

# -- Data Generation ----------------------------------------------------------------------------------------------------------------------------------

x_range <- c(-5, 5)
A_range <- c(0.1, 5)
b_range <- c(0, pi)
f <- function(x, theta) {
    theta$A * sin(x + theta$b)
}
DGP <- function(K, theta, task) {
    x <- matrix(runif(K, x_range[1], x_range[2]), ncol = 1)
    y <- f(x, theta) + rnorm(nrow(x), 0, 0)
    task <- rep(task, nrow(x))
    return(list(
        y = y,
        x = x,
        task = task
    ))
}
thetas <- lapply(1:(par$M_in + par$M_out), function(x) {
    return(list(
        A = runif(1, A_range[1], A_range[2]),
        b = runif(1, b_range[1], b_range[2])
    ))
})
thetas_in <- thetas[1:par$M_in]
thetas_out <- thetas[-(1:par$M_in)]

train_in <- rbind_list(lapply(seq_along(thetas_in), function(i) DGP(par$K_train, thetas_in[[i]], i)))
test_in <- rbind_list(lapply(seq_along(thetas_in), function(i) DGP(par$K_test, thetas_in[[i]], i)))
validation_in <- rbind_list(lapply(seq_along(thetas_in), function(i) DGP(par$K_validation, thetas_in[[i]], i)))
train_out <- rbind_list(lapply(seq_along(thetas_out), function(i) DGP(par$K_train, thetas_out[[i]], i)))
test_out <- rbind_list(lapply(seq_along(thetas_out), function(i) DGP(par$K_test, thetas_out[[i]], i)))
validation_out <- rbind_list(lapply(seq_along(thetas_out), function(i) DGP(par$K_validation, thetas_out[[i]], i)))

criterion_vector <- function(y_pred, y) {
    (y_pred - y)^2
}
criterion <- function(y_pred, y) {
    mean(criterion_vector(y_pred, y))
}

# -- base_model  ----------------------------------------------------------------------------------------------------------------------------------

input_size <- 1
layer_sizes <- c(40, 40, 1)
layer_transforms <- c(lapply(seq_len(length(layer_sizes) - 1), function(x) nnf_relu), list(function(x) x))
base_model <- FFNN(input_size, layer_sizes, layer_transforms)

fit <- train_model(
    model = base_model,
    criterion = criterion,
    train = train_in[1:2],
    test = test_in[1:2],
    epochs = par$base_epochs,
    minibatch = par$base_minibatch,
    lr = par$base_lr,
    temp_file_path = temp_file_path,
    patience = 10
)

base_model <- fit$model
loss_base_model <- list(
    train = as.array(criterion(base_model(train_in$x), train_in$y)),
    test = as.array(criterion(base_model(test_in$x), test_in$y)),
    validation = as.array(criterion(base_model(validation_in$x), validation_in$y))
)

# -- mtms ----------------------------------------------------------------------------------------------------------------------------------

mtms <- MtMs(
    base_model = base_model,
    num_tasks = as.array(max(train_in$task)),
    mesa_parameter_size = par$mesa_parameter_size,
    meta_bias = par$meta_bias,
    meta_dropout_p = par$meta_dropout_p
)

fit <- train_model(
    model = mtms,
    criterion = criterion,
    train = train_in,
    test = test_in,
    epochs = par$mtms_epochs,
    minibatch = minibatch_generator(par$mtms_minibatch, task = train_in$task),
    lr = par$mtms_lr,
    temp_file_path = temp_file_path,
    patience = 10
)

mtms <- fit$model
loss_mtms <- list(
    train = as.array(criterion(mtms(train_in$x, train_in$task), train_in$y)),
    test = as.array(criterion(mtms(test_in$x, test_in$task), test_in$y)),
    validation = as.array(criterion(mtms(validation_in$x, validation_in$task), validation_in$y))
)

# -- mesa_model ----------------------------------------------------------------------------------------------------------------------------------

losses_mesa <- rep(NA, par$M_out)
states_mesa <- lapply(seq_len(par$M_out), function(x) NA)
for (m in seq_len(par$M_out)) {
    if (m %% 10 == 0) {
        print(str_c("m: ", m, " Time:", Sys.time()))
    }

    rows_train_m <- torch_squeeze(train_out$task == m)
    rows_validation_m <- torch_squeeze(validation_out$task == m)
    train_out_m <- list(
        y = train_out$y[rows_train_m],
        x = train_out$x[rows_train_m]
    )
    validation_out_m <- list(
        y = validation_out$y[rows_validation_m],
        x = validation_out$x[rows_validation_m]
    )

    mesa_model <- mtms$MesaModel(mtms)()

    fit <- train_model(
        model = mesa_model,
        criterion = criterion,
        train = train_out_m,
        epochs = par$mesa_epochs,
        lr = par$mesa_lr,
        optimizer_type = "adadelta",
        print_every = Inf
    )

    mesa_model <- fit$model
    losses_mesa[m] <- as.array(criterion(mesa_model(validation_out_m$x), validation_out_m$y))
    states_mesa[[m]] <- as.array(mesa_model$state_dict()$mesa_parameter)
}

loss_mesa_model <- list(
    validation = mean(losses_mesa),
    validation_int = qnorm(0.975) * sd(losses_mesa) / sqrt(length(losses_mesa))
)
print(loss_mesa_model)

# -- Outputs Generation -----------------------------------------------------------------

metrics <- list(
    loss_base_model = loss_base_model,
    loss_mtms = loss_mtms,
    loss_mesa_model = loss_mesa_model
)
print(metrics)
write_yaml(metrics, file.path("outputs", "metrics", "metrics.yaml"))
write.csv(data.frame(var = names(unlist(metrics)), value = unlist(metrics)), file.path("outputs", "metrics", "metrics.csv"), row.names = FALSE)

mesa_mat <- as.data.table(do.call(rbind, states_mesa))
colnames(mesa_mat) <- str_c("mesa_", seq_len(ncol(mesa_mat)))
theta_mat <- as.data.table(do.call(rbind, lapply(thetas_out, as.data.frame)))
mesa_mat_quant <- apply(mesa_mat, 2, function(x) quantile(x, c(seq(0.1, 0.9, 0.1))))
x_support <- as.matrix(seq(x_range[1], x_range[2], 0.05))
d <- do.call(rbind, lapply(seq_len(ncol(mesa_mat_quant)), function(mesa_i) {
    do.call(rbind, lapply(seq_len(nrow(mesa_mat_quant)), function(q) {
        mesa_state <- mesa_mat_quant["50%", ]
        mesa_state[mesa_i] <- mesa_mat_quant[q, mesa_i]

        mesa_model <- mtms$MesaModel(mtms)()
        state <- mesa_model$state_dict()
        state$mesa_parameter <- torch_tensor(mesa_state)
        mesa_model$load_state_dict(state)
        y_pred <- mesa_model(torch_tensor(x_support))

        data.table(
            mesa_par = names(mesa_mat)[mesa_i],
            mesa = round(mesa_state[mesa_i], 3),
            mesa_quantile = rownames(mesa_mat_quant)[q],
            x = as.vector(x_support),
            y = as.vector(as.matrix(y_pred))
        )
    }))
}))
d[, mesa := as.factor(mesa)]

p1 <- ggplot(d[mesa_par == "mesa_1"], aes(x = x, y = y, colour = mesa)) +
    geom_line() +
    labs(colour = TeX(r"($\theta_{1}$)"), y = TeX(r"($\hat{y}=f_{\omega}(x;[\theta_{1},\theta_{2}])$)"))
p2 <- ggplot(d[mesa_par == "mesa_2"], aes(x = x, y = y, colour = mesa)) +
    geom_line() +
    labs(colour = TeX(r"($\theta_{2}$)"), y = TeX(r"($\hat{y}=f_{\omega}(x;[\theta_{1},\theta_{2}])$)"))
p <- plot_grid(p1, p2, ncol = 1)
ggsave(file.path("outputs", "plots", str_c("prediction_functions_K_", par$K_train, ".pdf")), plot = p, height = 7, width = 7)

print(Sys.time() - start_time)
