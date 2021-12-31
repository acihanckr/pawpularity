def train_alive(model, callbacks, fitting_params, history):
    history = model.fit(**fitting_params, callbacks = callbacks)
