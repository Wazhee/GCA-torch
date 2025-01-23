

def training_loop(n_classes=4):
  model = create_model(model, n_classes, image_shape)
#   model.compile(
#     optimizer = keras.optimizers.Adam(learning_rate),
#     loss = keras.losses.BinaryCrossentropy(),
#     metrics = keras.metrics.AUC(curve='ROC', name='auc', multi_label=True)
#   )
#   # Save model checkpoints based on validation loss
#   checkpoint = keras.callbacks.ModelCheckpoint(
#     os.path.join(MODEL_DIR, ckpt_dir, ckpt_name),
#     monitor = 'val_loss',
#     mode = 'min',
#     save_best_only = True
#   )
#   reduce_lr = keras.callbacks.ReduceLROnPlateau(
#     monitor = 'val_loss', 
#     factor = 0.5, 
#     patience = 5, 
#     min_lr = 1e-8)
#   callbacks = [checkpoint, reduce_lr]
#   # # Early stopping 
#   # if early_stopping:
#   #   callbacks += [
#   #     tf.keras.callbacks.EarlyStopping(
#   #       monitor = 'val_loss', 
#   #       patience = 10
#   #     )
#   #   ]
#   # Train model
#   logs = model.fit(
#     train_data, 
#     validation_data = val_data,
#     epochs = epochs, 
#     callbacks = callbacks,
#     use_multiprocessing = True
#   )
#   logs = pd.DataFrame(logs.history)
#   logs['epoch'] = np.arange(logs.shape[0])
#   logs = logs[['epoch', 'loss', 'auc', 'val_loss', 'val_auc']]
#   logs.to_csv(os.path.join(LOGS_DIR, ckpt_dir, f'{ckpt_name[:-5]}_logs.csv'), index=False)
    
    
if __name__ == "__main__":
    training_loop()

