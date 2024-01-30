if epoch % 10 == 0:
    train_source, train_true, train_pred = inference(train_loader, Net_model, Device)
    train_pred = clear_value_in_hole(train_pred, train_source, x_norm=x_normalizer)
    train_true = clear_value_in_hole(train_true, train_source, x_norm=x_normalizer)
    train_pred = y_normalizer.back(train_pred)
    train_true = y_normalizer.back(train_true)

    # for valid_loader in valid_loader_list:
    valid_source, valid_true, valid_pred = inference(valid_loader, Net_model, Device)
    valid_pred = clear_value_in_hole(valid_pred, valid_source, x_norm=x_normalizer)
    valid_true = clear_value_in_hole(valid_true, valid_source, x_norm=x_normalizer)
    valid_pred = y_normalizer.back(valid_pred)
    valid_true = y_normalizer.back(valid_true)

    train_abs_loss = Loss_real.abs(train_true, train_pred)
    train_rel_loss = Loss_real.rel(train_true, train_pred)
    valid_abs_loss = Loss_real.abs(valid_true, valid_pred)
    valid_rel_loss = Loss_real.rel(valid_true, valid_pred)

    if epoch > 0 and epoch % 100 == 0:
        for fig_id in range(15):
            fig, axs = plt.subplots(out_dim, 3, figsize=(18, 6), num=2)
            Visual.plot_fields_ms(fig, axs, train_true[fig_id], train_pred[fig_id], grids)
            fig.savefig(os.path.join(work_path, 'train_solution_' + str(fig_id) + '.jpg'))
            plt.close(fig)

        for fig_id in range(15):
            fig, axs = plt.subplots(out_dim, 3, figsize=(18, 6), num=3)
            Visual.plot_fields_ms(fig, axs, valid_true[fig_id], valid_pred[fig_id], grids)
            fig.savefig(os.path.join(work_path, 'valid_solution_' + str(fig_id) + '.jpg'))
            plt.close(fig)
