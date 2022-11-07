import part1 as pt1
import part2 as pt2

df = pt1.read_input_data()
df_preprocessed, minmax_scaler = pt1.data_preprocessing(df)


def build_and_evaluate_model_target_data(experiment, df_full, scaler, location_id, value_type_id, input_shape,
                                         activation,
                                         layers_number, neurons_number, loss, optimizer_name, epochs, batch_size):
    df_section = pt1.get_data_by_location_and_value_type(df_full, location_id, value_type_id)

    input_values = pt1.get_input_values(df_section)
    input_values, target_values = pt2.apply_windows(input_values, pt1.window_size)

    input_values_train, target_values_train, input_values_test, target_values_test = pt2.split_train_test(input_values,
                                                                                                          target_values,
                                                                                                          pt1.split_percentage)

    trained_model = pt2.build_and_evaluate_model(experiment, input_values_train, target_values_train, input_values_test,
                                                 target_values_test, scaler, input_shape, activation, layers_number,
                                                 neurons_number, loss, optimizer_name, epochs, batch_size)

    return input_values_train, target_values_train, input_values_test, target_values_test


################# Experiment 1 #################
experiment1_settings = dict(location_id=116.0, value_type_id=11, activation='relu', layers_number=3,
                            neurons_number=[32, 16, 8, 4], loss='mean_squared_error', optimizer_name='Adam', epochs=200,
                            batch_size=32, window_size=pt1.window_size, split_percentage=pt1.split_percentage)

experiment1 = 'case1'
pt2.write_experiment_settings(experiment1, experiment1_settings)

input_values_train, target_values_train, input_values_test, target_values_test = build_and_evaluate_model_target_data(
    experiment1, df_preprocessed, minmax_scaler, experiment1_settings['location_id'],
    experiment1_settings['value_type_id'], experiment1_settings['window_size'], experiment1_settings['activation'],
    experiment1_settings['layers_number'], experiment1_settings['neurons_number'], experiment1_settings['loss'],
    experiment1_settings['optimizer_name'], experiment1_settings['epochs'], experiment1_settings['batch_size'])

################# Experiment 2 #################
experiment2_settings = dict(location_id=116.0, value_type_id=11, activation='sigmoid', layers_number=4,
                            neurons_number=[32, 16, 8, 4, 4], loss='mean_squared_error', optimizer_name='SGD', epochs=200,
                            batch_size=32, window_size=pt1.window_size, split_percentage=pt1.split_percentage)

experiment2 = 'case2'
pt2.write_experiment_settings(experiment2, experiment2_settings)

input_values_train, target_values_train, input_values_test, target_values_test = build_and_evaluate_model_target_data(
    experiment2, df_preprocessed, minmax_scaler, experiment2_settings['location_id'],
    experiment2_settings['value_type_id'], experiment2_settings['window_size'], experiment2_settings['activation'],
    experiment2_settings['layers_number'], experiment2_settings['neurons_number'], experiment2_settings['loss'],
    experiment2_settings['optimizer_name'], experiment2_settings['epochs'], experiment2_settings['batch_size'])

