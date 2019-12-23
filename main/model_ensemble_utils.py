import numpy as np


def predict_from_models(p_models, p_images):


    boolean_predictions_list = []
    numeric_predictions_list = []

    for model in p_models:

        numeric_predictions = model.predict(p_images)
        numeric_predictions_list.append(numeric_predictions)

        boolean_predictions = np.argmax(numeric_predictions, axis=1)
        boolean_predictions_list.append(boolean_predictions)


    hard_ensemble_preds = np.array([ max(set(p), key=p.count).item()  for p in list(zip( *[l for l in boolean_predictions_list] )) ])

    ensembled_probs  =  [  list(np.mean(p, axis=0) ) for p in list(zip( *[l for l in numeric_predictions_list] )) ]

    average_ensemble_preds = np.array([  np.argmax(p) for p in  ensembled_probs ])

    return { 'hard_ensemble_preds': hard_ensemble_preds, 'average_ensemble_preds': average_ensemble_preds, 'ensembled_probs': ensembled_probs }




def get_model_performance_estimate(p_model_fn, p_train_images, p_train_labels, p_val_images, p_val_labels,  p_test_images, p_test_labels):

    num_models = 1

    models = [None]*num_models
    val_accs = [None]*num_models
    test_accs = [None] * num_models
    for i in range(num_models):

        models[i] = train_model(p_model=p_model_fn(), p_train_images=p_train_images, p_train_labels=p_train_labels, p_val_images=p_val_images, p_val_labels=p_val_labels)
        val_accs[i] = evaluate_model(models[i], images=p_val_images, labels=p_val_labels)
        test_accs[i] = evaluate_model(models[i], images=p_test_images, labels=p_test_labels)

    average_accs = ( sum(val_accs) + sum(test_accs)  ) / ( len(val_accs) + len(test_accs) )

    print('Estimated accuracy of model population on val and test data is %s' % average_accs)

    return models