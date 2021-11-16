def make_predictions(model, model_type, audio_features, prediction_scheme):
    """
    :param model:
    :type model:
    :param model_type:
    :type model_type:
    :param audio_features:
    :type audio_features:
    :param prediction_scheme:
    :type prediction_scheme:
    :return:
    :rtype:
    """
    # return: prediction_probability, final_prediction
    if model_type == 'conv':
        assigned_prob = model.predict(audio_features)[0][0]
        final_prediction = 1 * (assigned_prob >= 0.5)
    elif model_type == 'svm':
        final_prediction = model.predict(audio_features)
        pred = model.predict_proba(audio_features)[0]
        """
        We need to do this to reproduce the behaviour of a sigmoid function 
        We can always take the probability of class 1. 
        If prob_class_1 < prob_class_0, the whole prediction will naturally be nearer to class 0 
        since we will use an average of all the probability predictions
        """
        assigned_prob = pred[1]

    if prediction_scheme == 'majority':
        return int(final_prediction)
    else:
        return assigned_prob
        

