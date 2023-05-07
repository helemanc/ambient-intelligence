def make_predictions(model, model_type, audio_features, prediction_scheme):
    """
    :param model: an instance of the classifier
    :type model: a Keras object or a scikit-learn object
    :param model_type: 'conv' if convolutional model is used, 'svm' if the SVM classifier is used
    :type model_type: string
    :param audio_features: array of features
    :type audio_features: np.array, float64
    :param prediction_scheme: a string representing the aggregation strategy that needs to be used in the ensemble
    :type prediction_scheme: string
    :return: the final prediction if 'prediction_scheme' is set to 'majority', the prediction in terms of probability
             otherwise
    :rtype: int or float64
    """
    # return: prediction_probability, final_prediction
    if model_type == 'conv':
        assigned_prob = model.predict(audio_features)[0][0]
        final_prediction = 1 * (assigned_prob >= 0.5)
    elif model_type == 'svm':
        pred = model.predict_proba(audio_features)[0]
        final_prediction = model.predict(audio_features)
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
        

